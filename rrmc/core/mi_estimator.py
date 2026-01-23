"""
Self-Revision Mutual Information Estimator for RRMC.

Implements the core uncertainty estimation via MI between
initial and revised model answers.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .llm import LLMWrapper
from .clustering import SemanticClusterer, DiversitySampler


@dataclass
class MIEstimate:
    """Result of MI estimation."""
    mi: float  # Estimated mutual information
    n_samples: int
    n_initial_clusters: int
    n_revised_clusters: int
    initial_answers: List[str]
    revised_answers: List[str]
    weights: Optional[List[float]] = None


# =============================================================================
# Prompt Templates for Self-Revision
# =============================================================================

# Initial answer prompts (task-specific)
INITIAL_PROMPT_DC = """Based on the case background and interrogation history, who is the murderer?

Case Background:
{initial_info}

Interrogation History:
{history}

Analyze the evidence and provide your answer in JSON format:
{{"suspect": "[A, B, C, D, or E]", "rationale": "[brief reasoning]"}}"""

INITIAL_PROMPT_SP = """Based on the clues gathered from the questions, what is the hidden story?

Initial Situation:
{surface}

Questions and Answers:
{history}

Provide your best explanation in JSON format:
{{"explanation": "[your explanation of the hidden story]"}}"""

INITIAL_PROMPT_GN = """Based on the previous guesses and feedback, what is the secret 4-digit number?

Rules: The secret is a 4-digit number with unique digits.
Previous guesses:
{history}

Provide your next guess in JSON format:
{{"guess": "[4 unique digits]", "reasoning": "[brief reasoning]"}}"""

# Revision prompts
REVISION_PROMPT_BASE = """You previously answered:
{initial_answer}

Now, carefully reconsider your answer. Look for:
1. Any contradictions in the evidence
2. Alternative hypotheses you might have missed
3. Weaknesses in your reasoning

After reconsidering, provide your revised answer in the same JSON format.
If your answer remains the same, explain why you're confident.

Revised answer:"""

REVISION_PROMPT_SKEPTICAL = """You previously answered:
{initial_answer}

ASSUME YOUR ANSWER MIGHT BE WRONG. Actively look for counter-evidence and alternative explanations.

Consider:
1. What evidence contradicts your current answer?
2. What would need to be true for a different answer to be correct?
3. Are there any gaps in your reasoning?

Provide your revised answer in the same JSON format, even if it differs from your original:"""

REVISION_PROMPT_ALTERNATIVE = """You previously answered:
{initial_answer}

Generate a PLAUSIBLE ALTERNATIVE explanation or answer. Consider what other answer could reasonably fit the evidence.

Then decide: is your original answer still the most likely, or should you change it?

Provide your final answer in the same JSON format:"""


class SelfRevisionMI:
    """
    Self-Revision Mutual Information Estimator.

    Estimates MI between initial and revised model answers
    as an uncertainty signal.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 6,
        temperature: float = 0.7,
        top_p: float = 0.95,
        diversity_sampler: Optional['DiversitySampler'] = None,
        regime: str = "normal",
    ):
        """
        Initialize MI estimator.

        Args:
            llm: LLM wrapper for generation
            clusterer: Semantic clusterer for answer grouping
            k_samples: Number of sample pairs for MI estimation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            diversity_sampler: Optional DiversitySampler for diversity-steered sampling
            regime: Decoding regime ("normal" or "homogeneous" for stress testing)
        """
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.temperature = temperature
        self.top_p = top_p
        self.diversity_sampler = diversity_sampler
        self.regime = regime

        # Homogeneous regime uses low temperature to induce mode collapse
        if regime == "homogeneous":
            self.temperature = 0.1
            self.top_p = 1.0

    def estimate(
        self,
        task_type: str,
        state: Dict[str, Any],
        revision_variant: str = "base",
    ) -> MIEstimate:
        """
        Estimate self-revision MI for a given state.

        Args:
            task_type: Task type (DC, SP, GN)
            state: Environment state dict
            revision_variant: Type of revision prompt (base, skeptical, alternative)

        Returns:
            MIEstimate with MI value and sample info
        """
        # Get appropriate prompts
        initial_prompt = self._get_initial_prompt(task_type, state)
        revision_prompt_template = self._get_revision_prompt(revision_variant)

        # Sample pairs - optionally with diversity sampling
        if self.diversity_sampler is not None:
            initial_answers, revised_answers, weights = self._sample_with_diversity(
                task_type, initial_prompt, revision_prompt_template
            )
        else:
            initial_answers, revised_answers, weights = self._sample_standard(
                task_type, initial_prompt, revision_prompt_template
            )

        # Cluster answers
        all_answers = initial_answers + revised_answers
        cluster_result = self.clusterer.cluster(all_answers, task_type)

        c0 = cluster_result.cluster_ids[:self.k_samples]
        c1 = cluster_result.cluster_ids[self.k_samples:]

        # Compute MI with optional weights
        mi = self._compute_mi(c0, c1, weights=weights)

        # Count unique clusters
        n_initial_clusters = len(set(c0))
        n_revised_clusters = len(set(c1))

        return MIEstimate(
            mi=mi,
            n_samples=self.k_samples,
            n_initial_clusters=n_initial_clusters,
            n_revised_clusters=n_revised_clusters,
            initial_answers=initial_answers,
            revised_answers=revised_answers,
            weights=weights,
        )

    def _sample_standard(
        self,
        task_type: str,
        initial_prompt: str,
        revision_prompt_template: str,
    ) -> Tuple[List[str], List[str], Optional[List[float]]]:
        """Standard sampling without diversity filtering (parallel)."""
        # Step 1: Generate k initial answers in parallel
        initial_responses = self.llm.sample_n(
            messages=[{"role": "user", "content": initial_prompt}],
            n=self.k_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=256,
            parallel=True,
        )
        initial_raw = [r.content for r in initial_responses]
        initial_answers = [self._parse_answer(y, task_type) for y in initial_raw]

        # Step 2: Build revision prompts for each initial answer
        revision_prompts = []
        for y0 in initial_raw:
            revision_prompt = revision_prompt_template.format(initial_answer=y0)
            full_revision_prompt = initial_prompt + "\n\n" + revision_prompt
            revision_prompts.append(full_revision_prompt)

        # Step 3: Generate k revised answers in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def generate_revision(prompt):
            return self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=256,
            )

        revised_responses = [None] * self.k_samples
        max_workers = getattr(self.llm, 'max_workers', 8)
        with ThreadPoolExecutor(max_workers=min(self.k_samples, max_workers)) as executor:
            futures = {executor.submit(generate_revision, p): i for i, p in enumerate(revision_prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    revised_responses[idx] = future.result()
                except Exception as e:
                    print(f"Revision sample {idx} failed: {e}")
                    from ..core.llm import LLMResponse
                    revised_responses[idx] = LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

        revised_answers = [self._parse_answer(r.content, task_type) for r in revised_responses]

        return initial_answers, revised_answers, None

    def _sample_with_diversity(
        self,
        task_type: str,
        initial_prompt: str,
        revision_prompt_template: str,
    ) -> Tuple[List[str], List[str], List[float]]:
        """Sampling with diversity rejection and importance weighting."""
        # Define sample functions for diversity sampler
        def sample_initial():
            response = self.llm.generate(
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=256,
            )
            return response.content

        # Get diverse initial samples (diversity sampler handles its own sampling)
        initial_raw, initial_weights = self.diversity_sampler.sample_diverse(
            sample_func=sample_initial,
            n=self.k_samples,
        )

        initial_answers = [self._parse_answer(y, task_type) for y in initial_raw]

        # Build revision prompts
        revision_prompts = []
        for y0 in initial_raw:
            revision_prompt = revision_prompt_template.format(initial_answer=y0)
            full_revision_prompt = initial_prompt + "\n\n" + revision_prompt
            revision_prompts.append(full_revision_prompt)

        # Generate revised answers in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def generate_revision(prompt):
            return self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=256,
            )

        revised_responses = [None] * len(initial_raw)
        max_workers = getattr(self.llm, 'max_workers', 8)
        with ThreadPoolExecutor(max_workers=min(len(initial_raw), max_workers)) as executor:
            futures = {executor.submit(generate_revision, p): i for i, p in enumerate(revision_prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    revised_responses[idx] = future.result()
                except Exception as e:
                    print(f"Revision sample {idx} failed: {e}")
                    from ..core.llm import LLMResponse
                    revised_responses[idx] = LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

        revised_answers = [self._parse_answer(r.content, task_type) for r in revised_responses]

        # Combine weights (use initial weights for the pairs)
        return initial_answers, revised_answers, initial_weights

    def _get_initial_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
        """Get initial answer prompt for task type."""
        if task_type == "DC":
            return INITIAL_PROMPT_DC.format(
                initial_info=state.get("initial_info", ""),
                history=state.get("history_string", ""),
            )
        elif task_type == "SP":
            return INITIAL_PROMPT_SP.format(
                surface=state.get("surface", ""),
                history=state.get("history_string", ""),
            )
        elif task_type == "GN":
            return INITIAL_PROMPT_GN.format(
                history=state.get("history_string", ""),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _get_revision_prompt(self, variant: str) -> str:
        """Get revision prompt template."""
        if variant == "skeptical":
            return REVISION_PROMPT_SKEPTICAL
        elif variant == "alternative":
            return REVISION_PROMPT_ALTERNATIVE
        else:
            return REVISION_PROMPT_BASE

    def _parse_answer(self, response: str, task_type: str) -> str:
        """Parse answer from LLM response."""
        # Try to extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if task_type == "DC":
                    return data.get("suspect", response)
                elif task_type == "SP":
                    return data.get("explanation", response)
                elif task_type == "GN":
                    return data.get("guess", response)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: return cleaned response
        return response.strip()[:500]

    def _compute_mi(
        self,
        c0: List[int],
        c1: List[int],
        weights: Optional[List[float]] = None,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute mutual information between cluster assignments.

        Args:
            c0: Cluster IDs for initial answers
            c1: Cluster IDs for revised answers
            weights: Importance weights (optional)
            eps: Smoothing epsilon

        Returns:
            Estimated MI in nats
        """
        if len(c0) == 0 or len(c1) == 0:
            return 0.0

        k = len(c0)
        if weights is None:
            weights = [1.0] * k

        # Build joint count table
        unique_a = sorted(set(c0))
        unique_b = sorted(set(c1))
        a_to_idx = {a: i for i, a in enumerate(unique_a)}
        b_to_idx = {b: i for i, b in enumerate(unique_b)}

        # Count with weights
        n_ab = np.zeros((len(unique_a), len(unique_b)))
        for i, (a, b) in enumerate(zip(c0, c1)):
            n_ab[a_to_idx[a], b_to_idx[b]] += weights[i]

        total = n_ab.sum()
        if total == 0:
            return 0.0

        # Compute marginals
        n_a = n_ab.sum(axis=1)
        n_b = n_ab.sum(axis=0)

        # Add smoothing
        n_ab_smooth = n_ab + eps
        n_a_smooth = n_a + eps * len(unique_b)
        n_b_smooth = n_b + eps * len(unique_a)
        total_smooth = total + eps * len(unique_a) * len(unique_b)

        # Compute MI
        mi = 0.0
        for i, a in enumerate(unique_a):
            for j, b in enumerate(unique_b):
                p_ab = n_ab_smooth[i, j] / total_smooth
                p_a = n_a_smooth[i] / total_smooth
                p_b = n_b_smooth[j] / total_smooth
                if p_ab > 0 and p_a > 0 and p_b > 0:
                    mi += p_ab * np.log(p_ab / (p_a * p_b))

        return max(0.0, mi)  # Ensure non-negative


class RobustMI:
    """
    Robust MI estimator using multiple prompt variants
    and diversity-steered sampling.

    Computes max MI over variants to resist mode collapse.
    """

    # Prompt variants
    VARIANTS = ["base", "skeptical", "alternative"]

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 6,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_diversity_sampling: bool = True,
        similarity_threshold: float = 0.90,
        aggregation: str = "max",  # max or quantile_75
        regime: str = "normal",
    ):
        """
        Initialize robust MI estimator.

        Args:
            llm: LLM wrapper
            clusterer: Semantic clusterer
            k_samples: Samples per variant
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_diversity_sampling: Whether to use diversity rejection sampling
            similarity_threshold: Threshold for diversity sampling
            aggregation: How to aggregate MI across variants (max or quantile_75)
            regime: Decoding regime ("normal" or "homogeneous" for stress testing)
        """
        self.use_diversity_sampling = use_diversity_sampling
        self.similarity_threshold = similarity_threshold
        self.aggregation = aggregation
        self.regime = regime

        # Create diversity sampler if enabled
        self.diversity_sampler = DiversitySampler(
            clusterer=clusterer,
            similarity_threshold=similarity_threshold,
        ) if use_diversity_sampling else None

        # Initialize MI estimator with optional diversity sampler and regime
        self.mi_estimator = SelfRevisionMI(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            temperature=temperature,
            top_p=top_p,
            diversity_sampler=self.diversity_sampler if use_diversity_sampling else None,
            regime=regime,
        )

    def estimate(
        self,
        task_type: str,
        state: Dict[str, Any],
        variants: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, MIEstimate]]:
        """
        Estimate robust MI using multiple variants.

        Args:
            task_type: Task type (DC, SP, GN)
            state: Environment state
            variants: Which prompt variants to use (default: all)

        Returns:
            Tuple of (robust_mi, dict of variant -> MIEstimate)
        """
        if variants is None:
            variants = self.VARIANTS

        mi_estimates = {}
        mi_values = []

        for variant in variants:
            estimate = self.mi_estimator.estimate(
                task_type=task_type,
                state=state,
                revision_variant=variant,
            )
            mi_estimates[variant] = estimate
            mi_values.append(estimate.mi)

        # Aggregate
        if self.aggregation == "max":
            robust_mi = max(mi_values) if mi_values else 0.0
        else:  # quantile_75
            robust_mi = np.percentile(mi_values, 75) if mi_values else 0.0

        return robust_mi, mi_estimates

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        """
        Get the best answer (majority from revised answers).

        Args:
            task_type: Task type
            state: Environment state

        Returns:
            Most common revised answer
        """
        _, estimates = self.estimate(task_type, state)

        # Collect all revised answers
        all_revised = []
        for est in estimates.values():
            all_revised.extend(est.revised_answers)

        if not all_revised:
            return ""

        # Cluster and find majority
        cluster_result = self.mi_estimator.clusterer.cluster(all_revised, task_type)

        # Find largest cluster
        max_size = 0
        best_cluster = 0
        for cid, size in cluster_result.cluster_sizes.items():
            if size > max_size:
                max_size = size
                best_cluster = cid

        return cluster_result.representative_texts.get(best_cluster, all_revised[0])


def compute_homogeneity_score(
    answers: List[str],
    clusterer: SemanticClusterer,
    task_type: str = "DC",
) -> float:
    """
    Compute homogeneity score (max cluster frequency).

    High homogeneity (near 1.0) indicates mode collapse.

    Args:
        answers: List of answer strings
        clusterer: Semantic clusterer
        task_type: Task type for clustering

    Returns:
        Homogeneity score in [0, 1]
    """
    if not answers:
        return 1.0

    cluster_result = clusterer.cluster(answers, task_type)
    if cluster_result.n_clusters == 0:
        return 1.0

    max_size = max(cluster_result.cluster_sizes.values())
    return max_size / len(answers)
