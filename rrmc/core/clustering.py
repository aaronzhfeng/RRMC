"""
Semantic Clustering for RRMC.

Provides clustering of LLM outputs for entropy/MI estimation.
Supports both embedding-based and entailment-aware clustering.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    cluster_ids: List[int]  # Cluster ID for each input text
    n_clusters: int
    cluster_sizes: Dict[int, int]  # Cluster ID -> size
    representative_texts: Dict[int, str]  # Cluster ID -> representative text


class SemanticClusterer:
    """
    Semantic clustering for LLM outputs.

    Uses sentence embeddings for similarity computation and
    agglomerative clustering for grouping.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        distance_threshold: float = 0.25,
        use_entailment: bool = False,
        nli_model: str = "roberta-large-mnli",
    ):
        """
        Initialize semantic clusterer.

        Args:
            model_name: Sentence transformer model name
            distance_threshold: Cosine distance threshold for clustering
            use_entailment: Whether to use NLI for SP task clustering
            nli_model: NLI model for entailment-aware clustering
        """
        self.distance_threshold = distance_threshold
        self.use_entailment = use_entailment

        # Load sentence transformer
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(model_name)
        else:
            self.embedder = None
            print("Warning: sentence-transformers not installed. Using fallback clustering.")

        # Load NLI model if needed
        self.nli_pipeline = None
        if use_entailment and HAS_TRANSFORMERS:
            try:
                self.nli_pipeline = pipeline("text-classification", model=nli_model)
            except Exception as e:
                print(f"Warning: Could not load NLI model: {e}")

    def cluster(
        self,
        texts: List[str],
        task_type: str = "DC",
    ) -> ClusterResult:
        """
        Cluster texts semantically.

        Args:
            texts: List of text strings to cluster
            task_type: Task type (DC, SP, GN) for task-specific handling

        Returns:
            ClusterResult with cluster assignments
        """
        if not texts:
            return ClusterResult(
                cluster_ids=[],
                n_clusters=0,
                cluster_sizes={},
                representative_texts={},
            )

        # For DC and GN, use exact matching on canonical forms
        if task_type in ("DC", "GN"):
            return self._cluster_discrete(texts, task_type)

        # For SP, use semantic clustering (optionally with entailment)
        return self._cluster_semantic(texts)

    def _cluster_discrete(self, texts: List[str], task_type: str) -> ClusterResult:
        """Cluster by exact canonical labels (DC suspects, GN digits)."""
        canonical = []
        for text in texts:
            if task_type == "DC":
                # Extract suspect letter or name
                canon = self._extract_dc_answer(text)
            else:  # GN
                # Extract 4-digit number
                canon = self._extract_gn_answer(text)
            canonical.append(canon)

        # Assign cluster IDs based on unique canonical forms
        unique_forms = list(set(canonical))
        form_to_id = {form: i for i, form in enumerate(unique_forms)}

        cluster_ids = [form_to_id[c] for c in canonical]
        cluster_sizes = {}
        for cid in cluster_ids:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        representative_texts = {
            form_to_id[form]: texts[canonical.index(form)]
            for form in unique_forms
        }

        return ClusterResult(
            cluster_ids=cluster_ids,
            n_clusters=len(unique_forms),
            cluster_sizes=cluster_sizes,
            representative_texts=representative_texts,
        )

    def _extract_dc_answer(self, text: str) -> str:
        """Extract DC answer (A-E or suspect name)."""
        text = text.strip()

        # Look for letter A-E
        match = re.search(r'\b([A-E])\b', text.upper())
        if match:
            return match.group(1)

        # Look for "suspect" keyword
        match = re.search(r'suspect[:\s]+(\d|[A-E])', text, re.I)
        if match:
            return match.group(1).upper()

        # Fallback: use first significant word
        words = text.split()
        return words[0][:20] if words else "UNKNOWN"

    def _extract_gn_answer(self, text: str) -> str:
        """Extract GN answer (4-digit number)."""
        digits = re.sub(r'[^0-9]', '', text)
        return digits[:4] if len(digits) >= 4 else digits.ljust(4, '0')

    def _cluster_semantic(self, texts: List[str]) -> ClusterResult:
        """Cluster using semantic embeddings."""
        if self.embedder is None:
            # Fallback: exact string matching
            return self._cluster_exact(texts)

        # Get embeddings
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Compute pairwise cosine distances
        # cosine_distance = 1 - cosine_similarity
        similarity = embeddings @ embeddings.T
        distance = 1 - similarity

        # Agglomerative clustering using single linkage
        cluster_ids = self._agglomerative_cluster(distance, self.distance_threshold)

        # If entailment is enabled, refine clusters
        if self.use_entailment and self.nli_pipeline:
            cluster_ids = self._refine_with_entailment(texts, cluster_ids)

        # Compute cluster statistics
        unique_ids = list(set(cluster_ids))
        id_remap = {old: new for new, old in enumerate(unique_ids)}
        cluster_ids = [id_remap[cid] for cid in cluster_ids]

        cluster_sizes = {}
        for cid in cluster_ids:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        # Find representative (first occurrence)
        representative_texts = {}
        for i, cid in enumerate(cluster_ids):
            if cid not in representative_texts:
                representative_texts[cid] = texts[i]

        return ClusterResult(
            cluster_ids=cluster_ids,
            n_clusters=len(unique_ids),
            cluster_sizes=cluster_sizes,
            representative_texts=representative_texts,
        )

    def _cluster_exact(self, texts: List[str]) -> ClusterResult:
        """Fallback: exact string matching."""
        normalized = [t.strip().lower() for t in texts]
        unique = list(set(normalized))
        text_to_id = {t: i for i, t in enumerate(unique)}

        cluster_ids = [text_to_id[n] for n in normalized]
        cluster_sizes = {}
        for cid in cluster_ids:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        representative_texts = {i: texts[normalized.index(u)] for i, u in enumerate(unique)}

        return ClusterResult(
            cluster_ids=cluster_ids,
            n_clusters=len(unique),
            cluster_sizes=cluster_sizes,
            representative_texts=representative_texts,
        )

    def _agglomerative_cluster(
        self,
        distance_matrix: np.ndarray,
        threshold: float,
    ) -> List[int]:
        """
        Simple agglomerative clustering with single linkage.

        Args:
            distance_matrix: Pairwise distance matrix
            threshold: Distance threshold for merging

        Returns:
            List of cluster IDs
        """
        n = len(distance_matrix)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Initialize each point as its own cluster
        cluster_ids = list(range(n))

        # Find connected components under threshold
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] <= threshold:
                    # Merge clusters
                    old_id = cluster_ids[j]
                    new_id = cluster_ids[i]
                    for k in range(n):
                        if cluster_ids[k] == old_id:
                            cluster_ids[k] = new_id

        return cluster_ids

    def _refine_with_entailment(
        self,
        texts: List[str],
        cluster_ids: List[int],
    ) -> List[int]:
        """
        Refine clusters using NLI to split contradictions.

        Args:
            texts: Original texts
            cluster_ids: Initial cluster assignments

        Returns:
            Refined cluster assignments
        """
        if not self.nli_pipeline:
            return cluster_ids

        # Group by cluster
        clusters: Dict[int, List[int]] = {}
        for i, cid in enumerate(cluster_ids):
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(i)

        new_cluster_ids = cluster_ids.copy()
        next_cluster_id = max(cluster_ids) + 1

        # Check each cluster for contradictions
        for cid, indices in clusters.items():
            if len(indices) <= 1:
                continue

            # Check pairwise entailment within cluster
            # Split if contradiction detected
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    text_i, text_j = texts[idx_i], texts[idx_j]

                    try:
                        # Check bidirectional entailment
                        result_ij = self.nli_pipeline(f"{text_i} [SEP] {text_j}")[0]
                        result_ji = self.nli_pipeline(f"{text_j} [SEP] {text_i}")[0]

                        # If contradiction detected, split
                        if (result_ij['label'] == 'CONTRADICTION' or
                            result_ji['label'] == 'CONTRADICTION'):
                            new_cluster_ids[idx_j] = next_cluster_id
                            next_cluster_id += 1
                    except Exception:
                        pass  # Keep original clustering on error

        return new_cluster_ids

    def compute_cluster_distribution(
        self,
        cluster_result: ClusterResult,
    ) -> Dict[int, float]:
        """
        Compute probability distribution over clusters.

        Args:
            cluster_result: Clustering result

        Returns:
            Dict mapping cluster ID to probability
        """
        total = sum(cluster_result.cluster_sizes.values())
        if total == 0:
            return {}
        return {
            cid: count / total
            for cid, count in cluster_result.cluster_sizes.items()
        }

    def compute_entropy(self, cluster_result: ClusterResult) -> float:
        """
        Compute semantic entropy over clusters.

        Args:
            cluster_result: Clustering result

        Returns:
            Entropy in nats
        """
        dist = self.compute_cluster_distribution(cluster_result)
        if not dist:
            return 0.0

        entropy = 0.0
        for p in dist.values():
            if p > 0:
                entropy -= p * np.log(p)
        return entropy

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        if self.embedder is None:
            # Return random embeddings as fallback
            return np.random.randn(len(texts), 384)
        return self.embedder.encode(texts, convert_to_numpy=True)


class DiversitySampler:
    """
    Diversity-steered sampling with importance reweighting.

    Implements rejection sampling based on embedding similarity
    to increase diversity under mode collapse.
    """

    def __init__(
        self,
        clusterer: SemanticClusterer,
        similarity_threshold: float = 0.90,
        max_tries: int = 5,
    ):
        """
        Initialize diversity sampler.

        Args:
            clusterer: Semantic clusterer for embeddings
            similarity_threshold: Max cosine similarity for acceptance
            max_tries: Max attempts before forced acceptance
        """
        self.clusterer = clusterer
        self.similarity_threshold = similarity_threshold
        self.max_tries = max_tries

    def sample_diverse(
        self,
        sample_func,
        n: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Sample with diversity rejection.

        Args:
            sample_func: Function that returns a single sample string
            n: Number of samples to collect

        Returns:
            Tuple of (samples, importance_weights)
        """
        samples = []
        weights = []
        embeddings = []

        for _ in range(n):
            for attempt in range(self.max_tries):
                candidate = sample_func()

                if self.clusterer.embedder is not None:
                    emb = self.clusterer.embedder.encode([candidate])[0]
                else:
                    emb = np.random.randn(384)

                # Check similarity to existing samples
                if len(embeddings) == 0:
                    accept = True
                else:
                    # Compute max cosine similarity
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                    max_sim = max(
                        np.dot(emb_norm, e / (np.linalg.norm(e) + 1e-8))
                        for e in embeddings
                    )
                    accept = max_sim <= self.similarity_threshold

                if accept or attempt == self.max_tries - 1:
                    samples.append(candidate)
                    # Weight inversely proportional to rejection rate
                    weight = 1.0 + attempt * 0.2
                    weights.append(weight)
                    embeddings.append(emb)
                    break

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight * len(weights) for w in weights]

        return samples, weights
