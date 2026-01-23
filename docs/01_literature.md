# Literature: Uncertainty-Aware Active Reasoning

A curated collection supporting research on improving LLM uncertainty self-awareness for active reasoning tasks.

**Last updated:** 2025-01-09  
**Total papers:** ~157 (deduplicated from ~170 raw entries)

---

## Quick Navigation

| Section | Papers | Focus |
|---------|--------|-------|
| [Seed Papers](#seed-papers) | 00-08 | Q1 Team + Benchmark + Hivemind |
| [**Foundational Methods (Q1 Replications)**](#foundational-methods-q1-replications) | 00-06 | **Team replication studies with method details** |
| [1. UQ Methods](#1-uncertainty-quantification-methods) | 09-41 | Black-box UQ, Semantic Entropy, Calibration |
| [2. Bayesian DL & LoRA](#2-bayesian-deep-learning--lora) | 42-59 | Bayesian Adapters, Ensembles, Foundations |
| [3. Prompt Perturbation](#3-prompt-perturbation--robustness) | 60-74 | Sensitivity, Ensembles, Robustness |
| [4. Conformal/Risk](#4-conformal-prediction--risk-control) | 75-90 | Coverage Guarantees, Abstention |
| [5. Active Reasoning](#5-active-reasoning--information-gathering) | 91-114 | Clarification, Question Asking, Agents |
| [6. Self-Knowledge](#6-llm-self-knowledge--metacognition) | 115-133 | Introspection, Internal States, Retrieval |
| [7. Diversity/Collapse](#7-output-diversity--mode-collapse) | 134-157 | Degeneration, Homogeneity, Decoding |

---

# Seed Papers

### 00. To Believe or Not to Believe Your LLM
- **Authors:** Yasin Abbasi-Yadkori, Ilja Kuzborskij, András György, Csaba Szepesvári
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2406.02543](https://arxiv.org/abs/2406.02543)
- **PDF:** `literature/00_To-Believe-or-Not-to-Believe.pdf`
- **Summary:** Proposes using **mutual information (MI)** between prompt-response pairs to quantify epistemic uncertainty in LLMs. Shows that MI provides a computable lower bound on task-relevant epistemic uncertainty through iterative prompting chains. Key insight: high MI indicates the model "knows it doesn't know."
- **Relevance:** **Aaron's Q1 focus**. Core uncertainty signal for active reasoning—high epistemic uncertainty should trigger information-seeking behavior. Implementation in `quantify_credibility/`.

---

### 01. Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling
- **Authors:** Bairu Hou, Yujian Liu, Kaizhi Qian, Jacob Andreas, Shiyu Chang, Yang Zhang
- **Year/Venue:** 2024, ICML
- **arXiv:** [2403.02509](https://arxiv.org/abs/2403.02509)
- **PDF:** `literature/01_Decomposing-Uncertainty-ICE.pdf`
- **Summary:** Introduces **Input Clarification Ensembling (ICE)** to separate aleatoric uncertainty (inherent input ambiguity) from epistemic uncertainty (model knowledge gaps). Generates multiple clarifications of underspecified inputs and ensembles predictions. Aleatoric uncertainty = variance across clarifications; epistemic = variance within each clarification.
- **Relevance:** **Calwin's Q1 focus**. Directly applicable to AR-Bench: if uncertainty is aleatoric (ambiguous problem), model should ask clarifying questions; if epistemic (knowledge gap), different strategy needed.

---

### 02. Laplace-LoRA: Bayesian Low-Rank Adaptation for Large Language Models
- **Authors:** Adam X. Yang, Maxime Robeyns, Xi Wang, Laurence Aitchison
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2308.13111](https://arxiv.org/abs/2308.13111)
- **PDF:** `literature/02_Laplace-LoRA.pdf`
- **Summary:** Applies **Laplace approximation** to LoRA adapters for Bayesian uncertainty in fine-tuned LLMs. Computes a Gaussian posterior over low-rank adapter weights using the Hessian at the MAP estimate. Enables uncertainty quantification with minimal overhead over standard LoRA.
- **Relevance:** **Pranav's Q1 focus** (compare Gaussian vs Laplace noise, Appendix E2). Provides principled uncertainty for fine-tuned models; compare with BLoB approach.

---

### 03. BLoB: Bayesian Low-Rank Adaptation by Backpropagation
- **Authors:** Yibin Wang, Haizhou Shi, Ligong Han, Dimitris Metaxas, Hao Wang
- **Year/Venue:** 2024, NeurIPS
- **OpenReview:** [p6jsTidUkPx](https://openreview.net/forum?id=p6jsTidUkPx)
- **PDF:** `literature/03_BLoB-Bayesian-LoRA.pdf`
- **Summary:** Alternative Bayesian LoRA approach that learns posterior over adapter weights via backpropagation rather than post-hoc Laplace. Uses variational inference with reparameterization trick. Claims better calibration than Laplace-LoRA on certain benchmarks.
- **Relevance:** **Pranav's Q1 focus**. Head-to-head comparison with Laplace-LoRA for uncertainty quality in active reasoning scenarios.

---

### 04. Prompt Perturbation Consistency Learning for Robust Language Models
- **Authors:** Yao Qiang, Subhrangshu Nandi, Ninareh Mehrabi, Greg Ver Steeg, Aram Galstyan, Cho-Jui Hsieh
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.15833](https://arxiv.org/abs/2402.15833)
- **PDF:** `literature/04_Prompt-Perturbation-Consistency.pdf`
- **Summary:** Proposes **PPCL framework** that regularizes KL divergence between predictions on clean vs perturbed prompts. Studies three perturbation types: oronyms (phonetically similar), synonyms (semantically similar), and paraphrasing. Shows LLMs are highly sensitive to minor prompt variations—a single word change can flip predictions.
- **Relevance:** **Brooklyn's Q1 focus**. Prompt sensitivity directly impacts uncertainty estimation reliability. If model gives different answers to paraphrased questions, sampling-based UQ may be unreliable.

---

### 05. Conformal Risk Control
- **Authors:** Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2212.13629](https://arxiv.org/abs/2212.13629)
- **PDF:** `literature/05_Conformal-Risk-Control.pdf`
- **Summary:** Generalizes conformal prediction from coverage control to **arbitrary bounded loss functions**. Provides distribution-free, finite-sample guarantees that expected loss is below a user-specified threshold. Enables risk control for any monotone loss (e.g., FNR, set size).
- **Relevance:** **Fong's Q1 focus**. Foundation for controlling error rates in active reasoning—e.g., guarantee that abstention rate or incorrect answer rate stays below threshold.

---

### 06. Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models
- **Authors:** Zhen Lin, Shubhendu Trivedi, Jimeng Sun
- **Year/Venue:** 2023, arXiv (TMLR under review)
- **arXiv:** [2305.19187](https://arxiv.org/abs/2305.19187)
- **PDF:** `literature/06_Generating-with-Confidence.pdf`
- **Summary:** Systematizes **black-box UQ for open-ended generation**, distinguishing "uncertainty" (dispersion over plausible outputs) from "confidence" (certainty in a chosen output). Proposes practical measures using only sampling access: semantic clustering, self-consistency, and degree matrix eigenvalues. No logits required.
- **Relevance:** Shared reference. Key framework for API-only LLM uncertainty. Many AR-Bench models are black-box; need methods that work without internals.

---

### 07. AR-Bench: From Passive to Active Reasoning
- **Authors:** Zhipeng Zhou, Jiashuo Liu, Zeming Wei, Yining Ye, Hanlin Zhang, Peng Cui
- **Year/Venue:** 2025, ICML
- **arXiv:** [2506.08295](https://arxiv.org/abs/2506.08295)
- **GitHub:** [tmlr-group/AR-Bench](https://github.com/tmlr-group/AR-Bench)
- **PDF:** `literature/07_AR-Bench.pdf`
- **Summary:** Introduces **Active Reasoning Benchmark** with three tasks requiring models to actively gather missing information: **Detective Cases** (multi-clue deduction), **Situation Puzzles** (lateral thinking with yes/no questions), **Guessing Numbers** (binary search). Key finding: even GPT-4o achieves only 35% on Guessing Numbers—models fail to ask strategic questions and often guess prematurely.
- **Relevance:** **Q2 Target Benchmark**. The goal is to improve AR-Bench performance via better uncertainty awareness. Local clone in `AR-Bench/`.

---

### 08. Artificial Hivemind: The Open-Ended Homogeneity of Language Models
- **Authors:** Kunhao Jiang, Ashton Anderson
- **Year/Venue:** 2025, NeurIPS (Oral, Datasets & Benchmarks)
- **arXiv:** [2510.22954](https://arxiv.org/abs/2510.22954)
- **PDF:** `literature/08_Artificial-Hivemind.pdf`
- **Summary:** Studies **mode collapse and homogeneity** in LLMs using Infinity-Chat dataset (26K queries, 31K human annotations). Reveals two phenomena: (1) **intra-model repetition**—single model generates similar responses to diverse prompts, (2) **inter-model homogeneity**—different models produce similar outputs. Crucially shows LMs are **less well-calibrated on outputs where human preferences diverge**.
- **Relevance:** Explains why LLMs get stuck in repetitive questioning loops on AR-Bench. Sampling-based uncertainty may underestimate true uncertainty when outputs homogenize. Must consider diversity when estimating uncertainty.

---

# Foundational Methods (Q1 Replications)

This section describes the core papers replicated and tested by our team in Q1, providing the methodological foundation for our Q2 proposal. Each paper was implemented, evaluated, and analyzed to understand its strengths and limitations for uncertainty-aware active reasoning.

---

## Aaron: Mutual Information for Epistemic Uncertainty (Paper 00)

**Paper:** *To Believe or Not to Believe Your LLM* (Abbasi-Yadkori et al., 2024)

### Core Method

The key insight is that **ground truth responses to a query should be independent**—if I ask "Name a city in the UK" multiple times, the correct answers (London, Manchester, Birmingham...) shouldn't depend on what was answered before. However, if an LLM is uncertain, its responses *do* depend on previous context.

**Iterative Prompting Protocol:**
1. Ask query $x$, get response $Y_1$
2. Prompt: "One answer to $x$ is $Y_1$. Give another answer." → get $Y_2$  
3. Continue to build pseudo-joint distribution $\widetilde{Q}(Y_1, \ldots, Y_n)$

**Theoretical Result:**
$$D_{\text{KL}}(\widetilde{Q}, \widetilde{P}) \geq I(\widetilde{Q})$$

The mutual information of the LLM's pseudo-joint distribution is a **computable lower bound** on its divergence from ground truth. High MI = high epistemic uncertainty.

**MI Estimator (Listing Algorithm):**
$$\widehat{I}_k(\gamma_1, \gamma_2) = \sum_{u \in S} \hat{\mu}(u) \log \frac{\hat{\mu}(u) + \gamma_1}{\hat{\mu}^{\otimes}(u) + \gamma_2}$$

with stabilization parameters $\gamma = 1/k$ to handle missing mass in finite samples.

### Key Finding: Probability Amplification

Repeating an answer in the prompt can "amplify" its probability:
- **Low epistemic uncertainty:** Model resists—"London" stays correct even with 100 repetitions of "Paris"
- **High epistemic uncertainty:** Model succumbs—repeated wrong answers flip the output

### Inspiration for Proposal

- **Self-revision MI:** We adapt the two-step iterative prompting ($n=2$) to a "reconsider your answer" framing
- **Robust MI:** We extend with prompt-variant ensembling and diversity-steered sampling to resist mode collapse
- **Implementation:** Full reproduction in `quantify_credibility/llm-belief-mi-test/`

---

## Calwin: Input Clarification Ensembling (Paper 01)

**Paper:** *Decomposing Uncertainty for LLMs through Input Clarification Ensembling* (Hou et al., 2024)

### Core Method

**Problem:** Standard UQ conflates two fundamentally different uncertainty types:
- **Aleatoric uncertainty:** Inherent input ambiguity (the question itself is underspecified)
- **Epistemic uncertainty:** Model knowledge gaps (the question is clear but model doesn't know)

**ICE Framework:**
1. Generate $M$ clarifications of the input: $\{c_1, \ldots, c_M\}$
2. For each clarification, sample $N$ responses
3. Decompose uncertainty:
   - **Aleatoric** = variance *across* clarifications (different interpretations → different answers)
   - **Epistemic** = variance *within* each clarification (same interpretation but model uncertain)

**Formula:**
$$\text{Total Uncertainty} = \underbrace{\mathbb{E}_c[\text{Var}(Y|c)]}_{\text{Epistemic}} + \underbrace{\text{Var}_c[\mathbb{E}(Y|c)]}_{\text{Aleatoric}}$$

### Key Finding

On ambiguous queries, ICE correctly attributes high uncertainty to aleatoric sources, while standard methods (entropy, self-consistency) cannot distinguish the cause.

### Inspiration for Proposal

- **Actionable uncertainty:** Aleatoric uncertainty → ask clarifying question; Epistemic uncertainty → gather more evidence or abstain
- **Clarification generation:** ICE's approach to generating alternative interpretations informs our prompt-variant ensembling
- **AR-Bench connection:** Detective Cases often have ambiguous clues—ICE helps decide whether to ask for clarification vs. gather more evidence

---

## Pranav: Bayesian LoRA Methods (Papers 02 & 03)

**Papers:** 
- *Laplace-LoRA* (Yang et al., 2024)
- *BLoB: Bayesian Low-Rank Adaptation by Backpropagation* (Wang et al., 2024)

### Core Methods

**Laplace-LoRA (Post-hoc Bayesian):**
1. Fine-tune standard LoRA adapter to get MAP estimate $\theta^*$
2. Compute Hessian $H = \nabla^2 \mathcal{L}(\theta^*)$ at the optimum
3. Approximate posterior: $p(\theta | D) \approx \mathcal{N}(\theta^*; H^{-1})$
4. Sample from posterior for predictive uncertainty

**BLoB (Variational Bayesian):**
1. Learn full posterior during training via variational inference
2. Parameterize $q(\theta) = \mathcal{N}(\mu, \sigma^2)$ with learnable $\mu, \sigma$
3. Use reparameterization trick: $\theta = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, 1)$
4. Optimize ELBO during fine-tuning

### Key Comparison (Pranav's Focus: Appendix E2)

| Aspect | Laplace-LoRA | BLoB |
|--------|--------------|------|
| **When** | Post-hoc (after training) | During training |
| **Noise type** | Gaussian (from Hessian) | Learned variational |
| **Compute** | One Hessian computation | Training overhead |
| **Calibration** | Good on in-distribution | Better on OOD |

### Inspiration for Proposal

- **Fine-tuning for uncertainty:** If we fine-tune for AR-Bench, Bayesian LoRA provides principled uncertainty without sacrificing efficiency
- **Uncertainty comparison:** Bayesian weight uncertainty vs. output-based MI—when do they agree/disagree?
- **Practical tradeoff:** Laplace-LoRA is simpler (no training change); BLoB is more principled but requires modified training

---

## Brooklyn: Prompt Perturbation Consistency (Paper 04)

**Paper:** *Prompt Perturbation Consistency Learning for Robust Language Models* (Qiang et al., 2024)

### Core Method

**Problem:** LLMs are extremely sensitive to minor prompt variations—a single word change can flip predictions, undermining sampling-based UQ.

**Three Perturbation Types:**
1. **Oronyms:** Phonetically similar words ("their" → "there")
2. **Synonyms:** Semantically similar words ("big" → "large")  
3. **Paraphrasing:** Full sentence rewording

**PPCL Framework:**
$$\mathcal{L}_{\text{PPCL}} = \mathcal{L}_{\text{task}} + \lambda \cdot D_{\text{KL}}(p(y|\tilde{x}) \| p(y|x))$$

Regularize the model to give consistent predictions under meaning-preserving perturbations.

### Key Finding

Without PPCL, a single synonym substitution can change accuracy by 10-20%. LLMs learn spurious correlations to surface forms rather than semantic content.

### Inspiration for Proposal

- **Prompt-variant ensembling:** If models are sensitive to phrasing, we should *exploit* this for uncertainty estimation—disagreement across paraphrases signals unreliable predictions
- **Robust MI:** Our $\widehat{I}_{\text{rob}} = \max_v \widehat{I}_v$ takes the maximum MI across prompt variants, using sensitivity as a feature rather than a bug
- **Evaluation robustness:** AR-Bench prompts should be tested under perturbations to ensure results aren't artifacts of specific phrasing

---

## Fong: Conformal Risk Control (Paper 05)

**Paper:** *Conformal Risk Control* (Angelopoulos et al., 2022)

### Core Method

**Standard Conformal Prediction:**
- Controls *coverage*: $P(Y \in C(X)) \geq 1 - \alpha$
- Limited to "is the true answer in the set?" guarantee

**Conformal Risk Control (Generalization):**
- Controls *arbitrary bounded losses*: $\mathbb{E}[L(C(X), Y)] \leq \alpha$
- Examples: FNR, set size, any monotone loss function

**Algorithm:**
1. Define loss function $L(\lambda)$ indexed by threshold $\lambda$
2. On calibration set, find $\hat{\lambda}$ such that empirical risk $\leq \alpha$ with high probability
3. Use $\hat{\lambda}$ at test time

**Key Property:** Distribution-free, finite-sample guarantee under exchangeability.

### Inspiration for Proposal

- **Risk-controlled thresholding:** We adapt this framework for ask/answer decisions—control error rate among states where we answer
- **Weaker claims:** We use Clopper-Pearson binomial UCB rather than full conformal guarantees, appropriate for interactive settings where exchangeability may not hold
- **Loss function design:** What's the right loss for active reasoning? Set size? Premature answer rate? We explore these tradeoffs

---

## Shared Reference: Black-box UQ Framework (Paper 06)

**Paper:** *Generating with Confidence* (Lin et al., 2023)

### Key Contributions

Systematizes black-box UQ for open-ended generation without logit access:
- **Semantic clustering:** Group equivalent answers before computing uncertainty
- **Self-consistency:** Agreement rate as confidence proxy
- **Degree matrix eigenvalues:** Graph-based uncertainty from pairwise similarity

### Relevance

Many AR-Bench models are API-only. This framework enables UQ without internal access—essential for practical deployment.

---

## Related Reference: Semantic Density (Paper 10)

**Paper:** *Semantic Density: Uncertainty Quantification for LLMs through Confidence Measurement in Semantic Space* (Qiu & Miikkulainen, 2024)

### Method

Measures uncertainty via "crowdedness" of the semantic neighborhood:
- Sample $k$ answers, embed in semantic space
- High density (many similar answers) = low uncertainty
- Low density (scattered answers) = high uncertainty

### Relevance

Alternative to semantic entropy that may be more computationally efficient—uses embedding distances rather than entailment classification.

---

# 1. Uncertainty Quantification Methods

### 09. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
- **Authors:** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar
- **Year/Venue:** 2023, ICLR (Spotlight)
- **arXiv:** [2302.09664](https://arxiv.org/abs/2302.09664)
- **PDF:** `literature/09_Semantic-Uncertainty.pdf`
- **GitHub:** —
- **Summary:** Introduces **semantic entropy**—uncertainty computed over *meanings* rather than surface forms. Clusters sampled generations by semantic equivalence (using bidirectional entailment) before computing entropy. Solves the "many ways to say the same thing" problem that inflates naive entropy estimates.
- **Relevance:** ⭐ **Core method**. Semantic entropy is robust to paraphrase variation. For AR-Bench, helps distinguish "model is uncertain about the answer" from "model has many ways to phrase the same answer."

---

### 10. Semantic Density: Uncertainty Quantification for Large Language Models through Semantic Density Estimation
- **Authors:** Xin Qiu, Risto Miikkulainen
- **Year/Venue:** 2024, NeurIPS
- **Link:** [NeurIPS Poster](https://neurips.cc/virtual/2024/poster/94057)
- **PDF:** `literature/10_Semantic-Density.pdf`
- **Summary:** Proposes uncertainty score based on how "crowded" the semantic neighborhood of sampled answers is. High density = many semantically similar answers = low uncertainty. Uses embedding space distances rather than entailment classification.
- **Relevance:** Alternative to semantic entropy that may be more computationally efficient. Useful when entailment model is expensive or unreliable.

---

### 11. Language Models (Mostly) Know What They Know
- **Authors:** Saurav Kadavath, Tom Conerly, Amanda Askell, et al.
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2207.05221](https://arxiv.org/abs/2207.05221)
- **PDF:** `literature/11_LMs-Know-What-They-Know.pdf`
- **Summary:** Establishes the influential **P(True) paradigm**: generate an answer, then ask the model "Is this answer correct?" and use the probability assigned to "True" as confidence. Shows models have meaningful self-knowledge when properly prompted. Also introduces P(IK) for "I know" calibration.
- **Relevance:** ⭐ **Foundational**. Self-evaluation is key to active reasoning—model needs to know when it doesn't know to ask questions. But Paper 08 (Hivemind) suggests this may be unreliable when preferences diverge.

---

### 12. Teaching Models to Express Their Uncertainty in Words
- **Authors:** Stephanie C. Lin, Jacob Hilton, Owain Evans
- **Year/Venue:** 2022, TMLR
- **arXiv:** [2205.14334](https://arxiv.org/abs/2205.14334)
- **PDF:** `literature/12_Teaching-Uncertainty-Words.pdf`
- **Summary:** Trains models to output **verbalized probability statements** ("I'm 70% confident...") that map to calibrated correctness likelihoods. Key finding: models can learn to express uncertainty in natural language without requiring logits, enabling uncertainty communication in black-box settings.
- **Relevance:** For active reasoning, verbalized uncertainty could trigger question-asking: "I'm not sure, let me ask..." Natural interface for uncertainty-aware agents.

---

### 13. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- **Authors:** Miao Xiong, Zhiyuan Hu, Xinyang Lu, et al.
- **Year/Venue:** 2024, ICLR
- **arXiv:** [2306.13063](https://arxiv.org/abs/2306.13063)
- **PDF:** `literature/13_Can-LLMs-Express-Uncertainty.pdf`
- **Summary:** Comprehensive empirical study of black-box confidence elicitation methods: (1) verbalized confidence prompts, (2) multi-sample consistency, (3) hybrid aggregation. Finds persistent **overconfidence** across models and shows when each method helps. Provides practical guidance.
- **Relevance:** ⭐ **Practical guide**. Systematic comparison of methods applicable to AR-Bench. Overconfidence is a key failure mode—models answer when they should ask.

---

### 14. Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?
- **Authors:** Gal Yona, Roee Aharoni, Mor Geva
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2405.16908](https://arxiv.org/abs/2405.16908)
- **PDF:** `literature/14_Faithful-Uncertainty-Expression.pdf`
- **Summary:** Studies **faithfulness** of verbalized uncertainty—do hedging expressions ("I think...") actually correlate with model ambiguity? Finds prompting alone often fails; models use hedges inconsistently. Gap between expressed and actual uncertainty.
- **Relevance:** Caution: verbalized confidence may not reflect true model state. For active reasoning, need methods that access actual uncertainty, not just what model says.

---

### 15. Calibration-Tuning: Teaching Large Language Models to Know What They Don't Know
- **Authors:** Sanyam Kapoor, Nate Gruver, Manley Roberts, Andrew Gordon Wilson
- **Year/Venue:** 2024, UncertaiNLP Workshop (ACL)
- **Link:** [ACL Anthology](https://aclanthology.org/2024.uncertainlp-1.1.pdf)
- **PDF:** `literature/15_Calibration-Tuning.pdf`
- **Summary:** Proposes fine-tuning protocol that trains LLMs to output **concept-level calibrated probabilities** over correctness. Uses graded correctness labels (not just binary) to teach nuanced confidence. Shows modest fine-tuning substantially improves calibration.
- **Relevance:** Training-time solution for better uncertainty. If base models are miscalibrated, fine-tuning for calibration before active reasoning deployment.

---

### 16. Large Language Models Must Be Taught to Know What They Don't Know
- **Authors:** Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Ilia Sucholutsky, Andrew Gordon Wilson
- **Year/Venue:** 2024, NeurIPS
- **arXiv:** [2406.08391](https://arxiv.org/abs/2406.08391)
- **PDF:** `literature/16_LLMs-Must-Be-Taught-IDK.pdf`
- **Summary:** Argues **prompting-only confidence is insufficient** for reliable uncertainty—must fine-tune. Shows that training on graded correctness examples substantially improves calibration and generalizes across tasks/domains. Key message: "I don't know" must be learned, not just prompted.
- **Relevance:** ⭐ **Key insight**. For AR-Bench, may need to fine-tune models to recognize when they lack information, not just prompt for uncertainty.

---

### 17. Self-Evaluation Improves Selective Generation in Large Language Models
- **Authors:** Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, Balaji Lakshminarayanan
- **Year/Venue:** 2023, NeurIPS Workshop
- **Link:** [PMLR](https://proceedings.mlr.press/v239/ren23a.html)
- **PDF:** `literature/17_Self-Evaluation-Selective-Generation.pdf`
- **Summary:** Recasts open-ended generation as token-level self-evaluation with explicit **"none of the above" (abstention)** option. Model predicts which token is correct or abstains. Produces better selective-generation risk-coverage than sequence probability baselines.
- **Relevance:** Abstention mechanism directly applicable to AR-Bench: model should abstain from answering and instead ask a question when uncertain.

---

### 18. Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs
- **Authors:** Jiefeng Chen, Jinsung Yoon, Sayna Ebrahimi, Sercan Arik, Tomas Pfister, Somesh Jha
- **Year/Venue:** 2023, Findings of EMNLP
- **arXiv:** [2310.11689](https://arxiv.org/abs/2310.11689)
- **PDF:** `literature/18_Selective-Prediction-Adaptation.pdf`
- **Summary:** Combines **parameter-efficient adaptation** with self-evaluation signals for better abstention/selective prediction. Fine-tunes small adapter on self-evaluation task, improving uncertainty-aware deployment for QA.
- **Relevance:** Efficient approach to add uncertainty awareness to existing models without full fine-tuning.

---

### 19. Calibrating Large Language Models with Sample Consistency
- **Authors:** Qing Lyu, Kumar Shridhar, Chaitanya Malaviya, Li Zhang, Yanai Elazar, Niket Tandon, Marianna Apidianaki, Mrinmaya Sachan, Chris Callison-Burch
- **Year/Venue:** 2025, AAAI
- **arXiv:** [2402.13904](https://arxiv.org/abs/2402.13904)
- **PDF:** `literature/19_Sample-Consistency-Calibration.pdf`
- **Summary:** Derives confidence from **consistency across multiple sampled generations**. Proposes several consistency metrics and shows strong post-hoc calibration. Analyzes effects of explanations, model scale, and instruction tuning on consistency-based calibration.
- **Relevance:** Practical method for AR-Bench: sample multiple reasoning paths, use agreement as confidence for when to answer vs ask.

---

### 20. Calibrating Long-form Generations from Large Language Models
- **Authors:** Yukun Huang, Yixin Liu, Raghuveer Thirukovalluru, Arman Cohan, Bhuwan Dhingra
- **Year/Venue:** 2024, Findings of EMNLP
- **arXiv:** [2402.06544](https://arxiv.org/abs/2402.06544)
- **PDF:** `literature/20_Calibrating-Long-Form.pdf`
- **Summary:** Addresses calibration for **long-form outputs where correctness is graded/partial**. Proposes distributional calibration framework and confidence elicitation via self-consistency and self-evaluation. Shows temperature scaling and fine-tuning help.
- **Relevance:** AR-Bench reasoning traces are long-form. Need calibration methods that handle partial correctness, not just binary right/wrong.

---

### 21. Linguistic Calibration of Long-Form Generations
- **Authors:** Neil Band, Xuechen Li, Tengyu Ma, Tatsunori Hashimoto
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2404.00474](https://arxiv.org/abs/2404.00474)
- **PDF:** `literature/21_Linguistic-Calibration-Long-Form.pdf`
- **Summary:** Defines **linguistic calibration** from decision-making perspective: does model text induce calibrated beliefs in users? Trains models to produce uncertainty statements ("~30% chance...") that match actual correctness rates.
- **Relevance:** For human-in-the-loop active reasoning, model's uncertainty expressions should help users make good decisions about when to trust answers.

---

### 22. LitCab: Lightweight Language Model Calibration over Short- and Long-form Responses
- **Authors:** Xin Liu, Muhammad Khalifa, Lu Wang
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2310.19208](https://arxiv.org/abs/2310.19208)
- **PDF:** `literature/22_LitCab-Lightweight-Calibration.pdf`
- **Summary:** Proposes **lightweight add-on module** (small parameter overhead) that adjusts logits to improve calibration across both short and long generations. Addresses limitations of naive temperature scaling.
- **Relevance:** Efficient post-hoc calibration that could be applied to any base model before AR-Bench evaluation.

---

### 23. Calibrating the Confidence of Large Language Models by Eliciting Fidelity
- **Authors:** Mozhi Zhang, Mianqiu Huang, Rundong Shi, Linsen Guo, Chong Peng, Peng Yan, Yaqian Zhou, Xipeng Qiu
- **Year/Venue:** 2024, EMNLP
- **Link:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.173.pdf)
- **PDF:** `literature/23_UF-Calibration-Fidelity.pdf`
- **Summary:** Decomposes confidence into uncertainty about the **question** vs **fidelity** to the produced answer. Proposes plug-and-play **UF Calibration** that addresses post-RLHF overconfidence by separating these components.
- **Relevance:** RLHF models are notoriously overconfident. UF calibration could help AR-Bench models know when they truly have an answer vs when they're confabulating.

---

### 24. Atomic Calibration of LLMs in Long-Form Generations
- **Authors:** Caiqi Zhang, Ruihan Yang, Zhisong Zhang, Xinting Huang, Dong Yu, Fei Huang, Yongbin Li
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2410.13246](https://arxiv.org/abs/2410.13246)
- **PDF:** `literature/24_Atomic-Calibration.pdf`
- **Summary:** Moves from response-level to **claim/atomic-level calibration** in long-form text. Breaks outputs into fine-grained claims and estimates uncertainty per claim. Enables analysis of how uncertainty varies across the generation trajectory.
- **Relevance:** For AR-Bench, uncertainty may vary within a reasoning trace—some steps certain, others not. Atomic calibration helps identify which steps need clarification.

---

### 25. CLUE: Concept-Level Uncertainty Estimation for Large Language Models
- **Authors:** Yu-Hsiang Wang, Chun-Liang Li, Yifan Peng, Jinwoo Ahn
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2409.03021](https://arxiv.org/abs/2409.03021)
- **PDF:** `literature/25_CLUE-Concept-Level-UE.pdf`
- **Summary:** Converts output into **concept-level representations** and estimates uncertainty per concept. Addresses mismatch between sequence-level scores and multi-fact outputs where only some facts may be uncertain.
- **Relevance:** Similar motivation to atomic calibration—AR-Bench answers may contain multiple claims with different uncertainty levels.

---

### 26. Conformal Language Modeling
- **Authors:** Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, Regina Barzilay
- **Year/Venue:** 2024, ICLR
- **arXiv:** [2306.10193](https://arxiv.org/abs/2306.10193)
- **PDF:** `literature/26_Conformal-Language-Modeling.pdf`
- **Summary:** Adapts conformal prediction to open-ended generation via **sampling-based stopping rule**. Returns a *set* of candidate generations with statistical coverage guarantees—at least one is correct with probability ≥ 1-α.
- **Relevance:** ⭐ **Key method**. For AR-Bench, conformal sets could indicate when model needs more information: large set = high uncertainty = ask question.

---

### 27. API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access
- **Authors:** Jiayuan Su, Jing Luo, Hongwei Wang, Lu Cheng
- **Year/Venue:** 2024, Findings of EMNLP
- **arXiv:** [2403.01216](https://arxiv.org/abs/2403.01216)
- **PDF:** `literature/27_API-Is-Enough-Conformal.pdf`
- **Summary:** Develops conformal prediction for **API-only LLMs** without logits. Builds nonconformity measures from sampled outputs using frequency and semantic similarity. Works for both closed- and open-ended QA.
- **Relevance:** Many AR-Bench models are API-only. This enables conformal guarantees without internal access.

---

### 28. ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees
- **Authors:** Zhiyuan Wang, Jinhao Duan, Lu Cheng, Yue Zhang, Qingni Wang, Hengtao Shen, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu
- **Year/Venue:** 2024, Findings of EMNLP
- **arXiv:** [2407.00499](https://arxiv.org/abs/2407.00499)
- **PDF:** `literature/28_ConU-Conformal-Uncertainty.pdf`
- **Summary:** Proposes self-consistency-derived uncertainty + conformal calibration to produce prediction sets with **user-specified correctness coverage** guarantees. Compact sets across multiple black-box LLMs and datasets.
- **Relevance:** Practical conformal method for LLMs. Could trigger AR-Bench question-asking when prediction set is large.

---

### 29. BayesFormer: Transformer with Uncertainty Estimation
- **Authors:** Karthik Abinav Sankararaman, Sinong Wang, Han Fang
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2206.00826](https://arxiv.org/abs/2206.00826)
- **PDF:** `literature/29_BayesFormer.pdf`
- **Summary:** Extends variational-inference-inspired dropout to Transformer architectures with Bayesian grounding. Enables epistemic uncertainty via **MC dropout** across language modeling and classification tasks.
- **Relevance:** Architectural approach to uncertainty. If fine-tuning for AR-Bench, could use BayesFormer-style uncertainty.

---

### 30. Transformer Uncertainty Estimation with Hierarchical Stochastic Attention
- **Authors:** Jiahuan Pei, Cheng Wang, György Szarvas
- **Year/Venue:** 2022, AAAI
- **Link:** [AAAI PDF](https://cdn.aaai.org/ojs/21364/21364-13-25377-1-2-20220628.pdf)
- **PDF:** `literature/30_Hierarchical-Stochastic-Attention.pdf`
- **Summary:** Makes attention **stochastic** (hierarchically) via Gumbel-Softmax sampling for principled predictive uncertainty while preserving performance. Architectural (not post-hoc) route to Transformer UQ.
- **Relevance:** Alternative architectural uncertainty approach. Attention patterns could also indicate what model is uncertain about.

---

### 31. How Certain is Your Transformer?
- **Authors:** Artem Shelmanov, Evgenii Tsymbalov, Dmitri Puzyrev, Kirill Fedyanin, Alexander Panchenko, Maxim Panov
- **Year/Venue:** 2021, EACL
- **Link:** [ACL Anthology](https://aclanthology.org/2021.eacl-main.157.pdf)
- **PDF:** `literature/31_How-Certain-Transformer.pdf`
- **Summary:** Early focused study of **MC dropout for Transformer uncertainty**. Evaluates how uncertainty scores help detect error-prone instances. Also proposes DPP-based variant to reduce cost of uncertainty estimates.
- **Relevance:** Baseline method and empirical insights on when dropout-based uncertainty works for Transformers.

---

### 32. Wat zei je? Detecting Out-of-Distribution Translations with Variational Transformers
- **Authors:** Tim Z. Xiao, Aidan N. Gomez, Yarin Gal
- **Year/Venue:** 2020, arXiv
- **arXiv:** [2006.08344](https://arxiv.org/abs/2006.08344)
- **PDF:** `literature/32_Variational-Transformers-OOD.pdf`
- **Summary:** Develops sequence-oriented uncertainty for Transformer NMT under dropout-based approximate Bayesian inference. Tackles intractability of naive uncertainty metrics on long discrete sequences.
- **Relevance:** Sequence-level uncertainty methods relevant to multi-step AR-Bench reasoning.

---

### 33. Bayesian Prompt Ensembles: Model Uncertainty Estimation for Black-box Large Language Models
- **Authors:** Francesco Tonolini, Jack M. McGowan, Phillip Stanley-Marbell
- **Year/Venue:** 2024, Findings of ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2024.findings-acl.728.pdf)
- **PDF:** `literature/33_Bayesian-Prompt-Ensembles.pdf`
- **Summary:** Introduces **BayesPE**—Bayesian treatment of prompt ensembles for black-box LLM uncertainty. Treats semantically equivalent prompts as approximate Bayesian "input layer," learning prompt weights via variational inference on validation set.
- **Relevance:** Principled way to combine prompt perturbation with uncertainty estimation. Connects to Paper 04 (PPCL).

---

### 34. Distinguishing the Knowable from the Unknowable with Language Models
- **Authors:** Gustaf Ahdritz, Tian Qin, Nikhil Vyas, Boaz Barak, Benjamin L. Edelman
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.03563](https://arxiv.org/abs/2402.03563)
- **PDF:** `literature/34_Knowable-vs-Unknowable.pdf`
- **Summary:** Targets the core epistemic problem: separating questions the model **can in principle answer** ("knowable") from those it **cannot** ("unknowable"). Proposes methods to distinguish these cases.
- **Relevance:** ⭐ **Key framing**. For AR-Bench, "unknowable" questions should trigger information-seeking; "knowable" questions should be answered.

---

### 35. Semantically Diverse Language Generation for Uncertainty Estimation in Language Models
- **Authors:** Lukas Aichberger, Kajetan Schweighofer, Sepp Hochreiter
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2406.04306](https://arxiv.org/abs/2406.04306)
- **PDF:** `literature/35_Semantic-Diverse-Generation-UQ.pdf`
- **Summary:** Proposes **steering decoding** to produce semantically diverse alternatives for better uncertainty over meanings. Addresses mode collapse issue where sampling gives near-duplicate answers.
- **Relevance:** Directly addresses Hivemind problem (Paper 08). If samples collapse, uncertainty is underestimated. Diversity steering helps.

---

### 36. Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling
- **Authors:** Ji Won Park, Kyunghyun Cho
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2510.21310](https://arxiv.org/abs/2510.21310)
- **PDF:** `literature/36_Diversity-Steered-Sampling.pdf`
- **Summary:** Introduces diversity-steered sampler that **penalizes semantic redundancy** during decoding, then corrects bias via importance reweighting. Improves sample-efficiency of semantic uncertainty estimates. Explicitly motivated by mode-collapse issues.
- **Relevance:** ⭐ **Solution to Hivemind**. Efficient way to get diverse samples for better uncertainty estimation despite model homogeneity.

---

### 37. LM-Polygraph: Uncertainty Estimation for Language Models
- **Authors:** Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, et al.
- **Year/Venue:** 2023, EMNLP Demo
- **arXiv:** [2311.07383](https://arxiv.org/abs/2311.07383)
- **PDF:** `literature/37_LM-Polygraph.pdf`
- **Summary:** Engineering-centric **framework implementing many UQ methods** for text generation with unified evaluation interfaces. Enables reproducible comparison and practical integration of UQ signals in LLM systems.
- **Relevance:** **Practical toolkit**. Use for implementing and comparing UQ methods on AR-Bench.

---

### 38. Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- **Authors:** Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2406.15627](https://arxiv.org/abs/2406.15627)
- **PDF:** `literature/38_Benchmarking-UQ-LM-Polygraph.pdf`
- **Summary:** Benchmark suite for consistent evaluation of UQ baselines and confidence normalization methods across text generation tasks. Helps standardize methodology.
- **Relevance:** Evaluation methodology for AR-Bench UQ experiments.

---

### 39. Revisiting Uncertainty Estimation and Calibration of Large Language Models
- **Authors:** Linwei Tao, Younan Zhu, Haolan Zhan, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2505.23854](https://arxiv.org/abs/2505.23854)
- **PDF:** `literature/39_Revisiting-UQ-Calibration.pdf`
- **Summary:** Large-scale empirical study across many LLM families comparing black-box uncertainty signals. Analyzes how post-training, reasoning modes, scale, and quantization affect calibration and selectivity.
- **Relevance:** Empirical insights on which UQ methods work for which models/settings.

---

### 40. A Survey of Uncertainty Estimation Methods on Large Language Models
- **Authors:** Ziyang Xia, Puhan Zhang, Wenhao Huang, Shibo Hao, Zheng Wang
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2503.00172](https://arxiv.org/abs/2503.00172)
- **PDF:** `literature/40_Survey-UQ-LLMs.pdf`
- **Summary:** Dedicated survey organizing LLM UQ methods: black-box elicitation, semantic methods, ensembles, calibration, evaluation. Useful for filling gaps around metrics, datasets, methodological pitfalls.
- **Relevance:** **Reference survey**. Comprehensive overview of the UQ landscape.

---

### 41. FLUE: Streamlined Uncertainty Estimation for Large Language Models
- **Authors:** Shiqi Gao, Yichuan Wang, Qibing Ren, Junyang Lin, Jingren Zhou, Chang Zhou
- **Year/Venue:** 2025, AAAI
- **DOI:** 10.1609/aaai.v39i16.33840
- **PDF:** `literature/41_FLUE-Streamlined-UQ.pdf`
- **Summary:** Targets **lower-overhead UQ** by producing token-level stochasticity in streamlined way and learning mapping from token uncertainty to sequence uncertainty. Reduces cost of multi-sample UQ.
- **Relevance:** Efficiency matters for AR-Bench since models may need many interactions. Fast UQ enables more question-asking rounds.

---

# 2. Bayesian Deep Learning & LoRA

### 42. C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models
- **Authors:** Amir Hossein Rahmati, et al.
- **Year/Venue:** 2025, NeurIPS
- **Link:** [OpenReview](https://openreview.net/forum?id=siPeAstQLq)
- **PDF:** `literature/42_C-LoRA-Contextual.pdf`
- **Summary:** Proposes **input-dependent (contextual) LoRA modules** where uncertainty estimates vary with example characteristics. Improves calibration and robustness in few-shot/data-scarce regimes.
- **Relevance:** Uncertainty that adapts to input context—useful for AR-Bench where different questions may have different inherent difficulty.

---

### 43. ScalaBL: Scalable Bayesian Low-Rank Adaptation via Stochastic Variational Subspace Inference
- **Authors:** Colin Samplawski, et al.
- **Year/Venue:** 2025, UAI
- **Link:** [PMLR](https://proceedings.mlr.press/v286/samplawski25a.html)
- **PDF:** `literature/43_ScalaBL-Stochastic-Variational.pdf`
- **Summary:** Performs Bayesian inference in r-dimensional subspace aligned with LoRA rank. Enables stochastic variational inference with **extremely small parameter overhead** while scaling to larger LLMs.
- **Relevance:** Scalable Bayesian uncertainty for large models. If fine-tuning for AR-Bench, can maintain uncertainty awareness efficiently.

---

### 44. Improving LoRA with Variational Learning
- **Authors:** Bai Cong, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2506.14280](https://arxiv.org/abs/2506.14280)
- **PDF:** `literature/44_Variational-LoRA-IVON.pdf`
- **Summary:** Uses IVON optimizer + posterior pruning for variational Bayesian LoRA at billion-parameter scale. Aims to reduce overhead/fragility of earlier Bayesian LoRA approaches.
- **Relevance:** Modern variational approach for Bayesian adapters.

---

### 45. Minimal Ranks, Maximum Confidence: Parameter-efficient Uncertainty Quantification for LoRA
- **Authors:** Patryk Marszałek, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2502.12122](https://arxiv.org/abs/2502.12122)
- **PDF:** `literature/45_Minimal-Ranks-UQ-LoRA.pdf`
- **Summary:** Bayesian LoRA variant with **tighter parameter budgets** while retaining calibration benefits. Keeps effective stochastic parameter space very small.
- **Relevance:** When parameter budget is limited but need uncertainty.

---

### 46. Training-Free Bayesianization for Low-Rank Adapters of Large Language Models
- **Authors:** Haizhou Shi, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2412.05723](https://arxiv.org/abs/2412.05723)
- **PDF:** `literature/46_Training-Free-Bayesianization.pdf`
- **Summary:** Converts trained LoRA to Bayesian adapter **without additional training** by searching for acceptable posterior variance within constrained low-rank Gaussian family.
- **Relevance:** Post-hoc Bayesianization—add uncertainty to existing fine-tuned models.

---

### 47. Gaussian Stochastic Weight Averaging for Bayesian Low-Rank Adaptation
- **Authors:** Emre Onal, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2405.03425](https://arxiv.org/abs/2405.03425)
- **PDF:** `literature/47_SWAG-LoRA.pdf`
- **Summary:** Combines LoRA with **SWAG** to form Gaussian posterior over adapter weights from fine-tuning trajectory. Simple tooling, improved calibration.
- **Relevance:** Practical approach using established SWAG method with LoRA.

---

### 48. Uncertainty Quantification in Fine-tuned LLMs using LoRA Ensembles
- **Authors:** Oleksandr Balabanov, Hampus Linander
- **Year/Venue:** 2025, ICLR Workshop
- **arXiv:** [2402.12264](https://arxiv.org/abs/2402.12264)
- **PDF:** `literature/48_LoRA-Ensembles-UQ.pdf`
- **Summary:** **Ensembles of LoRA adapters** for uncertainty—computationally efficient since only adapters differ across ensemble members.
- **Relevance:** Ensemble uncertainty with minimal overhead. Good baseline for AR-Bench.

---

### 49. LoRA Ensembles for Large Language Model Fine-tuning
- **Authors:** Xi Wang, Laurence Aitchison, Maja Rudolph
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2310.00035](https://arxiv.org/abs/2310.00035)
- **PDF:** `literature/49_LoRA-Ensembles.pdf`
- **Summary:** Deep-ensemble style uncertainty made practical by ensembling only lightweight LoRA adapters. Targets calibration and OOD uncertainty.
- **Relevance:** Foundation for LoRA ensemble approaches.

---

### 50. Amortized Bayesian Meta-Learning for Low-Rank Adaptation
- **Authors:** Yunyi Zhang, et al.
- **Year/Venue:** 2025, UncertaiNLP Workshop
- **Link:** [Workshop PDF](https://uncertainlp.github.io/2025/papers/82_paper.pdf)
- **PDF:** `literature/50_Amortized-Bayesian-Meta-LoRA.pdf`
- **Summary:** Meta-learning to **predict Bayesian posteriors** for LoRA adapters across tasks. Faster uncertainty-aware adaptation at deployment.
- **Relevance:** Quick uncertainty estimation for new tasks.

---

### 51. Bayesian Mixture of Experts for Large Language Models
- **Authors:** Abdelrahman Dialameh, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2511.08968](https://arxiv.org/abs/2511.08968)
- **PDF:** `literature/51_Bayesian-MoE-LLM.pdf`
- **Summary:** Bayesian ideas for MoE-style LLMs using structured Laplace approximation over expert-layer weights. Uncertainty influences expert routing.
- **Relevance:** For MoE models used in AR-Bench.

---

### 52. Towards Scalable Bayesian Transformers: Stochastic Subset Selection for NLP
- **Authors:** Peter Johannes Tejlgaard Kampen, et al.
- **Year/Venue:** 2024, UAI
- **Link:** [PMLR](https://proceedings.mlr.press/v244/kampen24a.html)
- **PDF:** `literature/52_Scalable-Bayesian-Transformers.pdf`
- **Summary:** Partially stochastic Bayesian transformers via LA/SWAG with **stochastic subset selection**. Reduces memory/compute while keeping UQ quality.
- **Relevance:** Scalable Bayesian methods for large transformers.

---

### 53. Bayesian Transformer Language Models for Speech Recognition
- **Authors:** Boyang Xue, et al.
- **Year/Venue:** 2021, arXiv
- **arXiv:** [2102.04754](https://arxiv.org/abs/2102.04754)
- **PDF:** `literature/53_Bayesian-Transformer-LM-Speech.pdf`
- **Summary:** Full Bayesian learning for Transformer LMs using variational inference over multiple components (attention, FFN, embeddings).
- **Relevance:** Full-model Bayesian treatment reference.

---

### 54. Uncertainty Estimation of Transformer Predictions for Misclassification Detection
- **Authors:** Artem Vazhentsev, et al.
- **Year/Venue:** 2022, ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2022.acl-long.566.pdf)
- **PDF:** `literature/54_UE-Transformer-Misclassification.pdf`
- **Summary:** Large empirical study of UE methods for transformer **misclassification detection** on text classification + NER. Proposes efficient UE modifications.
- **Relevance:** Misclassification detection relevant to knowing when to abstain.

---

### 55. Uncertainty-Aware Natural Language Inference with Stochastic Weight Averaging
- **Authors:** Aarne Talman, et al.
- **Year/Venue:** 2023, NoDaLiDa
- **Link:** [ACL Anthology](https://aclanthology.org/2023.nodalida-1.37.pdf)
- **PDF:** `literature/55_Uncertainty-NLI-SWA.pdf`
- **Summary:** SWA/SWAG for transformer-based NLI. Studies relationship between model uncertainty and **human annotation disagreement**.
- **Relevance:** NLI used in semantic entropy clustering—uncertainty in entailment affects semantic uncertainty.

---

### 56. Laplace Redux — Effortless Bayesian Deep Learning
- **Authors:** Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig
- **Year/Venue:** 2021, NeurIPS
- **arXiv:** [2106.14806](https://arxiv.org/abs/2106.14806)
- **PDF:** `literature/56_Laplace-Redux.pdf`
- **Summary:** ⭐ **Core reference** for Laplace approximations in deep learning. Covers scalable Hessian approximations and practical variants. Foundation for Laplace-LoRA.
- **Relevance:** Theoretical and practical foundation for Laplace-based uncertainty.

---

### 57. A Simple Baseline for Bayesian Uncertainty in Deep Learning (SWAG)
- **Authors:** Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson
- **Year/Venue:** 2019, NeurIPS
- **arXiv:** [1902.02476](https://arxiv.org/abs/1902.02476)
- **PDF:** `literature/57_SWAG.pdf`
- **Summary:** ⭐ **Foundational**. Introduces SWAG (SWA-Gaussian)—scalable Gaussian posterior from SGD trajectories. Widely reused in LoRA+SWAG variants.
- **Relevance:** Core method underlying many Bayesian LoRA approaches.

---

### 58. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- **Authors:** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell
- **Year/Venue:** 2017, NeurIPS
- **arXiv:** [1612.01474](https://arxiv.org/abs/1612.01474)
- **PDF:** `literature/58_Deep-Ensembles.pdf`
- **Summary:** ⭐ **Foundational**. Establishes deep ensembles as high-performing, easy baseline for predictive uncertainty. Later adapted to LLMs via LoRA ensembling.
- **Relevance:** Ensemble uncertainty baseline; adapted via LoRA for efficiency.

---

### 59. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
- **Authors:** Yarin Gal, Zoubin Ghahramani
- **Year/Venue:** 2016, ICML
- **arXiv:** [1506.02142](https://arxiv.org/abs/1506.02142)
- **PDF:** `literature/59_MC-Dropout-Bayesian.pdf`
- **Summary:** ⭐ **Foundational**. Justifies MC dropout as approximate variational Bayesian inference. Underpins many transformer/LLM uncertainty methods.
- **Relevance:** Theoretical foundation for dropout-based uncertainty in transformers.

---

# 3. Prompt Perturbation & Robustness

### 60. Detecting Hallucinations in Large Language Models using Semantic Entropy
- **Authors:** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal
- **Year/Venue:** 2024, Nature
- **DOI:** [10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0)
- **PDF:** `literature/60_Hallucination-Semantic-Entropy.pdf`
- **Summary:** ⭐ **High-impact**. Extends semantic entropy for **hallucination detection** (specifically "confabulations"). Shows uncertainty signals can robustly flag unreliable generations. Published in Nature.
- **Relevance:** Hallucination detection is critical for AR-Bench—model shouldn't confidently give wrong answers.

---

### 61. CAPE: Calibrating Language Models via Augmented Prompt Ensembles
- **Authors:** Mingjian Jiang, et al.
- **Year/Venue:** 2023, ICML Workshop
- **Link:** [OpenReview](https://openreview.net/forum?id=L0dc4wqbNs)
- **PDF:** `literature/61_CAPE-Augmented-Prompt-Ensembles.pdf`
- **Summary:** Uses prompt augmentations (template paraphrases, option permutations) as efficient **prompt ensemble** for calibration without additional training.
- **Relevance:** Prompt ensemble approach for calibration. Could combine with AR-Bench prompting.

---

### 62. Strength in Numbers: Estimating Confidence by Prompt Agreement
- **Authors:** Gwenyth Portillo Wightman, et al.
- **Year/Venue:** 2023, TrustNLP Workshop
- **DOI:** [10.18653/v1/2023.trustnlp-1.28](https://doi.org/10.18653/v1/2023.trustnlp-1.28)
- **PDF:** `literature/62_Prompt-Agreement-Confidence.pdf`
- **Summary:** Agreement across multiple prompt rephrasings yields **better-calibrated confidence** than single-prompt scores.
- **Relevance:** Multi-prompt agreement as confidence signal for when to answer vs ask.

---

### 63. Robust and Cheap Hallucination Detection in LLMs
- **Authors:** Jannik Kossen, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2406.15927](https://arxiv.org/abs/2406.15927)
- **PDF:** `literature/63_Cheap-Hallucination-Detection.pdf`
- **Summary:** Efficient approximations of semantic-entropy-style uncertainty for **practical hallucination detection** with reduced compute.
- **Relevance:** Efficient uncertainty estimation important for AR-Bench with many interactions.

---

### 64. FormatSpread: Quantifying LLM Sensitivity to Spurious Features in Prompt Design
- **Authors:** Melanie Sclar, et al.
- **Year/Venue:** 2024, ICLR
- **Link:** [OpenReview](https://openreview.net/forum?id=RIu5lyNXjT)
- **PDF:** `literature/64_FormatSpread-Prompt-Sensitivity.pdf`
- **Summary:** Shows LLMs extremely sensitive to **meaning-preserving formatting changes**. Introduces FormatSpread to characterize performance intervals across plausible formats.
- **Relevance:** Format sensitivity affects evaluation reliability. AR-Bench results may vary with prompt formatting.

---

### 65. POSIX: A Prompt Sensitivity Index for Large Language Models
- **Authors:** Anwoy Chatterjee, et al.
- **Year/Venue:** 2024, Findings of EMNLP
- **arXiv:** [2410.02185](https://arxiv.org/abs/2410.02185)
- **PDF:** `literature/65_POSIX-Prompt-Sensitivity-Index.pdf`
- **Summary:** Prompt sensitivity metric based on **log-likelihood changes** under intent-preserving variants. Enables systematic sensitivity comparisons.
- **Relevance:** Metric for quantifying how sensitive AR-Bench models are to prompt variations.

---

### 66. On the Worst Prompt Performance of Large Language Models
- **Authors:** Bowen Cao, et al.
- **Year/Venue:** 2024, NeurIPS
- **arXiv:** [2406.10248](https://arxiv.org/abs/2406.10248)
- **PDF:** `literature/66_Worst-Prompt-Performance.pdf`
- **Summary:** Argues robustness should consider **worst-case over semantically equivalent prompts**. Introduces RobustAlpacaEval. Large gaps between best vs worst prompt.
- **Relevance:** Worst-case robustness matters for reliable AR-Bench performance.

---

### 67. What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering
- **Authors:** Federico Errica, et al.
- **Year/Venue:** 2025, NAACL
- **arXiv:** [2406.12334](https://arxiv.org/abs/2406.12334)
- **PDF:** `literature/67_Sensitivity-Consistency-Metrics.pdf`
- **Summary:** Two metrics: **sensitivity** (prediction instability across rephrasings) and **consistency** (within-class stability). Diagnoses prompt brittleness beyond accuracy.
- **Relevance:** Metrics for understanding AR-Bench prompt robustness.

---

### 68. PromptSET: Benchmarking Prompt Sensitivity in Large Language Models
- **Authors:** Amirhossein Razavi, et al.
- **Year/Venue:** 2025, ECIR
- **arXiv:** [2502.06065](https://arxiv.org/abs/2502.06065)
- **PDF:** `literature/68_PromptSET-Benchmark.pdf`
- **Summary:** Benchmark and "prompt sensitivity prediction" task. Existing approaches struggle to predict when rephrasing breaks behavior.
- **Relevance:** Predicting sensitivity failures could help AR-Bench robustness.

---

### 69. Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs
- **Authors:** Andong Hua, et al.
- **Year/Venue:** 2025, EMNLP
- **Link:** [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1006.pdf)
- **PDF:** `literature/69_Prompt-Sensitivity-Artifact.pdf`
- **Summary:** When does sensitivity reflect true brittleness vs evaluation artifacts? Sharpens interpretation of prompt-variation results.
- **Relevance:** Understanding what sensitivity results mean for AR-Bench evaluation.

---

### 70. Paraphrase Types Elicit Prompt Engineering Capabilities
- **Authors:** Jan Philip Wahle, et al.
- **Year/Venue:** 2024, EMNLP
- **arXiv:** [2406.19898](https://arxiv.org/abs/2406.19898)
- **PDF:** `literature/70_Paraphrase-Types-Capabilities.pdf`
- **Summary:** Analyzes which **paraphrase types** (morphology, syntax, lexicon, discourse) most affect LLM behavior across tasks.
- **Relevance:** Understanding which perturbations matter for AR-Bench uncertainty estimation.

---

### 71. Open (Clinical) LLMs are Sensitive to Instruction Phrasings
- **Authors:** Alberto Mario Ceballos Arroyo, et al.
- **Year/Venue:** 2024, BioNLP @ ACL
- **arXiv:** [2407.09429](https://arxiv.org/abs/2407.09429)
- **PDF:** `literature/71_Clinical-LLM-Instruction-Sensitivity.pdf`
- **Summary:** Substantial performance variance across **instruction phrasings** with fairness shifts under non-adversarial rewordings.
- **Relevance:** Domain-specific sensitivity analysis.

---

### 72. Sensitivity and Robustness of LLMs to Prompt Template in Japanese Text Classification
- **Authors:** Chengguang Gan, et al.
- **Year/Venue:** 2023, PACLIC
- **Link:** [PACLIC PDF](https://paclic2023.github.io/downloads/PACLIC_37/PACLIC_37_paper_5.pdf)
- **PDF:** `literature/72_Japanese-Prompt-Robustness.pdf`
- **Summary:** Large performance swings from template/sentence-structure changes in Japanese. GPT-4 included.
- **Relevance:** Multilingual prompt sensitivity.

---

### 73. Calibrate Before Use: Improving Few-Shot Performance of Language Models
- **Authors:** Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh
- **Year/Venue:** 2021, ICML
- **arXiv:** [2102.09690](https://arxiv.org/abs/2102.09690)
- **PDF:** `literature/73_Calibrate-Before-Use.pdf`
- **Summary:** ⭐ **Foundational**. Few-shot prompting is **high-variance** w.r.t. format/example choice/order. Introduces **contextual calibration** to reduce variance.
- **Relevance:** Contextual calibration could help stabilize AR-Bench few-shot prompting.

---

### 74. Rationale-Augmented Ensembles in Language Models
- **Authors:** Xuezhi Wang, et al.
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2207.00747](https://arxiv.org/abs/2207.00747)
- **PDF:** `literature/74_Rationale-Augmented-Ensembles.pdf`
- **Summary:** Robustness via ensembling over **diverse rationales** (structured perturbation in reasoning space).
- **Relevance:** Ensemble reasoning for AR-Bench—multiple reasoning paths could indicate uncertainty.

---

# 4. Conformal Prediction & Risk Control

### 75. Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees
- **Authors:** Yu Gui, et al.
- **Year/Venue:** 2024, NeurIPS
- **arXiv:** [2405.10301](https://arxiv.org/abs/2405.10301)
- **PDF:** `literature/75_Conformal-Alignment.pdf`
- **Summary:** Conformalized selection/certification framework filtering outputs so selected subset satisfies **alignment criterion with distribution-free guarantees**.
- **Relevance:** Trust guarantees for AR-Bench output selection.

---

### 76. Selective Generation for Controllable Language Models
- **Authors:** Yoonho Lee, et al.
- **Year/Venue:** 2024, NeurIPS (Spotlight)
- **arXiv:** [2307.09254](https://arxiv.org/abs/2307.09254)
- **PDF:** `literature/76_Selective-Generation.pdf`
- **Summary:** ⭐ **Key reference**. Selective generation where model **abstains when uncertain**. Explicit answer/abstain tradeoffs.
- **Relevance:** Core paradigm for AR-Bench: abstain from answering → ask question instead.

---

### 77. Mitigating LLM Hallucinations via Conformal Abstention
- **Authors:** Yasin Abbasi-Yadkori, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2405.01563](https://arxiv.org/abs/2405.01563)
- **PDF:** `literature/77_Conformal-Abstention-Hallucination.pdf`
- **Summary:** Conformal abstention policies controlling **error/hallucination rates** under exchangeability. Answer or abstain with guarantees.
- **Relevance:** Hallucination control for AR-Bench reliability.

---

### 78. C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models
- **Authors:** Mintong Kang, et al.
- **Year/Venue:** 2024, ICML
- **arXiv:** [2402.03181](https://arxiv.org/abs/2402.03181)
- **PDF:** `literature/78_C-RAG-Certified-Risk.pdf`
- **Summary:** ⭐ **Key method**. Conformal risk framework for RAG with **certified upper bounds on generation risk**.
- **Relevance:** If AR-Bench uses retrieval, certified risk control applies.

---

### 79. TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction
- **Authors:** Shuo Li, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2307.04642](https://arxiv.org/abs/2307.04642)
- **PDF:** `literature/79_TRAQ-Trustworthy-RAG.pdf`
- **Summary:** Conformal prediction sets for RAG with **end-to-end correctness guarantees**.
- **Relevance:** RAG-based active reasoning with guarantees.

---

### 80. Conformal Prediction with Large Language Models for Multi-Choice Question Answering
- **Authors:** Bhawesh Kumar, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2305.18404](https://arxiv.org/abs/2305.18404)
- **PDF:** `literature/80_Conformal-MCQA.pdf`
- **Summary:** Conformal prediction for LLM **multiple-choice QA** with distribution-free coverage.
- **Relevance:** Some AR-Bench tasks involve multiple choice scenarios.

---

### 81. SConU: Selective Conformal Uncertainty in Large Language Models
- **Authors:** Zhiyuan Wang, et al.
- **Year/Venue:** 2025, ACL
- **arXiv:** [2504.14154](https://arxiv.org/abs/2504.14154)
- **PDF:** `literature/81_SConU-Selective-Conformal.pdf`
- **Summary:** Extends conformal to **selective settings** via conformal p-values for distributional mismatch detection.
- **Relevance:** Detecting when AR-Bench inputs are OOD.

---

### 82. COIN: Uncertainty-Guarding Selective Question Answering with Provable Risk Guarantees
- **Authors:** Zhiyuan Wang, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2506.20178](https://arxiv.org/abs/2506.20178)
- **PDF:** `literature/82_COIN-Selective-QA-Guarantees.pdf`
- **Summary:** Selecting **single answer per question** under FDR-style risk guarantees.
- **Relevance:** Single-answer guarantees for AR-Bench final answers.

---

### 83. Prune 'n Predict: Optimizing LLM Decision-making with Conformal Prediction
- **Authors:** Harit Vishwakarma, et al.
- **Year/Venue:** 2025, ICML
- **arXiv:** [2501.00555](https://arxiv.org/abs/2501.00555)
- **PDF:** `literature/83_Prune-n-Predict.pdf`
- **Summary:** Conformal prediction to **prune low-confidence options** in multi-choice decisions while maintaining guarantees.
- **Relevance:** Pruning uncertain options in AR-Bench decision-making.

---

### 84. Conformal Tail Risk Control for Large Language Model Alignment
- **Authors:** Catherine Yu-Chi Chen, et al.
- **Year/Venue:** 2025, ICML
- **arXiv:** [2502.20285](https://arxiv.org/abs/2502.20285)
- **PDF:** `literature/84_Conformal-Tail-Risk.pdf`
- **Summary:** Controls **tail events** (toxic outputs) via distortion risk measures from loss quantiles.
- **Relevance:** Controlling worst-case AR-Bench failures.

---

### 85. Learning Conformal Abstention Policies for Adaptive Risk Management
- **Authors:** Sina Tayebati, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2502.06884](https://arxiv.org/abs/2502.06884)
- **PDF:** `literature/85_Learnable-Conformal-Abstention.pdf`
- **Summary:** **RL-driven adaptive conformal thresholds** for utility/accuracy tradeoffs.
- **Relevance:** Learning when to abstain/ask for AR-Bench.

---

### 86. Multi-group Uncertainty Quantification for Long-form Text Generation
- **Authors:** Terrance Liu, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2407.21057](https://arxiv.org/abs/2407.21057)
- **PDF:** `literature/86_Multigroup-UQ-Long-Form.pdf`
- **Summary:** Claim-level calibration + conformal with **multicalibration for group-wise validity**.
- **Relevance:** Group-conditional guarantees across AR-Bench task types.

---

### 87. Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality
- **Authors:** Naihe Feng, et al.
- **Year/Venue:** 2025, SIGIR
- **arXiv:** [2506.20978](https://arxiv.org/abs/2506.20978)
- **PDF:** `literature/87_Conformal-RAG.pdf`
- **Summary:** Conformal prediction with RAG-internal signals for response/claim quality guarantees.
- **Relevance:** Quality guarantees for RAG-based AR-Bench approaches.

---

### 88. Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction
- **Authors:** Debashish Chakraborty, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2511.17908](https://arxiv.org/abs/2511.17908)
- **PDF:** `literature/88_Principled-Context-Engineering-RAG.pdf`
- **Summary:** Conformal for **pre-generation context selection** in RAG.
- **Relevance:** Selecting what information to gather for AR-Bench.

---

### 89. Trust or Escalate: LLM Judges with Provable Guarantees
- **Authors:** Jung, et al.
- **Year/Venue:** 2025, ICLR
- **Link:** [OpenReview](https://openreview.net/forum?id=gjeQKFxFpZ)
- **PDF:** `literature/89_Trust-or-Escalate.pdf`
- **Summary:** Calibrated decision rules for **accept answer vs escalate** (defer to humans/tools).
- **Relevance:** Decision to answer vs ask question in AR-Bench.

---

### 90. Robots That Ask For Help: Uncertainty Alignment for LLM Planners
- **Authors:** Allen Z. Ren, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2307.01928](https://arxiv.org/abs/2307.01928)
- **PDF:** `literature/90_Robots-Ask-For-Help.pdf`
- **Summary:** Conformal-based framework for LLM planner to **request assistance** with task completion guarantees.
- **Relevance:** ⭐ **Directly relevant**. Exactly the ask-for-help paradigm needed for AR-Bench.

---

# 5. Active Reasoning & Information Gathering

### 91. LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models
- **Authors:** Marwa Abdulhai, et al.
- **Year/Venue:** 2025, ICML
- **Link:** [PMLR](https://proceedings.mlr.press/v267/abdulhai25a.html)
- **PDF:** `literature/91_LMRL-Gym.pdf`
- **Summary:** Multi-turn interaction benchmarks + RL framework for training/evaluating LLM agents on **clarifying questions and information gathering**.
- **Relevance:** Training framework for AR-Bench-style multi-turn reasoning.

---

### 92. ACT: Learning to Clarify via Action-Based Contrastive Self-Training
- **Authors:** Maximillian Chen, et al.
- **Year/Venue:** 2025, ICLR
- **arXiv:** [2406.00222](https://arxiv.org/abs/2406.00222)
- **PDF:** `literature/92_ACT-Learning-to-Clarify.pdf`
- **Summary:** ⭐ **Key method**. Data-efficient preference training for multi-turn dialogue—targets failures where agents **guess instead of asking**.
- **Relevance:** Directly addresses AR-Bench failure mode of premature guessing.

---

### 93. ECLAIR: Enhanced Clarification for Interactive Responses
- **Authors:** John Murzaku, et al.
- **Year/Venue:** 2025, AAAI
- **arXiv:** [2503.15739](https://arxiv.org/abs/2503.15739)
- **PDF:** `literature/93_ECLAIR-Interactive-Disambiguation.pdf`
- **Summary:** End-to-end interactive disambiguation: generate clarification questions, ingest feedback, resolve ambiguity.
- **Relevance:** Full clarification pipeline for AR-Bench.

---

### 94. Learning to Clarify by Reinforcement Learning Through Reward-Weighted Fine-Tuning
- **Authors:** Subhojyoti Mukherjee, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2506.06964](https://arxiv.org/abs/2506.06964)
- **PDF:** `literature/94_RL-Clarifying-Questions.pdf`
- **Summary:** Trains QA agents to ask clarifying questions via **reward-weighted fine-tuning**.
- **Relevance:** RL approach for learning when to ask in AR-Bench.

---

### 95. MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning
- **Authors:** Shuyue Stella Li, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2406.00922](https://arxiv.org/abs/2406.00922)
- **PDF:** `literature/95_MediQ-Interactive-Clinical.pdf`
- **Summary:** ⭐ **Related benchmark**. Interactive clinical QA where expert decides **when to ask follow-up vs diagnose**. Naive prompting to "ask questions" can hurt.
- **Relevance:** Domain-specific active reasoning benchmark with similar structure to AR-Bench.

---

### 96. Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in LLMs
- **Authors:** Zhiyuan Hu, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.03271](https://arxiv.org/abs/2402.03271)
- **PDF:** `literature/96_UoT-Uncertainty-of-Thoughts.pdf`
- **Summary:** ⭐ **Core method**. **UoT**: uncertainty-aware planning that simulates futures and uses **information-gain rewards** to choose questions that reduce uncertainty.
- **Relevance:** Directly applicable to AR-Bench—choose questions that maximize expected information gain.

---

### 97. Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain
- **Authors:** Davide Mazzaccara, et al.
- **Year/Venue:** 2024, Findings of EMNLP
- **Link:** [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.291.pdf)
- **PDF:** `literature/97_EIG-Informative-Questions.pdf`
- **Summary:** Uses **Expected Information Gain (EIG)** in 20 Questions + DPO to learn more informative questions.
- **Relevance:** ⭐ **Key method**. EIG-based question selection directly applicable to AR-Bench Guessing Numbers.

---

### 98. Melon: Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search
- **Authors:** Yifei Yuan, et al.
- **Year/Venue:** 2024, WWW
- **DOI:** [10.1145/3589334.3645483](https://doi.org/10.1145/3589334.3645483)
- **PDF:** `literature/98_Melon-Multimodal-Clarification.pdf`
- **Summary:** Multimodal (text+image) clarifying questions dataset and model.
- **Relevance:** Extension to multimodal active reasoning.

---

### 99. Empowering Language Models with Active Inquiry for Deeper Understanding
- **Authors:** Ruiming Pang, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.03719](https://arxiv.org/abs/2402.03719)
- **PDF:** `literature/99_Active-Inquiry.pdf`
- **Summary:** Model asks **targeted questions to fill missing information** before answering.
- **Relevance:** Core active inquiry paradigm for AR-Bench.

---

### 100. Human and Model Uncertainty Guidance to Ask Clarifying Questions
- **Authors:** Alberto Testoni, et al.
- **Year/Venue:** 2024, EACL
- **Link:** [ACL Anthology](https://aclanthology.org/2024.eacl-short.8.pdf)
- **PDF:** `literature/100_Uncertainty-Guidance-Clarification.pdf`
- **Summary:** Using uncertainty signals to decide **when clarification is warranted** ("ask vs answer" boundary).
- **Relevance:** ⭐ **Core concept**. Uncertainty triggers question-asking—exactly what AR-Bench needs.

---

### 101. Probing the Multi-turn Planning Capabilities of LLMs via 20 Question Games
- **Authors:** Yizhe Zhang, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2310.01468](https://arxiv.org/abs/2310.01468)
- **PDF:** `literature/101_20-Questions-Multi-Turn.pdf`
- **Summary:** **20 Questions** evaluation for multi-turn state tracking and strategic question asking. Explores RL to improve.
- **Relevance:** ⭐ **Directly relevant**. 20Q is similar to AR-Bench Guessing Numbers.

---

### 102. CAMBIGNQ: Asking Clarification Questions to Handle Ambiguity in Open-Domain QA
- **Authors:** Dongryeol Lee, et al.
- **Year/Venue:** 2023, OpenReview
- **Link:** [OpenReview](https://openreview.net/pdf?id=HsvZUde6wT)
- **PDF:** `literature/102_CAMBIGNQ-Clarification-QA.pdf`
- **Summary:** Ambiguous questions dataset + 3-stage pipeline: **ambiguity detection → CQ generation → clarification-conditioned QA**.
- **Relevance:** Pipeline structure applicable to AR-Bench.

---

### 103. Prompting and Evaluating LLMs for Proactive Dialogues
- **Authors:** Yang Deng, et al.
- **Year/Venue:** 2023, Findings of EMNLP
- **Link:** [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.1012.pdf)
- **PDF:** `literature/103_Proactive-Dialogue-Eval.pdf`
- **Summary:** Evaluates LLM prompting for **proactive behaviors** including clarification. Surfaces systematic failure modes.
- **Relevance:** Understanding why LLMs fail at proactive behavior on AR-Bench.

---

### 104. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **Authors:** Shunyu Yao, et al.
- **Year/Venue:** 2023, NeurIPS
- **arXiv:** [2305.10601](https://arxiv.org/abs/2305.10601)
- **PDF:** `literature/104_Tree-of-Thoughts.pdf`
- **Summary:** ⭐ **Key method**. Explicit search over multiple thought trajectories with evaluation/backtracking. **Diversity-promoting exploration**.
- **Relevance:** ToT structure could help AR-Bench explore different question strategies.

---

### 105. Toolformer: Language Models Can Teach Themselves to Use Tools
- **Authors:** Timo Schick, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2302.04761](https://arxiv.org/abs/2302.04761)
- **PDF:** `literature/105_Toolformer.pdf`
- **Summary:** ⭐ **Foundational**. LMs learn **when to call external tools** (including search/QA systems). Information seeking as learned policy.
- **Relevance:** AR-Bench information gathering as tool use paradigm.

---

### 106. Self-Ask: Measuring and Narrowing the Compositionality Gap in Language Models
- **Authors:** Ofir Press, et al.
- **Year/Venue:** 2023, Findings of EMNLP
- **arXiv:** [2210.03350](https://arxiv.org/abs/2210.03350)
- **PDF:** `literature/106_Self-Ask-Compositionality.pdf`
- **Summary:** Model generates and answers **intermediate questions** before final answer. Operationalizes internal question generation.
- **Relevance:** Self-ask for AR-Bench—decompose problem into sub-questions.

---

### 107. ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors:** Shunyu Yao, et al.
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2210.03629](https://arxiv.org/abs/2210.03629)
- **PDF:** `literature/107_ReAct.pdf`
- **Summary:** ⭐ **Foundational**. Interleaves reasoning with actions (e.g., querying knowledge sources) to **actively gather missing information**.
- **Relevance:** Core agentic paradigm applicable to AR-Bench information gathering.

---

### 108. Diverse and Specific Clarification Question Generation with Keywords
- **Authors:** Zhiling Zhang, et al.
- **Year/Venue:** 2021, WWW
- **arXiv:** [2104.10317](https://arxiv.org/abs/2104.10317)
- **PDF:** `literature/108_Diverse-Specific-CQ-Keywords.pdf`
- **Summary:** CQ generation producing **multiple questions that are specific and diverse** via keyword conditioning.
- **Relevance:** Diverse question generation addresses Hivemind homogeneity problem.

---

### 109. Towards Facet-Driven Generation of Clarifying Questions for Conversational Search
- **Authors:** Ivan Sekulić, et al.
- **Year/Venue:** 2021, ICTIR
- **DOI:** [10.1145/3471158.3472257](https://doi.org/10.1145/3471158.3472257)
- **PDF:** `literature/109_Facet-Driven-CQ.pdf`
- **Summary:** Clarifying questions guided by explicit **query facets** (query+facet→question).
- **Relevance:** Facet-based question generation for structured AR-Bench exploration.

---

### 110. Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions
- **Authors:** Mohammad Aliannejadi, et al.
- **Year/Venue:** 2021, EMNLP
- **arXiv:** [2109.05794](https://arxiv.org/abs/2109.05794)
- **PDF:** `literature/110_Open-Domain-Dialogue-CQ.pdf`
- **Summary:** Dataset and evaluation for **when systems should ask clarification** and how to evaluate CQ quality.
- **Relevance:** Evaluation methodology for AR-Bench question quality.

---

### 111. Generating Clarifying Questions for Information Retrieval
- **Authors:** Hamed Zamani, et al.
- **Year/Venue:** 2020, ACM
- **DOI:** [10.1145/3340531.3412778](https://doi.org/10.1145/3340531.3412778)
- **PDF:** `literature/111_CQ-Information-Retrieval.pdf`
- **Summary:** **Foundational IR framing** of clarifying questions for resolving ambiguous queries.
- **Relevance:** IR perspective on active information gathering.

---

### 112. ClarQ: A Large-scale and Diverse Dataset for Clarification Question Generation
- **Authors:** Vaibhav Kumar, et al.
- **Year/Venue:** 2020, ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2020.acl-main.651/)
- **PDF:** `literature/112_ClarQ-Dataset.pdf`
- **Summary:** Large-scale CQ dataset from StackExchange (~2M examples) for training CQ generators.
- **Relevance:** Training data for AR-Bench question generation.

---

### 113. ConvAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ)
- **Authors:** Mohammad Aliannejadi, et al.
- **Year/Venue:** 2020, SCAI/EMNLP Workshop
- **arXiv:** [2009.11352](https://arxiv.org/abs/2009.11352)
- **PDF:** `literature/113_ConvAI3-ClariQ.pdf`
- **Summary:** Shared-task setup for mixed-initiative dialogue requiring **ambiguity detection and clarification**.
- **Relevance:** Evaluation framework for active dialogue systems.

---

### 114. Qulac: Asking Clarifying Questions in Open-Domain Information-Seeking Conversations
- **Authors:** Mohammad Aliannejadi, et al.
- **Year/Venue:** 2019, SIGIR
- **arXiv:** [1907.06554](https://arxiv.org/abs/1907.06554)
- **PDF:** `literature/114_Qulac-Clarification-Search.pdf`
- **Summary:** ⭐ **Foundational**. Evaluation setup for clarifying-question selection/utility in conversational search.
- **Relevance:** Foundation for AR-Bench clarification evaluation.

---

# 6. LLM Self-Knowledge & Metacognition

### 115. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from LLMs
- **Authors:** Katherine Tian, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2305.14975](https://arxiv.org/abs/2305.14975)
- **PDF:** `literature/115_Just-Ask-Calibration.pdf`
- **Summary:** Prompt-based elicitation of confidence from RLHF-tuned models. Simple prompting choices materially affect calibration.
- **Relevance:** Practical prompting for AR-Bench confidence elicitation.

---

### 116. Shifting Attention to Relevance: Towards Predictive UQ of Free-Form LLMs
- **Authors:** Jinhao Duan, et al.
- **Year/Venue:** 2024, ACL
- **arXiv:** [2307.01379](https://arxiv.org/abs/2307.01379)
- **PDF:** `literature/116_Shifting-Attention-Relevance.pdf`
- **Summary:** Improves UQ for long generations by **reweighting toward semantically relevant** tokens/sentences.
- **Relevance:** Focus uncertainty on relevant parts of AR-Bench reasoning.

---

### 117. Relying on the Unreliable: The Impact of LLMs' Reluctance to Express Uncertainty
- **Authors:** Kaitlyn Zhou, et al.
- **Year/Venue:** 2024, ACL
- **arXiv:** [2401.06730](https://arxiv.org/abs/2401.06730)
- **PDF:** `literature/117_Reluctance-Express-Uncertainty.pdf`
- **Summary:** Deployed LMs often **reluctant to express uncertainty**, can be overconfident. User studies on downstream reliance.
- **Relevance:** Understanding why AR-Bench models answer when they should ask.

---

### 118. Navigating the Grey Area: How Expressions of Uncertainty Affect Language Models
- **Authors:** Kaitlyn Zhou, et al.
- **Year/Venue:** 2023, EMNLP
- **arXiv:** [2302.13439](https://arxiv.org/abs/2302.13439)
- **PDF:** `literature/118_Grey-Area-Uncertainty-Expressions.pdf`
- **Summary:** Linguistic epistemic markers ("I think...") **systematically skew** model outputs and accuracy.
- **Relevance:** How uncertainty expressions affect AR-Bench behavior.

---

### 119. Prudent Silence or Foolish Babble? Examining LLMs' Responses to the Unknown
- **Authors:** Genglin Liu, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2311.09731](https://arxiv.org/abs/2311.09731)
- **PDF:** `literature/119_Prudent-Silence-Babble.pdf`
- **Summary:** LLM behavior under unknown queries: when models **abstain vs hallucinate**.
- **Relevance:** Understanding hallucination vs abstention in AR-Bench.

---

### 120. On Verbalized Confidence Scores for LLMs
- **Authors:** D. Yang, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2412.14737](https://arxiv.org/abs/2412.14737)
- **PDF:** `literature/120_Verbalized-Confidence-Analysis.pdf`
- **Summary:** When verbalized confidence is reliable; prompting choices strongly affect calibration.
- **Relevance:** Making verbalized confidence work for AR-Bench.

---

### 121. CalibVIP: Calibrating Verbalized Probabilities for LLMs
- **Authors:** Sinan Tan, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2410.06707](https://arxiv.org/abs/2410.06707)
- **PDF:** `literature/121_CalibVIP-Verbalized-Probabilities.pdf`
- **Summary:** Techniques to calibrate **verbalized probability statements** to match empirical correctness.
- **Relevance:** Making "I'm 70% sure" actually mean 70% for AR-Bench.

---

### 122. Llamas Know What GPTs Don't Show: Surrogate Models for Confidence Estimation
- **Authors:** Vaishnavi Shrivastava, et al.
- **Year/Venue:** 2023, arXiv
- **arXiv:** [2311.08877](https://arxiv.org/abs/2311.08877)
- **PDF:** `literature/122_Surrogate-Confidence-Estimation.pdf`
- **Summary:** Trains surrogate models to **estimate confidence for closed/API models** without logprobs.
- **Relevance:** Confidence estimation for black-box AR-Bench models.

---

### 123. INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection
- **Authors:** Chao Chen, et al.
- **Year/Venue:** 2024, ICLR
- **arXiv:** [2402.03744](https://arxiv.org/abs/2402.03744)
- **PDF:** `literature/123_INSIDE-Internal-States.pdf`
- **Summary:** ⭐ **Key finding**. Internal hidden-state geometry **detects hallucinations** beyond token-probability heuristics.
- **Relevance:** Internal signals could trigger AR-Bench question-asking.

---

### 124. The Internal State of an LLM Knows When It's Lying
- **Authors:** Amos Azaria, Tom Mitchell
- **Year/Venue:** 2023, Findings of EMNLP
- **arXiv:** [2304.13734](https://arxiv.org/abs/2304.13734)
- **PDF:** `literature/124_Internal-State-Lying.pdf`
- **Summary:** Classifier on hidden activations predicts truthfulness **better than likelihood heuristics**.
- **Relevance:** Internal truthfulness detection for AR-Bench reliability.

---

### 125. Cognitive Dissonance: Why Do LLM Outputs Disagree with Internal Representations?
- **Authors:** Kevin Liu, et al.
- **Year/Venue:** 2023, EMNLP
- **arXiv:** [2312.03729](https://arxiv.org/abs/2312.03729)
- **PDF:** `literature/125_Cognitive-Dissonance.pdf`
- **Summary:** When output truth judgments and **probe-based internal signals disagree** (confabulation/deception).
- **Relevance:** Understanding when model "knows" it's wrong internally but says otherwise.

---

### 126. Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation
- **Authors:** Hanxing Ding, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.10612](https://arxiv.org/abs/2402.10612)
- **PDF:** `literature/126_Adaptive-Retrieval-Hallucination.pdf`
- **Summary:** Triggers retrieval **selectively based on uncertainty/hallucination indicators**.
- **Relevance:** Adaptive information gathering for AR-Bench.

---

### 127. Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval
- **Authors:** Ingeol Baek, et al.
- **Year/Venue:** 2025, Findings of NAACL
- **arXiv:** [2410.13339](https://arxiv.org/abs/2410.13339)
- **PDF:** `literature/127_Probing-RAG.pdf`
- **Summary:** Uses hidden states + prober to decide **whether more retrieval is needed** (metacognitive control).
- **Relevance:** Internal signals guiding AR-Bench information gathering.

---

### 128. RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation
- **Authors:** Zihan Zhang, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2402.16457](https://arxiv.org/abs/2402.16457)
- **PDF:** `literature/128_RetrievalQA-Adaptive.pdf`
- **Summary:** Benchmark for **adaptive RAG** (retrieval only when needed). Failure modes of thresholded heuristics.
- **Relevance:** Understanding when retrieval/questioning helps vs hurts.

---

### 129. Teaching Language Models to Faithfully Express their Uncertainty
- **Authors:** Bryan Eikema, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2510.12587](https://arxiv.org/abs/2510.12587)
- **PDF:** `literature/129_Faithful-Uncertainty-Tuning.pdf`
- **Summary:** **Faithful Uncertainty Tuning (FUT)**: trains models to hedge in ways that match sample variability (faithfulness).
- **Relevance:** Training models for faithful AR-Bench uncertainty expression.

---

### 130. DINCO: Calibrating Verbalized Confidence with Self-Generated Distractors
- **Authors:** Victor Wang, Elias Stengel-Eskin
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2509.25532](https://arxiv.org/abs/2509.25532)
- **PDF:** `literature/130_DINCO-Self-Distractors.pdf`
- **Summary:** Normalizing confidence across self-generated alternatives reduces **suggestibility-driven overconfidence**.
- **Relevance:** Reducing AR-Bench overconfidence.

---

### 131. Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence
- **Authors:** Glenn Zhang, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2512.11998](https://arxiv.org/abs/2512.11998)
- **PDF:** `literature/131_Direct-Confidence-Alignment.pdf`
- **Summary:** Preference optimization aligning **verbalized confidence with internal token-probability confidence**.
- **Relevance:** Making expressed and actual AR-Bench confidence consistent.

---

### 132. Uncertainty in Natural Language Processing: Sources, Quantification, and Applications
- **Authors:** Mengting Hu, et al.
- **Year/Venue:** 2023, arXiv (Survey)
- **arXiv:** [2306.04459](https://arxiv.org/abs/2306.04459)
- **PDF:** `literature/132_Survey-Uncertainty-NLP.pdf`
- **Summary:** Survey of uncertainty sources and quantification methods in NLP.
- **Relevance:** **Reference survey** for NLP uncertainty background.

---

### 133. Conformal Prediction for Natural Language Processing
- **Authors:** Margarida Campos, et al.
- **Year/Venue:** 2024, TACL
- **DOI:** [10.1162/tacl_a_00715](https://doi.org/10.1162/tacl_a_00715)
- **PDF:** `literature/133_Survey-Conformal-NLP.pdf`
- **Summary:** Survey of conformal prediction techniques for NLP.
- **Relevance:** **Reference survey** for conformal methods in NLP.

---

# 7. Output Diversity & Mode Collapse

### 134. The Curious Case of Neural Text Degeneration
- **Authors:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
- **Year/Venue:** 2020, ICLR
- **arXiv:** [1904.09751](https://arxiv.org/abs/1904.09751)
- **PDF:** `literature/134_Neural-Text-Degeneration.pdf`
- **Summary:** ⭐ **Foundational**. Characterizes "degeneration" (blandness + repetition) in ML decoding. Introduces **nucleus (top-p) sampling**.
- **Relevance:** Understanding why AR-Bench models give repetitive answers.

---

### 135. Neural Text Generation With Unlikelihood Training
- **Authors:** Sean Welleck, et al.
- **Year/Venue:** 2020, ICLR
- **arXiv:** [1908.04319](https://arxiv.org/abs/1908.04319)
- **PDF:** `literature/135_Unlikelihood-Training.pdf`
- **Summary:** **Unlikelihood training** objectives that downweight repeated/undesirable continuations at model level.
- **Relevance:** Training to reduce AR-Bench repetition.

---

### 136. Closing the Curious Case of Neural Text Degeneration
- **Authors:** Matthew Finlayson, et al.
- **Year/Venue:** 2024, ICLR
- **Link:** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/file/34899013589ef41aea4d7b2f0ef310c1-Paper-Conference.pdf)
- **PDF:** `literature/136_Closing-Degeneration-Case.pdf`
- **Summary:** Mechanistic explanation of why truncation-based sampling mitigates degeneration.
- **Relevance:** Understanding degeneration mechanisms.

---

### 137. Contrastive Decoding: Open-ended Text Generation as Optimization
- **Authors:** Xiang Lisa Li, et al.
- **Year/Venue:** 2023, ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2023.acl-long.687/)
- **PDF:** `literature/137_Contrastive-Decoding.pdf`
- **Summary:** Expert LM vs amateur LM to **penalize generic continuations** while maintaining plausibility.
- **Relevance:** Decoding for more diverse AR-Bench responses.

---

### 138. SimCTG: A Contrastive Framework for Neural Text Generation
- **Authors:** Yixuan Su, et al.
- **Year/Venue:** 2022, NeurIPS
- **Link:** [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/871cae8f599cb8bbfcb0f58fe1af95ad-Abstract-Conference.html)
- **PDF:** `literature/138_SimCTG-Contrastive-Search.pdf`
- **Summary:** Contrastive training + **contrastive search** decoding for diverse-yet-coherent generations.
- **Relevance:** Diversity-promoting decoding for AR-Bench.

---

### 139. Locally Typical Sampling
- **Authors:** Clara Meister, et al.
- **Year/Venue:** 2022, arXiv
- **arXiv:** [2202.00666](https://arxiv.org/abs/2202.00666)
- **PDF:** `literature/139_Typical-Sampling.pdf`
- **Summary:** **Typical decoding**: selecting tokens with "typical" information content. Diversity-quality balance.
- **Relevance:** Alternative sampling for AR-Bench diversity.

---

### 140. Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity
- **Authors:** Sourya Basu, et al.
- **Year/Venue:** 2021, ICLR
- **arXiv:** [2007.14966](https://arxiv.org/abs/2007.14966)
- **PDF:** `literature/140_Mirostat.pdf`
- **Summary:** Feedback-based adaptive sampler targeting desired perplexity. Addresses **repetition traps**.
- **Relevance:** Adaptive sampling to avoid AR-Bench loops.

---

### 141. Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models
- **Authors:** Ashwin K. Vijayakumar, et al.
- **Year/Venue:** 2016, arXiv
- **arXiv:** [1610.02424](https://arxiv.org/abs/1610.02424)
- **PDF:** `literature/141_Diverse-Beam-Search.pdf`
- **Summary:** **Foundational**. Augments beam search with diversity terms to avoid near-duplicate beams.
- **Relevance:** Diverse candidate generation for AR-Bench.

---

### 142. Determinantal Beam Search
- **Authors:** Clara Meister, et al.
- **Year/Venue:** 2021, ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2021.acl-long.512/)
- **PDF:** `literature/142_Determinantal-Beam-Search.pdf`
- **Summary:** Beam search as **DPP-inspired subset selection**, explicitly discouraging overlap.
- **Relevance:** Principled diverse search for AR-Bench.

---

### 143. A Diversity-Promoting Objective Function for Neural Conversation Models
- **Authors:** Jiwei Li, et al.
- **Year/Venue:** 2016, NAACL
- **Link:** [ACL Anthology](https://aclanthology.org/N16-1014/)
- **PDF:** `literature/143_MMI-Diversity-Dialogue.pdf`
- **Summary:** ⭐ **Foundational**. **MMI-style objectives** to avoid generic responses. Popularizes distinct-n metrics.
- **Relevance:** Diversity objectives for AR-Bench dialogue.

---

### 144. HUSE: Unifying Human and Statistical Evaluation for NLG
- **Authors:** Tatsunori B. Hashimoto, et al.
- **Year/Venue:** 2019, NAACL
- **arXiv:** [1904.02792](https://arxiv.org/abs/1904.02792)
- **PDF:** `literature/144_HUSE-Quality-Diversity.pdf`
- **Summary:** Framework jointly capturing **quality and diversity** in evaluation.
- **Relevance:** Evaluation framework for AR-Bench output diversity.

---

### 145. Evaluating the Evaluation of Diversity in Natural Language Generation
- **Authors:** Guy Tevet, et al.
- **Year/Venue:** 2021, EACL
- **arXiv:** [2004.02990](https://arxiv.org/abs/2004.02990)
- **PDF:** `literature/145_Evaluating-Diversity-Evaluation.pdf`
- **Summary:** Many automatic diversity metrics respond to **surface-form rather than semantic diversity**.
- **Relevance:** Understanding diversity metric limitations for AR-Bench.

---

### 146. MAUVE: Measuring the Gap Between Neural Text and Human Text
- **Authors:** Krishna Pillutla, et al.
- **Year/Venue:** 2021, NeurIPS
- **arXiv:** [2102.01454](https://arxiv.org/abs/2102.01454)
- **PDF:** `literature/146_MAUVE.pdf`
- **Summary:** ⭐ **Key metric**. Distribution-level comparison revealing **quality-only and diversity-only failure modes**.
- **Relevance:** Detecting AR-Bench output collapse toward narrow regions.

---

### 147. Texygen: A Benchmarking Platform for Text Generation Models
- **Authors:** Yaoming Zhu, et al.
- **Year/Venue:** 2018, arXiv
- **arXiv:** [1802.01886](https://arxiv.org/abs/1802.01886)
- **PDF:** `literature/147_Texygen.pdf`
- **Summary:** Benchmarking suite including diversity-oriented metrics.
- **Relevance:** Diversity evaluation toolkit.

---

### 148. GDPP: Learning Diverse Generations Using Determinantal Point Process
- **Authors:** Mohamed Elfeki, et al.
- **Year/Venue:** 2019, ICML
- **arXiv:** [1812.00068](https://arxiv.org/abs/1812.00068)
- **PDF:** `literature/148_GDPP-DPP-Diversity.pdf`
- **Summary:** **DPP-based penalty** targeting mode collapse via training objective.
- **Relevance:** Training-time diversity for AR-Bench models.

---

### 149. Evaluating the Diversity and Quality of LLM Generated Content
- **Authors:** Andrew Shypula, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2504.12522](https://arxiv.org/abs/2504.12522)
- **PDF:** `literature/149_LLM-Diversity-Quality-Eval.pdf`
- **Summary:** Evaluates lexical vs embedding-based diversity metrics on LLM outputs.
- **Relevance:** Modern diversity metrics for AR-Bench evaluation.

---

### 150. Benchmarking Linguistic Diversity of Large Language Models
- **Authors:** Yanzhu Guo, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2412.10271](https://arxiv.org/abs/2412.10271)
- **PDF:** `literature/150_Benchmarking-Linguistic-Diversity.pdf`
- **Summary:** Corpus-level framework benchmarking LLM outputs across **lexical, syntactic, semantic diversity**.
- **Relevance:** Multi-level diversity analysis for AR-Bench.

---

### 151. LLM Output Homogenization is Task Dependent
- **Authors:** Shomik Jain, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2509.21267](https://arxiv.org/abs/2509.21267)
- **PDF:** `literature/151_Task-Dependent-Homogenization.pdf`
- **Summary:** Homogenization must be defined per task. Introduces **task-anchored functional diversity**.
- **Relevance:** Task-specific diversity matters for AR-Bench task types.

---

### 152. We're Different, We're the Same: Creative Homogeneity Across LLMs
- **Authors:** Emily Wenger, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2501.19361](https://arxiv.org/abs/2501.19361)
- **PDF:** `literature/152_Creative-Homogeneity-Across-LLMs.pdf`
- **Summary:** Population-level similarity across LLMs on divergent-thinking tasks **higher than humans**.
- **Relevance:** Inter-model homogeneity on creative AR-Bench tasks.

---

### 153. Generative AI Enhances Individual Creativity but Reduces Collective Diversity
- **Authors:** Anil R. Doshi, et al.
- **Year/Venue:** 2024, Science Advances
- **DOI:** [10.1126/sciadv.adn5290](https://doi.org/10.1126/sciadv.adn5290)
- **PDF:** `literature/153_AI-Creativity-Collective-Diversity.pdf`
- **Summary:** ⭐ **Key finding**. AI assistance raises individual creativity while **reducing group-level diversity**.
- **Relevance:** Systemic diversity reduction from LLM assistance.

---

### 154. Understanding the Repeat Curse in Large Language Models
- **Authors:** J. Yao, et al.
- **Year/Venue:** 2025, Findings of ACL
- **Link:** [ACL Anthology](https://aclanthology.org/2025.findings-acl.406.pdf)
- **PDF:** `literature/154_Repeat-Curse.pdf`
- **Summary:** Studies repetition across granularities (token-to-paragraph). Analyzes causes and mitigation.
- **Relevance:** Understanding AR-Bench repetition loops.

---

### 155. Improving Uncertainty Quantification in LLMs via Semantic Embeddings
- **Authors:** Y. S. Grewal, et al.
- **Year/Venue:** 2024, arXiv
- **arXiv:** [2410.22685](https://arxiv.org/abs/2410.22685)
- **PDF:** `literature/155_Semantic-Embeddings-UQ.pdf`
- **Summary:** Semantic-embedding structure makes **semantic uncertainty estimates smoother**.
- **Relevance:** Improved semantic uncertainty for AR-Bench.

---

### 156. Inv-Entropy: A Fully Probabilistic Framework for Uncertainty Quantification in Language Models
- **Authors:** Haoyi Song, et al.
- **Year/Venue:** 2025, arXiv
- **arXiv:** [2506.09684](https://arxiv.org/abs/2506.09684)
- **PDF:** `literature/156_Inv-Entropy-Probabilistic-UQ.pdf`
- **Summary:** Probabilistic inverse-model view defining uncertainty via **diversity under perturbations**.
- **Relevance:** Diversity-based uncertainty framework.

---

### 157. AI Models Collapse When Trained on Recursively Generated Data
- **Authors:** Ilia Shumailov, et al.
- **Year/Venue:** 2024, Nature
- **Link:** [Nature](https://www.nature.com/articles/s41586-024-07566-y)
- **PDF:** `literature/157_Model-Collapse-Recursive-Data.pdf`
- **Summary:** ⭐ **High-impact**. **Model collapse** under iterative training on synthetic data. Distributions drift and tails disappear.
- **Relevance:** Background on ecosystem-level homogenization affecting future AR-Bench models.

---

# Cross-Reference Index

## By Theme

| Theme | Papers |
|-------|--------|
| Semantic Entropy/Uncertainty | 00, 09, 10, 35, 36, 60 |
| Self-Evaluation/P(True) | 11, 13, 17, 18 |
| Verbalized Confidence | 12, 14, 20, 21, 115, 120, 121, 129, 130, 131 |
| Conformal Prediction | 05, 26-28, 75-90 |
| Bayesian LoRA/Adapters | 02, 03, 42-59 |
| Prompt Sensitivity | 04, 61-74 |
| Clarification Questions | 91-114 |
| Information Gain | 96, 97, 101 |
| Mode Collapse/Diversity | 08, 35, 36, 134-157 |
| Internal States | 123-125, 127 |

## By AR-Bench Task

| Task | Relevant Papers |
|------|-----------------|
| Detective Cases | 01 (ICE), 09 (Semantic), 26 (Conformal), 92 (ACT) |
| Situation Puzzles | 100 (Uncertainty-guided CQ), 102 (CAMBIGNQ), 103 (Proactive) |
| Guessing Numbers | 96 (UoT), 97 (EIG), 101 (20Q), 141 (Diverse Beam) |

---

*Last updated: 2025-01-09*  
*Papers: 157 total (9 seed + 148 discovered)*  
*PDFs: All papers downloaded to `literature/`*
