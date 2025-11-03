---

marp: true
theme: default
paginate: true
--------------

# Instantly Learning Preference Alignment via In-Context DPO

**Song et al., 2025**

**Review by Jason Gillette**

![bg right:40%](./assets/alignment_imagery.png)

---

## Introduction

**Paper summary**

* Proposes *In-Context Direct Preference Optimization (ICDPO)* — a *tuning-free* alignment method.
* Integrates **in-context learning (ICL)** with **Direct Preference Optimization (DPO)**.
* Alignment *without fine-tuning* by conditioning LLMs on retrieved preferences.
* Matches DPO accuracy while cutting compute and storage costs.

**Key points**

* No parameter updates — alignment entirely from prompt context.
* Uses *expert–amateur collaboration* to simulate fine-tuning.
* Employs a two-stage retriever (BM25 → SBERT).
* Outperforms supervised fine-tuning and rivals DPO.

---

## Background

### Human Preference Alignment

* Ensures model outputs are *helpful, harmless, honest*.
* Prevents harmful or unsafe completions.
* Traditionally achieved via **RLHF**, which uses:

  1. Base model (π₀)
  2. Reward model (r)
  3. Policy model (π) optimized with PPO.

---

### In-Context Learning (ICL)

* Model “learns” behaviors from examples *within the prompt* — no weight updates.
* Behaves as if fine-tuned temporarily:

  > *Condition → predict → adapt instantly.*
* Enables alignment through *prompt design* instead of retraining.

---

### Direct Preference Optimization (DPO)

* Simplifies RLHF by removing the reward model.
* Optimizes policy directly from human preference pairs (y⁺, y⁻).
* Grounds responses the base model would also deem probable (KL-regularized).
* Loss encourages higher likelihood for preferred over dis-preferred responses.
* Method still depends on additional training.

---

## Contributions

* **ICDPO Framework** — merges DPO’s preference logic with ICL.
* **Tuning-free Alignment** — performs DPO entirely in-context.
* **Expert–Amateur Collaboration** — contrastive scoring between conditioned and unconditioned model states.
* **Two-Stage Retrieval System** — BM25 + SBERT for efficient, relevant demonstration selection.
* **Comparable Accuracy, Fractional Cost** — achieves DPO-level results using only inference.

---

## Proposed Architecture

* Retriever R:

  * **BM25** → fast lexical candidate selection.
  * **SBERT** → semantic reranking for relevance.
* Prompt Builder: inject top-k demonstrations (prompt, preferred, dispreferred).
* LLM Scoring:

  * **Expert Score S(x,y)** — log-probability with demonstrations.
  * **Amateur Score Ŝ(x,y)** — log-probability without demonstrations.
  * **Δ(x,y)=S−Ŝ** → improvement from context.
  * **D(x)=Δ(y⁺)−Δ(y⁻)** → preference signal.
* Decision: choose or rerank by D(x).

*(See diagram of ICDPO architecture)*

---

## Methods / Experiment Set-up

### Datasets

* **HH-RLHF (Anthropic Helpful–Harmless)**
  Human-annotated safe/helpful response pairs.
* **OASST (OpenAssistant)**
  Crowdsourced instruction-response dataset with human preference labels.

### Models Tested

* LLaMA-2
* GPT-J

### Baselines

* Supervised Fine-Tuning (SFT)
* Direct Preference Optimization (DPO)
* In-Context DPO (ICDPO)

---

### Evaluation Metrics

* **Pairwise Preference Accuracy:**
  % of times the method ranks the human-preferred response higher.
* **Win Rate / KL-Regularization Analogs:**
  Used to assess stability and alignment strength.

*(Insert table PNG of results here.)*

---

## Results

| Dataset | DPO       | ICDPO     | Observation             |
| ------- | --------- | --------- | ----------------------- |
| HH-RLHF | ≈ 67 %    | ≈ 65–66 % | Comparable              |
| OASST   | ≈ 70 %    | ≈ 69–71 % | Equal / slightly better |
| SFT     | ≈ 58–60 % | —         | Much lower performance  |

* ICDPO ≈ DPO accuracy, vastly lower cost.
* Single forward pass vs training pipeline.

---

## Ablation Study

* Removing **SBERT reranker** → large drop in accuracy.
* Using **random demos** → performance near SFT levels.
* **> 5 demos** → diminishing returns; optimal few-shot window ≈ 3–5.
* Confirms retriever quality is key to ICDPO success.

---

## Potential Weaknesses

* Reliance on retriever quality — poor matches harm alignment.
* Unclear robustness to domain shift or safety edge cases.
* No true weight updates → benefits are temporary.
* Still requires annotated human preference data.
* Limited evaluation scope (only two datasets and two models).

---

## Questions for Further Study

1. How does ICDPO perform on non-instruction or multi-turn dialog tasks?
2. Can retriever bias lead to alignment failure?
3. Would using synthetic or active-learning preferences improve results?
4. How can contextual window limitations be mitigated?
5. Could a hybrid ICDPO + LoRA tuning yield persistent alignment?

---

## References

* [Instantly Learning Preference Alignment via In-context DPO](https://aclanthology.org/2025.naacl-long.8/) (Song et al., NAACL 2025)
