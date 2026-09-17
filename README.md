# Sanskrit–English Machine Translation & Evaluation using NLLB

This repository demonstrates the use of Meta's **NLLB (No Language Left Behind)** model for **Sanskrit → English** machine translation, and evaluates translation quality using **BLEU**, **CHRF**, and **BERTScore**. The project is organized into three progressive tasks — from a basic translation demo to a full evaluation pipeline on the **Ramayana Anvaya** dataset.

---

## Table of Contents

- [Overview](#overview)
- [Model: NLLB](#model-nllb)
- [Environment & Setup](#environment--setup)
- [Repository Structure](#repository-structure)
- [Task 1 — NLLB Demo + Custom Test Set Evaluation](#task-1--nllb-demo--custom-test-set-evaluation)
- [Task 2 — Ramayana Anvaya Test Set Translation & Comparison](#task-2--ramayana-anvaya-test-set-translation--comparison)
- [Task 3 — Full Train Set Evaluation with BERTScore](#task-3--full-train-set-evaluation-with-bertscore)
- [Evaluation Metrics — Theory](#evaluation-metrics--theory)
  - [BLEU](#bleu)
  - [CHRF](#chrf)
  - [BERTScore](#bertscore)
  - [Choosing Between Them](#choosing-between-them)
- [Output File Schemas](#output-file-schemas)
- [How to Run](#how-to-run)
- [Notes, Caveats & Tips](#notes-caveats--tips)
- [References](#references)

---

## Overview

| Task | Goal | Dataset | Metrics |
|------|------|---------|---------|
| **Task 1** | Demo NLLB translation (Sanskrit→English) + build & evaluate a small custom test set | 3 demo sentences + 5 custom sentences (manually/ChatGPT-generated) | BLEU, CHRF |
| **Task 2** | Translate *sloka* and *prose* (anvaya) independently, compare translations | [`sanganaka/ramayana-anvaya`](https://huggingface.co/datasets/sanganaka/ramayana-anvaya) — **test split (1.83k rows)** | BLEU |
| **Task 3** | Repeat on the **train split**, extend the metric suite | Same dataset — **train split** | BLEU, CHRF, BERTScore |

All translation is performed using **`facebook/nllb-200-distilled-600M`** (or a larger NLLB checkpoint), with Sanskrit as the source language (`san_Deva`) and English as the target (`eng_Latn`).

---

## Model: NLLB

**NLLB-200** (No Language Left Behind) is a multilingual seq2seq translation model from Meta AI supporting 200+ languages, including low-resource languages such as Sanskrit.

- HF documentation: https://huggingface.co/docs/transformers/en/model_doc/nllb
- Model checkpoints used: `facebook/nllb-200-distilled-600M` (fast, Colab-friendly) — swap for `facebook/nllb-200-1.3B` or `facebook/nllb-200-3.3B` for higher quality if GPU memory allows.
- Sanskrit uses the **FLORES-200 language code**: `san_Deva` (Devanagari script). English is `eng_Latn`.
- Translation direction is controlled via the tokenizer's `src_lang` and the `forced_bos_token_id` set to the target language's token ID.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="san_Deva")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_sa_to_en(text, max_length=200):
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=max_length,
    )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
```

---

## Environment & Setup

Designed to run on **Google Colab** (T4 GPU recommended for batch inference over 1.83k+ rows).

```bash
pip install -q transformers sentencepiece datasets sacrebleu bert-score pandas tqdm accelerate
```

| Package | Purpose |
|---|---|
| `transformers` | Load & run NLLB model/tokenizer |
| `sentencepiece` | Required tokenizer backend for NLLB |
| `datasets` | Load `sanganaka/ramayana-anvaya` from Hugging Face Hub |
| `sacrebleu` | BLEU and CHRF scoring |
| `bert-score` | BERTScore (Task 3) |
| `pandas` | CSV read/write, dataframe manipulation |
| `tqdm` | Progress bars during batch inference |
| `accelerate` | Efficient model loading/inference on GPU |

**Recommended runtime:** Colab GPU (Runtime → Change runtime type → GPU). CPU works but is significantly slower for 1.83k+ samples.

---

## Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── task1_nllb_demo_and_custom_testset.ipynb
│   ├── task2_ramayana_test_translation.ipynb
│   └── task3_ramayana_train_translation_bertscore.ipynb
├── data/
│   ├── custom_test_sentences.csv          # Task 1 input (Sanskrit + Ground Truth English)
│   ├── task1_results.csv                  # Task 1 output (+ Model Output, BLEU, CHRF)
│   ├── task2_ramayana_test_results.csv     # Task 2 output
│   └── task3_ramayana_train_results.csv    # Task 3 output
└── requirements.txt
```

---

## Task 1 — NLLB Demo + Custom Test Set Evaluation

### Step 1: Demo translation (3 Sanskrit sentences)

Any 3 Sanskrit sentences (can be generated via ChatGPT) are translated using NLLB in a single Colab cell, purely to demonstrate the model's translation capability. Example sentences used:

1. `विद्या ददाति विनयं।` — *"Knowledge gives humility."*
2. `सत्यमेव जयते।` — *"Truth alone triumphs."*
3. `अहिंसा परमो धर्मः।` — *"Non-violence is the highest duty."*

```python
demo_sentences = [
    "विद्या ददाति विनयं।",
    "सत्यमेव जयते।",
    "अहिंसा परमो धर्मः।",
]

for s in demo_sentences:
    print(f"SA: {s}\nEN: {translate_sa_to_en(s)}\n")
```

### Step 2: Build a custom test dataset (`custom_test_sentences.csv`)

A hand-curated CSV of **5 Sanskrit sentences** with human/ChatGPT-verified English ground truth:

| Sanskrit | English (Ground Truth) |
|---|---|
| ... | ... |

Saved as `custom_test_sentences.csv` with two columns: `sanskrit`, `english_ground_truth`.

### Step 3: Run inference + compute metrics

The NLLB model translates each Sanskrit sentence in the CSV; sacrebleu computes **sentence-level BLEU** and **CHRF** between the ground truth and the model output. Results are appended as new columns and written back out.

➡ See [Output File Schemas](#output-file-schemas) for the exact column layout of `task1_results.csv`.

---

## Task 2 — Ramayana Anvaya Test Set Translation & Comparison

**Dataset:** [`sanganaka/ramayana-anvaya`](https://huggingface.co/datasets/sanganaka/ramayana-anvaya) on Hugging Face — **test split only (1,830 samples)**.

Each row in this dataset contains a Sanskrit **sloka** (verse, poetic word order) and its corresponding **anvaya / prose** (the same content in plain prose word order, syntactically simpler for parsing/translation). Both are semantically equivalent but structurally different.

### Pipeline

1. Load the dataset's `test` split via `datasets.load_dataset("sanganaka/ramayana-anvaya", split="test")`.
2. For each sample:
   - Translate `sloka` → English using NLLB → **E1**
   - Translate `prose` (anvaya) → English using NLLB → **E2**
3. Compute **BLEU(E1, E2)** — treating one as hypothesis and the other as reference — to measure how consistently the model translates the same underlying meaning when expressed in verse form vs. prose form.
4. Write everything to `task2_ramayana_test_results.csv`.

This is a useful diagnostic: since sloka (verse) word order is harder for models trained mostly on prose-like data, a **low BLEU between E1 and E2** signals that the model struggles more with poetic/inverted Sanskrit syntax than with plain prose — even though both describe the same event.

---

## Task 3 — Full Train Set Evaluation with BERTScore

Repeats the Task 2 pipeline over the **train split** of the same dataset, and extends the evaluation with additional CHRF comparisons and BERTScore.

### Pipeline

1. Load `datasets.load_dataset("sanganaka/ramayana-anvaya", split="train")`.
2. Translate `sloka` → **E1** and `prose` → **E2** using NLLB (same as Task 2).
3. Compute **CHRF** for three different pairs:
   - **CHRF(sloka, E1)** — how much character-level overlap exists between the original Sanskrit sloka and its own English translation (a sanity/self-consistency style check across languages — interpret with care, see note below).
   - **CHRF(prose, E2)** — same idea, for the prose/anvaya side.
   - **CHRF(E1, E2)** — character-level similarity between the two English translations, complementing the BLEU(E1, E2) score from Task 2.
4. Compute **BERTScore(E1, E2)** — a semantic, embedding-based similarity score between the two English translations, more robust to paraphrasing than BLEU/CHRF.
5. Write the combined results to `task3_ramayana_train_results.csv`.

> **Note on cross-lingual CHRF (sloka↔E1, prose↔E2):** CHRF was designed for same-language (or same-script) comparison. Comparing Devanagari Sanskrit text directly against English character n-grams will produce very low, largely uninformative scores — it's included here as requested, but should be interpreted as a rough/diagnostic number rather than a meaningful translation-quality metric. The two *meaningful* comparisons are CHRF/BLEU/BERTScore computed **between E1 and E2** (English vs. English).

---

## Evaluation Metrics — Theory

### BLEU

**BLEU (Bilingual Evaluation Understudy)** — Papineni et al., 2002 — measures **n-gram precision overlap** between a candidate translation and one or more reference translations, with a **brevity penalty** to discourage overly short outputs.

- Computes precision of matching 1-gram, 2-gram, 3-gram, and 4-gram sequences between candidate and reference.
- Combines these via geometric mean, then multiplies by a brevity penalty if the candidate is shorter than the reference.
- Formula (simplified):

  ```
  BLEU = BP × exp( Σ (wₙ × log pₙ) )     for n = 1..4
  ```
  where `pₙ` is the modified n-gram precision and `BP` is the brevity penalty.

- **Range:** 0–100 (sacrebleu reports as a percentage-style score).
- **Strengths:** Fast, standard, widely comparable across papers.
- **Weaknesses:** Surface-level (exact word match), penalizes valid paraphrases/synonyms, sensitive to tokenization, weak at sentence level (better suited to corpus-level aggregation), no notion of semantic similarity.
- In this project, `sacrebleu`'s `sentence_bleu` / `corpus_bleu` is used for standardized, reproducible scoring (avoids tokenization inconsistencies of the original NLTK/Perl BLEU scripts).

### CHRF

**CHRF (Character n-gram F-score)** — Popović, 2015 — measures **character-level n-gram F-score** (typically 6-gram) between candidate and reference, rather than word-level.

- Because it operates on characters rather than whitespace-tokenized words, it is **more robust to morphological variation** (word inflections, compounding, spelling variants) — a significant advantage for morphologically rich languages, and useful here even though our comparisons are in English.
- Formula (simplified): the F-score (with configurable β, default β=2 to favor recall) computed over character n-gram precision and recall:

  ```
  CHRF = (1 + β²) × (chrP × chrR) / (β² × chrP + chrR)
  ```

- **Range:** 0–100.
- **Strengths:** No dependency on word tokenization, works well cross-lingually/morphologically, correlates better with human judgment than BLEU for many language pairs.
- **Weaknesses:** Still a surface-form metric (no real semantic understanding); very short strings can produce noisy scores; not designed for cross-script comparison (see the caveat in Task 3).
- Computed here via `sacrebleu.CHRF()`.

### BERTScore

**BERTScore** — Zhang et al., 2020 — measures **semantic similarity** using contextual embeddings from a pretrained transformer (e.g., BERT/RoBERTa), rather than surface n-gram overlap.

- Each token in the candidate and reference sentences is embedded using a pretrained language model.
- For every candidate token, the **maximum cosine similarity** to any reference token is found (and vice versa), producing greedy-matched **precision** and **recall**.
- These are combined into an F1 score:

  ```
  BERTScore_F1 = 2 × (P × R) / (P + R)
  ```
  where `P` = average max-similarity of candidate tokens matched to reference, `R` = average max-similarity of reference tokens matched to candidate.

- **Range:** typically 0–1 (often rescaled/baseline-adjusted for interpretability).
- **Strengths:** Captures **semantic equivalence** — correctly rewards valid paraphrases and synonyms that BLEU/CHRF would penalize; correlates more strongly with human judgment on many tasks.
- **Weaknesses:** Computationally heavier (needs a transformer forward pass); quality depends on the underlying embedding model's domain/language coverage; less interpretable than a simple overlap percentage; can be less reliable for very short sentences.
- Computed here via the `bert-score` Python package (`bert_score.score(...)`), typically using an English-focused model (e.g., `roberta-large` or `microsoft/deberta-xlarge-mnli`) since both E1 and E2 are in English.

### Choosing Between Them

| Metric | Level | Captures Semantics? | Robust to Paraphrase? | Speed | Best Used For |
|---|---|---|---|---|---|
| **BLEU** | Word n-gram | No | No | Fast | Standard MT benchmark comparability |
| **CHRF** | Character n-gram | No | Partially (morphology) | Fast | Morphologically rich languages, sub-word robustness |
| **BERTScore** | Embedding/semantic | Yes | Yes | Slower (needs GPU/model) | Judging meaning-preserving quality, paraphrase tolerance |

Using all three together (as done in Task 3) gives a fuller picture: BLEU/CHRF catch surface-level fidelity, while BERTScore checks whether the *meaning* was preserved even when the wording differs — which matters a great deal when comparing a sloka's translation against its prose counterpart, since the two source texts are worded very differently despite meaning the same thing.

---

## Output File Schemas

**`task1_results.csv`**

| Column | Description |
|---|---|
| `sanskrit` | Original Sanskrit sentence |
| `english_ground_truth` | Human/ChatGPT reference English translation |
| `model_output_english` | NLLB-generated English translation |
| `bleu` | sacrebleu BLEU score (ground truth vs. model output) |
| `chrf` | sacrebleu CHRF score (ground truth vs. model output) |

**`task2_ramayana_test_results.csv`**

| Column | Description |
|---|---|
| `sloka` | Original Sanskrit sloka (verse) |
| `prose` | Original Sanskrit anvaya/prose |
| `sloka_translation_E1` | NLLB English translation of the sloka |
| `prose_translation_E2` | NLLB English translation of the prose |
| `bleu_E1_E2` | BLEU score between E1 and E2 |

**`task3_ramayana_train_results.csv`**

| Column | Description |
|---|---|
| `sloka` | Original Sanskrit sloka (verse) |
| `prose` | Original Sanskrit anvaya/prose |
| `sloka_translation_E1` | NLLB English translation of the sloka |
| `prose_translation_E2` | NLLB English translation of the prose |
| `bleu_E1_E2` | BLEU score between E1 and E2 |
| `chrf_sloka_E1` | CHRF between original sloka and E1 (diagnostic, cross-lingual) |
| `chrf_prose_E2` | CHRF between original prose and E2 (diagnostic, cross-lingual) |
| `chrf_E1_E2` | CHRF between E1 and E2 |
| `bertscore_E1_E2` | BERTScore F1 between E1 and E2 |

---

## How to Run

1. Open the relevant notebook in Google Colab (`Runtime → Change runtime type → GPU`).
2. Run the setup/install cell.
3. Run the model-loading cell (downloads `facebook/nllb-200-distilled-600M` from the Hub — a few hundred MB, cached after first run).
4. Run each task's cells top-to-bottom. Task 2/3 use `datasets.load_dataset` to stream the Ramayana dataset directly from the Hub — no manual download needed.
5. Resulting CSVs are written to `/content/` (Colab) — download via the Files panel or mount Google Drive to persist them.

**Batching tip:** For 1.83k+ rows (Tasks 2 & 3), translate in batches (e.g., batch size 16–32) rather than one sentence at a time, to substantially speed up inference on GPU.

---

## Notes, Caveats & Tips

- **Sanskrit is low-resource** for NLLB relative to major languages — expect noisier translations than, say, Hindi or French. Consider using a larger NLLB checkpoint (`1.3B` or `3.3B`) if quality is unsatisfactory and compute allows.
- **Sentence-level BLEU is noisy.** BLEU was designed for corpus-level aggregation; treat individual per-row BLEU scores in the CSVs as indicative rather than definitive, and also look at corpus-level aggregates (mean/median across all rows) when drawing conclusions.
- **Cross-lingual CHRF (Sanskrit vs. English)** in Task 3 is included per the assignment's requirements but is not a standard or reliable use of the metric — it's a diagnostic curiosity, not a translation-quality judgment.
- **BERTScore model choice matters.** Since E1/E2 are both English, an English or multilingual embedding model works; results may shift somewhat depending on which backbone is selected (default vs. `roberta-large` vs. `deberta-xlarge-mnli`).
- **Reproducibility:** Fix random seeds where sampling is involved (e.g., if only a subset of train is processed for a quick pass) and log the exact model checkpoint/version used, since NLLB checkpoint updates can shift results.

---

## References

- NLLB Model Documentation (Hugging Face): https://huggingface.co/docs/transformers/en/model_doc/nllb
- NLLB Team et al., *"No Language Left Behind: Scaling Human-Centered Machine Translation,"* 2022.
- Papineni et al., *"BLEU: a Method for Automatic Evaluation of Machine Translation,"* ACL 2002.
- Popović, M., *"chrF: character n-gram F-score for automatic MT evaluation,"* WMT 2015.
- Zhang et al., *"BERTScore: Evaluating Text Generation with BERT,"* ICLR 2020.
- `sacrebleu` library: https://github.com/mjpost/sacrebleu
- `bert-score` library: https://github.com/Tiiiger/bert_score
- Dataset: [`sanganaka/ramayana-anvaya`](https://huggingface.co/datasets/sanganaka/ramayana-anvaya) on Hugging Face
