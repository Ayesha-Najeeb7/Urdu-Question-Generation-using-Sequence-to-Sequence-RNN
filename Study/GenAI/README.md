# Urdu Question Generation  From-Scratch Sequence-to-Sequence Model

A from-scratch sequence-to-sequence model that reads an Urdu sentence with an answer marked inside it, and generates the question that answer responds to. Built entirely with PyTorch primitives — no pretrained weights, no Transformers, no off-the-shelf seq2seq models — as part of the Generative AI course (Fall 2026), Assignment 1.

## Team

- Nalain-e-Muhammad
- Ayesha Najeeb

## Problem

Given an Urdu sentence with the answer span wrapped in `<ans>...</ans>`, generate the Urdu question whose answer is exactly that span.

**Example (shown in English):** "The Indus is about `<ans>3,180 km</ans>` long" → "How long is the Indus?"

This is the same task family as translation and summarization: variable-length input mapped to variable-length output with no fixed alignment.

## Dataset

[UQA: Corpus for Urdu Question Answering](https://github.com/sameearif/UQA) (Arif, Farid, Athar & Raza, LREC-COLING 2024) — a translation of SQuAD 2.0 into Urdu that preserves answer-span character offsets (~142k context–question–answer rows). Following Du et al. (2017), only the sentence containing the answer is used as the source (not the full paragraph), which makes the task learnable for a from-scratch model on this data scale.

Out-of-domain generalization is tested on [Wiki-UQA](https://huggingface.co/datasets/uqa/Wiki-UQA), a separate 210-row Urdu Wikipedia-based QA set using the same schema.

## Model Architecture

| Component | Detail |
|---|---|
| Encoder | 2-layer bidirectional LSTM |
| Decoder | 2-layer LSTM |
| Attention | Bahdanau (additive) attention |
| Embedding size | 256 |
| Hidden size | 512 |
| Dropout | 0.3 |
| Tokenizer | SentencePiece (unigram), vocab size 8,000, trained from scratch on train sources/targets |
| Decoding | Greedy and beam search (beam width 3) |

No pretrained embeddings or checkpoints are used anywhere in the pipeline.

## Repository Structure

```
.
├── train.tsv, valid.tsv          # sentence-level (source, target) pairs
├── wiki_uqa.tsv                   # out-of-domain test pairs
├── ur_sp.model, ur_sp.vocab       # SentencePiece tokenizer
├── best_model.pt                  # best checkpoint (by validation loss)
├── notebook.ipynb                 # full training + evaluation pipeline
├── results/
│   ├── samples_uqa_valid.tsv
│   ├── samples_wiki_uqa.tsv
│   ├── table4_human_eval.csv
│   └── attention_heatmap.png
├── frontend/                      # minimal web UI (Streamlit/Gradio/Flask)
└── README.md
```

## Setup

```bash
pip install datasets sentencepiece sacrebleu rouge-score torch
```

Use Colab or Kaggle for GPU access — training takes 1–2 hours on a free GPU (batch size 64, 10–15 epochs).

## Running It

1. **Data prep + tokenizer training** — run the notebook cells under "Task 1" and "Task 2" to build `train.tsv` / `valid.tsv` and train the SentencePiece model.
2. **Train the model** — run the training loop cell (teacher forcing, Adam, lr = 0.001, batch size 64). Best checkpoint is saved automatically by validation loss.
3. **Evaluate** — run the Task 4 cells to compute perplexity, BLEU-4, ROUGE-L, and generate qualitative samples for both UQA validation and Wiki-UQA.
4. **Front end** — launch the web UI, paste an Urdu sentence, mark the answer span, and view greedy/beam outputs live.

## Results

### Table 1 — Dataset Statistics

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | N/A |
| Pairs after length filter | 75,067 | 10,018 | 177 |
| Mean source length (tokens) | 32.5 | N/A | N/A |
| Mean target length (tokens) | 12.0 | N/A | N/A |

*Validation and Wiki-UQA mean lengths are not yet computed — see "Known Gaps" below.*

### Table 2 — Model Configuration

| | |
|---|---|
| Encoder / decoder type | Bidirectional LSTM / LSTM + Bahdanau attention |
| Layers / embedding / hidden size | 2 / 256 / 512 |
| Vocabulary size | 8,000 |
| Trainable parameters | 24,742,209 |
| Optimiser, learning rate | Adam, lr = 0.001 |
| Batch size, epochs | 64; best epoch 8 (of 11 run) |

### Table 3 — Automatic Metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | 0.51 | 0.0000 | 25.35 | 0.00 |
| UQA valid | beam (k=3) | 0.70 | 0.0001 | 25.35 | 0.00 |
| Wiki-UQA | greedy | 0.29 | 0.0000 | 50.08 | 0.00 |
| Wiki-UQA | beam (k=3) | 0.36 | 0.0000 | 50.08 | 0.00 |

**Note:** BLEU-4/ROUGE-L are well below the expected range for this task (Du et al. 2017 report ~12 BLEU-4 on English SQuAD; 6–13 was expected here). Per the assignment's own guidance, a score this low more likely signals a pipeline issue than a fundamentally unlearnable task — see Discussion below.

Ratings not yet collected — see `results/table4_human_eval.csv`. Both members must independently rate the same fixed-seed 50 validation samples before this table is complete.

### Figures

1. Training and validation loss per epoch — *(insert loss_curve.png)*
2. Source/target length histograms — *(insert Figure_2_source.png / Figure_2_Train.png)*
3. Attention heat-map — *(insert results/attention_heatmap.png)*
4. Front-end screenshot — *(insert screenshot)*

## Discussion

- **Overfitting:** Training loss falls monotonically to 2.25 by epoch 11; validation loss bottoms out at epoch 8 (3.23) and creeps back up — the checkpoint used for evaluation is from epoch 8, not the final epoch.
- **Domain gap:** Perplexity roughly doubles from UQA validation (25.35) to Wiki-UQA (50.08), indicating the model doesn't transfer well from SQuAD-style translated text to naturally-written Wikipedia-derived Urdu.
- **Low BLEU-4/ROUGE-L:** Generated questions tend to collapse toward a narrow set of generic templates rather than varying with the marked answer span — a common failure mode for small-capacity, from-scratch seq2seq models trained for relatively few epochs on a morphologically rich language. This is flagged as a likely pipeline/training issue rather than an expected outcome, per the assignment's own sanity-check guidance.

## Known Gaps (to fill before final submission)

- [ ] Validation and Wiki-UQA mean source/target lengths (Table 1)
- [ ] Wiki-UQA answerable row count (Table 1)
- [ ] Human evaluation ratings from both members (Table 4)
- [ ] Front-end screenshot (Figure 4)
- [ ] Medium blog link
- [ ] LinkedIn post link
- [ ] Wall-clock training time and exact GPU model (Table 2)

## Citation

```
@inproceedings{arif-etal-2024-uqa,
  title     = "{UQA}: Corpus for {U}rdu Question Answering",
  author    = "Arif, Samee and Farid, Sualeha and Athar, Awais and Raza, Agha Ali",
  booktitle = "Proceedings of LREC-COLING 2024",
  year      = "2024",
  address   = "Torino, Italia",
  publisher = "ELRA and ICCL",
  url       = "https://aclanthology.org/2024.lrec-main.1497"
}
```
