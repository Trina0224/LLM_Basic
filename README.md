# LLM Basic

This repository collects beginner-friendly materials for explaining how large language models work.

The first module uses one inspectable sentence from the notebook:

> At Oracle Park, Trina asked the old oracle whether Oracle Linux would boot before the database demo.

That sentence is used to walk through the core Transformer flow step by step: tokenization, embeddings, Q/K/V, attention, residual connections, layer normalization, the feed-forward block, logits, softmax, and next-token prediction.

## Contents

- `notebooks/transformer_one_sentence_en.ipynb` - Colab notebook for the English one-sentence Transformer walkthrough.
- `reports/transformer_one_sentence_en/` - Markdown report and supporting assets generated from the notebook.
- `results/transformer_one_sentence_en.zip` - Downloadable bundle for the English report.
- `results/transformer_lab_outputs.zip` - Output bundle from the Transformer lab.
- `results/equations.zip` - Supporting equation assets.

## Open in Colab

[Open the Transformer notebook in Colab](https://colab.research.google.com/github/trinashih/LLM_Basic/blob/main/notebooks/transformer_one_sentence_en.ipynb)

## How to Use

Open the notebook in Colab if you want to run the walkthrough interactively. Use the report folder or zip files when preparing a presentation or sharing the material with readers who do not need to run the notebook.

## Notes

The repository is intended for educational use and keeps the examples small on purpose. The goal is to make the Transformer pipeline readable before moving into heavier math or model-training details.
