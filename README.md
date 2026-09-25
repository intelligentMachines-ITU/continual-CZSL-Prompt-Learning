# Prompt-Based Continual Compositional Zero-Shot Learning

[[PDF]](https://arxiv.org/pdf/2512.09172)**

This project hosts the benchmark datasets and code for the implementation of **[PROMPT-BASED CONTINUAL COMPOSITIONAL ZERO-SHOT
LEARNING](https://arxiv.org/pdf/2512.09172)** (arXiv).

## Overview
PromptCCZSL is a vision–language framework for Continual Compositional Zero-Shot Learning (CCZSL), where models incrementally learn evolving attribute–object compositions while retaining previously acquired knowledge.
Unlike conventional continual learning, CCZSL allows attributes and objects to recur across sessions while their compositions change, creating challenges such as semantic interference, compositional drift, and catastrophic forgetting.
The central goal of PromptCCZSL is to achieve a better balance between stability (retaining previously learned primitive and compositional knowledge) and plasticity (adapting to newly introduced compositions). The framework is evaluated on multiple CCZSL benchmarks and compared with existing VLM and non-VLM baselines.

![](cczsl-problem.png)

## Dataset
We evaluate our method on three compositional zero-shot learning benchmarks: MIT-States, UT-Zappos, and C-GQA.

## Acknowledgement
Our code references the following projects:
[[Troika]](https://github.com/bighuang624/troika)**
[[CSP]](https://github.com/BatsResearch/csp/tree/main)**

## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@misc{maryam2025promptbasedcontinualcompositionalzeroshot,
      title={Prompt-Based Continual Compositional Zero-Shot Learning}, 
      author={Sauda Maryam and Sara Nadeem and Faisal Qureshi and Mohsen Ali},
      year={2025},
      eprint={2512.09172},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.09172}, 
}
```
