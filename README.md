# Prompt-CCZSL: Prompt-Based Continual Compositional Zero-Shot Learning

* **Title**: *Prompt-Based Continual Compositional Zero-Shot Learning*
* **Paper**: Under review
* **Code**: This repository

---

## Overview

Continual Compositional Zero-Shot Learning (CCZSL) aims to recognize novel attribute–object
compositions (e.g., *old car*, *broken chair*) across incremental learning sessions, while
preserving previously learned compositional knowledge.

In this work, we propose a **prompt-based CCZSL framework** built upon vision–language models.
Our approach introduces learnable compositional prompts that are incrementally updated across
sessions, enabling the model to:
- adapt to new attributes, objects, and compositions,
- generalize to unseen compositions,
- mitigate catastrophic forgetting without replay.

The framework integrates **Session-aware Fusion Module** and **Multi Teacher Knowledge Distillation**, allowing stable prompt adaptation under continual learning constraints.

---

## Method Highlights

- Prompt-based formulation for continual compositional learning  
- Session-aware mechanisms for updating compositional representations
- Knowledge distillation across learning sessions
- Cosine anchor alignment for semantic consistency  
- Compatible with existing CZSL and VLM-based frameworks  

---

## Results

Quantitative and qualitative results will be released upon paper acceptance.

---

## Code Structure

