# LLM Systems Learning Curriculum

Languages: [中文](../README.md) | English | [日本語](../ja/README.md)

This is not an API tutorial, and it is not a high-level explainer that only covers concepts.

This project is a Chinese-first LLM systems curriculum that is meant to be runnable, testable, modifiable, and reproducible. It starts from the smallest possible training loop and gradually builds toward language models, tokenizers, attention, Transformers, fine-tuning, RAG, distillation, evaluation, and domain-specific small-model engineering.

The curriculum assumes you already know the basics of Python. It does not assume prior systematic training in deep learning, probabilistic modeling, information theory, GPU training, or distributed systems.

## Core Thread

The course is organized around one central question:

> If we want a model to generate reliable answers from context, what capabilities do we need to add one by one, and how can experiments, tests, and failure cases validate those capabilities?

That thread is divided into five layers:

1. **Optimization layer**: how a model updates its parameters from mistakes through loss, computation graphs, gradients, and optimizers.
2. **Sequence-modeling layer**: how a language model turns "continue this sentence" into next-token probability.
3. **Representation and architecture layer**: how tokenizers, embeddings, attention, and Transformers turn text into computable, trainable, composable context representations.
4. **Adaptation and augmentation layer**: how SFT, LoRA/QLoRA, RAG, and distillation turn a general model into a useful task model.
5. **Engineering governance layer**: how evaluation, safety, model cards, deployment, and domain project templates make model behavior verifiable, traceable, and iterative.

The curriculum reuses one running example whenever possible: starting from risk detection in contract clauses, then moving toward legal QA with RAG, domain fine-tuning, evaluation, safe refusal behavior, and model cards. Medical QA appears as a second high-risk domain case to emphasize safety boundaries and human review.

## What This Curriculum Does Not Do

- It does not start from Python syntax.
- It does not reduce LLMs to prompt techniques.
- It does not only provide demos that happen to run.
- It does not treat "the loss went down" as a substitute for serious evaluation.
- It does not blur RAG, LoRA, and distillation into one vague bucket.

## Standard Deliverables for Each Chapter

Each chapter contains at least:

- `lessons/xx_*.md`: the main lesson text, written in a question-driven style.
- `notebooks/xx_*.ipynb`: interactive experiments.
- `src/`: reusable implementations.
- `tests/`: behavioral acceptance tests.
- A failed experiment for the chapter: common mistakes and how to diagnose them.

Each chapter must answer:

1. What capability gap are we addressing?
2. What are the key mathematical objects?
3. What are the shapes of the inputs, outputs, parameters, and loss?
4. What is the minimal implementation?
5. How do we design experiments to observe it?
6. What are the common failure modes?
7. How do the tests prove that the implementation is still correct?

## Chapter 1 Learning Contract

Chapter 1 is not about "getting an MLP to run." It establishes the training habits that every later chapter will reuse:

- Training loop: `forward -> loss -> backward -> optimizer.step`.
- Computation graph: understanding what autograd records and what it backpropagates.
- train/val: training-set performance is not the same as generalization.
- Reproducible experiments: fixed seeds, data splits, shuffling, and initialization.
- overfit tiny: if the model cannot overfit a tiny dataset, the training pipeline probably has a bug.
- Tests are not smoke tests: tests should verify parameter updates, decreasing loss, no gradients during eval, and differences between train and eval modes.

## Quick Start

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate llm-from-zero
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest -q
```

Run the Chapter 1 basic training experiment:

```bash
python -m src.training.simple_mlp --experiment baseline
```

Run the Chapter 1 tiny-data overfitting experiment:

```bash
python -m src.training.simple_mlp --experiment overfit_tiny
```

Run notebooks:

```bash
jupyter lab
```

Then open the notebook for the relevant chapter, such as `notebooks/01_tensor_autograd.ipynb`.

## Repository Layout

- `lessons/`: Chinese lesson text written in a question-driven style.
- `notebooks/`: teaching notebooks designed to be read and run side by side.
- `src/`: reusable code that evolves from minimal training modules into language models, fine-tuning, RAG, distillation, and evaluation.
- `tests/`: pytest checks for shapes, numerical behavior, training behavior, reproducibility, and failure modes.
- `projects/`: domain project templates, such as legal contract review and medical QA assistants.
- `reports/`: model evaluation reports and model card templates.
- `workflows/`: Pro + Codex collaborative writing workflows.

## Current Status

- The curriculum project skeleton has been created, along with the Pro + Codex writing workflow.
- The 19 course articles listed in `roadmap.md` are complete, with the main text written in a question-driven structure.
- Chapters 1-19 now include lesson text, notebooks, `src`, and test scaffolds, covering the training loop, language modeling, tokenizers, attention, Transformers, fine-tuning, LoRA, domain data, RAG, distillation, evaluation, safety governance, deployment, and domain project templates.
- The current focus is to make each chapter's correct experiments, intentionally failed experiments, pytest gates, notebook observations, and command-line acceptance checks more concrete, and to prove delivery status with `pytest -q` and report artifacts.
- `projects/` and `reports/` now include legal contract review, medical QA assistant, a domain small-model template, evaluation reports, and model card examples.
- Deliverables are currently accepted through local tests and static checks. Individual articles are not being submitted to Pro for chapter-by-chapter review yet; they can be packaged later by the user for a full Pro review.
