Languages: [中文](../roadmap.md) | English | [日本語](../ja/roadmap.md)

# Roadmap: From Zero to Small Domain Models with LLMs

This course starts from the intuition behind training and eventually leads to the engineering of small domain models for areas such as law and medicine. The goal is not to read scattered tutorials, but to build a learning project that can be maintained, run, tested, and reproduced over time.

## Main Thread

```text
training loop
  -> next-token language modeling
    -> tokenizer and dataset
      -> embedding and fixed-context models
        -> causal self-attention
          -> transformer block
            -> mini GPT
              -> open-source model workflow
                -> SFT instruction tuning
                  -> LoRA / QLoRA
                    -> domain data engineering
                      -> RAG
                        -> distillation
                          -> evaluation
                            -> safety and model cards
                              -> quantized deployment
                                -> domain model projects
```

In one sentence:

> Start from a minimal, verifiable training loop and understand why language modeling is next-token prediction. Then add text digitization, contextual representations, dynamic attention, and Transformers step by step. Finally, use data, fine-tuning, RAG, distillation, evaluation, safety, and deployment to turn the model into a runnable, testable, traceable small domain model.

## Running Example

To keep the second half of the course from turning into a checklist of engineering topics, the course keeps three contract-risk examples running throughout:

```text
Excessive liquidated damages: 合同 违约金 过高 ， 它 可能 存在 风险
Overly broad liability: compensate all losses, including indirect losses, lost profits, and attorney fees
Insufficient information: missing jurisdiction, regulation version, contract type, or retrieval evidence
```

Chapters 1-7 use these examples to observe the training loop, next-token prediction, tokenizers, embeddings, attention, and Mini GPT. Chapters 8-13 migrate the same examples into Hugging Face, SFT, LoRA, domain data, RAG, and distillation. Chapters 14-19 then turn them into evaluation items, safety gates, deployment logs, and a domain template.

Every chapter should be able to answer one question:

```text
What new capability does this chapter add that brings the system closer to cautious contract-clause risk detection?
```

## Component Boundaries at a Glance

Remember this table before studying Chapters 11-13, so fine-tuning, retrieval, and distillation do not blur into one thing:

| Component | Main Role | What It Cannot Replace |
| --- | --- | --- |
| SFT | Teaches instruction format, output structure, and behavior boundaries | Cannot guarantee facts or citations are real |
| LoRA / QLoRA | Reduces fine-tuning cost and makes adapters easier to save and roll back | Cannot compensate for dirty data or vague tasks |
| Domain data engineering | Defines tasks, sources, licenses, de-identification, labels, and eval | Cannot automatically make the model use fresh external knowledge |
| RAG | Provides updatable evidence and citation links | Cannot guarantee the model will use evidence correctly |
| Distillation | Transfers verifiable behavior from a stronger model to a student | Cannot treat the teacher as a source of truth |
| Evaluation / safety | Exposes failures, defines release gates, and sets human-review boundaries | Cannot automatically fix model failures |

## Minimal Experiment Loop for Each Chapter

Each chapter can have extended experiments, but the main thread only needs one minimal loop:

| Chapter | Main Experiment |
| --- | --- |
| 1 | Contract toy classifier: baseline / no step / no zero_grad / overfit tiny |
| 2 | Bigram LM: correct shift vs incorrect shift |
| 3 | Tokenizer + collator: whether padding enters loss |
| 4 | Current-token LM vs causal-mean LM |
| 5 | Attention with causal mask vs without causal mask |
| 6 | Compare loss and grad_norm across 1 / 2 / 4 blocks |
| 7 | MiniGPT tiny-corpus checkpoint round trip |
| 8 | Tiny HF model: load / generate / train one step / save_pretrained |
| 9 | Overfit 20 SFT examples to a JSON output format |
| 10 | Compare LoRA rank and target_modules |
| 11 | Generate SFT / RAG / distill / eval forms for the same contract clause |
| 12 | Top-k retrieval and citation support for 5 queries |
| 13 | Teacher-sample filtering: pass rate and reject_reason |
| 14 | Generate eval_report and failure_cases from 5 eval items |
| 15 | Safety policy blocks high-risk out-of-bound outputs |
| 16 | Compare quality and latency for the same model in fp16 / int8 / int4 |
| 17 | End-to-end risk JSON output for one contract clause |
| 18 | Chest pain + shortness of breath triggers red flags |
| 19 | Release gate blocks versions missing reports or rollback targets |

## Stage 0: Project Setup and Learning Method

Goal: build a long-term maintainable learning project, not a pile of disconnected tutorials.

Core content:

- Project structure
- Learning path
- Environment setup
- Notebook conventions
- Testing conventions
- Textbook-writing conventions

Deliverables:

- `README.md`
- `roadmap.md`
- `AGENTS.md`
- `pyproject.toml`
- `environment.yml`
- `lessons/`
- `notebooks/`
- `src/`
- `tests/`
- `projects/`

## Stage 1: PyTorch and Training Intuition

Core question: how does a neural network actually "learn"?

Topics:

- Tensor
- shape
- parameters
- loss
- gradient
- backward
- optimizer
- `nn.Module`
- `Dataset`
- `DataLoader`
- training loop

Deliverables:

- An MLP classifier
- A complete training loop
- A pytest test file

Related chapter: `01_pytorch_training_intuition`

## Stage 2: Language Model Foundations

Core question: if the model is not classifying images but continuing a sentence, how should it be trained?

Topics:

- next-token prediction
- offset between `input_ids` and `labels`
- cross entropy
- logits
- softmax
- sampling
- temperature
- top-k / top-p
- bigram language model

Deliverables:

- A bigram language model
- A text-generation function `generate()`
- A tiny Chinese-corpus training experiment

Related chapter: `02_language_modeling`

## Stage 3: Tokenizers and Dataset Construction

Core question: models cannot consume Chinese text directly, so how does text become numbers?

Topics:

- token
- vocab
- token id
- encode / decode
- padding
- truncation
- attention mask
- BPE intuition
- WordPiece intuition
- language modeling dataset
- chat / SFT data format

Deliverables:

- A simple tokenizer
- An LM Dataset
- An SFT data-format example

Related chapter: `03_tokenizer_and_dataset`

## Stage 4: Embeddings and Neural Language Models

Core question: token ids are only numbers, so how does the model learn meaning from them?

Topics:

- embedding table
- embedding lookup
- hidden dimension
- context window
- neural language model
- parameter update

Deliverables:

- An embedding-based language model
- An experiment that makes embedding-parameter changes observable

Related chapter: `04_embedding_and_neural_lm`

## Stage 5: Attention

Core question: how does each token in a sentence decide what it should look at?

Topics:

- Q / K / V
- scaled dot-product attention
- attention weights
- causal mask
- self-attention
- attention visualization

Deliverables:

- Hand-written scaled dot-product attention
- Causal-mask verification
- Attention-weight visualization

Related chapter: `05_attention`

## Stage 6: Transformer Block

Core question: attention is only one module. What else does a complete LLM block need?

Topics:

- multi-head attention
- feed-forward network
- residual connection
- LayerNorm / RMSNorm
- dropout
- pre-norm / post-norm
- Transformer block

Deliverables:

- Hand-written Multi-Head Attention
- Hand-written Transformer Block
- Input/output shape tests

Related chapter: `06_transformer_block`

## Stage 7: Implementing Mini GPT from Scratch

Core question: if we combine a tokenizer, embeddings, Transformer blocks, and an LM head, do we get GPT?

Topics:

- decoder-only architecture
- positional embeddings / RoPE intuition
- LM head
- causal language modeling
- training mini GPT
- text generation
- checkpoint saving and loading

Deliverables:

- A trainable mini GPT
- A training script
- A text-generation script
- Checkpoint save/load

Related chapter: `07_mini_gpt`

## Stage 8: Hugging Face Workflow

Core question: in real work, we cannot rewrite the model from scratch every time. How do we use open-source models?

Topics:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `datasets`
- `Trainer`
- Accelerate
- `model.generate`
- model saving
- model loading
- chat template

Deliverables:

- Load a small open-source model
- Run one inference
- Complete one minimal fine-tuning step
- Save model results

Related chapter: `08_huggingface_workflow`

## Stage 9: SFT Instruction Tuning

Core question: how do we turn a model from "continuing text" into "following instructions"?

Topics:

- instruction tuning
- SFT
- system / user / assistant messages
- chat dataset
- data cleaning
- train / val / test split
- format consistency
- overfitting observation

Deliverables:

- An SFT dataset
- An SFT training script
- A before/after comparison

Related chapter: `09_sft_instruction_tuning`

## Stage 10: LoRA / QLoRA Parameter-Efficient Fine-Tuning

Core question: full fine-tuning is expensive. Can we train only a small number of parameters?

Topics:

- PEFT
- LoRA
- rank
- alpha
- target_modules
- adapter
- merge adapter
- QLoRA
- 4-bit quantization
- memory optimization

Deliverables:

- A LoRA fine-tuning script
- A QLoRA fine-tuning script
- An adapter save/load workflow

Related chapter: `10_lora_qlora`

## Stage 11: Domain Data Engineering

Core question: where does the capability of a domain model mainly come from: the model, or the data?

Topics:

- domain data collection
- data licenses and usage boundaries
- data cleaning
- deduplication and near-duplicate checks
- de-identification
- quality filtering
- instruction-data construction
- RAG chunk construction
- distillation-data construction
- evaluation-data construction and eval-set freezing
- legal/medical data risks

Deliverables:

- Domain SFT data format
- Domain eval-set format
- Data-cleaning scripts
- Data quality report
- Data manifest

Related chapter: `11_domain_data_engineering`

## Stage 12: RAG Retrieval-Augmented Generation

Core question: model parameters are not a database. How can the model look things up before answering?

Topics:

- chunking
- embedding model
- vector store
- retriever
- top-k retrieval
- rerank intuition
- prompt with context
- citation
- RAG hallucination

Deliverables:

- A local RAG baseline
- A small knowledge base
- A retrieval + generation pipeline
- Output with cited sources

Related chapter: `12_rag_baseline`

## Stage 13: Distilling Small Models

Core question: large models perform well but cost too much. How do we transfer capability to a smaller model?

Topics:

- teacher model
- student model
- response distillation
- logit distillation intuition
- preference distillation
- distillation-data generation
- distillation-data filtering
- student training

Deliverables:

- A teacher-data generation script
- A student-training script
- A base / teacher / student comparison

Related chapter: `13_distillation`

## Stage 14: Model Evaluation

Core question: the model looks fluent. How do we prove it has actually improved?

Topics:

- eval set
- automatic evaluation
- human scoring
- format accuracy
- factual accuracy
- refusal ability
- hallucination tests
- RAG citation accuracy
- legal/medical safety evaluation

Deliverables:

- `eval_runner.py`
- `metrics.py`
- `eval_report.md`
- Table of high-risk failure cases

Related chapter: `14_evaluation`

## Stage 15: Safety, Compliance, and Model Cards

Core question: legal and medical domain models cannot just optimize for plausible answers. They must also know when not to answer.

Topics:

- data de-identification
- privacy protection
- refusal boundaries
- uncertainty expression
- safety prompts
- legal disclaimers
- medical disclaimers
- model card
- risk report
- human review

Deliverables:

- `model_card_template.md`
- `risk_report.md`
- Safety test set
- Refusal test set

Related chapter: `15_safety_and_model_card`

## Stage 16: Quantization and Deployment

Core question: once the model is trained, how do we make verifiable engineering trade-offs among cost, latency, throughput, quality, and safety?

Topics:

- FP32 / FP16 / BF16
- INT8
- INT4
- bitsandbytes
- GGUF intuition
- vLLM intuition
- API server
- batching
- latency
- throughput
- quality / safety regression
- release gate
- rollback

Deliverables:

- Quantized inference script
- Local API server
- Simple benchmark
- Release gate and rollback configuration

Related chapter: `16_quantization_and_serving`

## Stage 17: Legal Small Domain Model Project

Core question: how do we combine fine-tuning, RAG, distillation, and evaluation into a legal model?

Project directions:

- Contract-risk detection
- Clause explanation
- Contract revision suggestions
- Legal QA RAG
- Legal-reference citation checks

Deliverables:

```text
legal_contract_review/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

Related chapter: `17_legal_domain_project`

## Stage 18: Medical Small Domain Model Project

Core question: how do we build a careful, safe, and evaluable medical education assistant?

Project directions:

- Medical education QA
- Symptom explanation
- Care-seeking guidance
- Medical guideline RAG
- Red-flag detection
- Not replacing physician diagnosis

Deliverables:

```text
medical_qa_assistant/
├── data/
├── sft/
├── rag/
├── distill/
├── eval/
├── train_lora.py
├── rag_pipeline.py
├── evaluate.py
└── model_card.md
```

Related chapter: `18_medical_domain_project`

## Stage 19: Complete Engineering Template for Domain Models

Core question: how do we turn a domain model project into a reusable template?

Topics:

- Project directory conventions
- Data version management
- Training configuration management
- Experiment records
- Evaluation reports
- Model release
- Inference service
- Continuous iteration

Deliverables:

```text
domain_model_template/
├── configs/
├── data/
├── scripts/
├── src/
├── tests/
├── reports/
└── README.md
```

Related chapter: `19_domain_model_template`

## Recommended Short Learning Order

If you only follow the main path and skip extensions, the order is:

1. PyTorch training intuition
2. Language model foundations
3. Tokenizer and Dataset
4. Embeddings and neural language models
5. Attention
6. Transformer Block
7. Mini GPT
8. Hugging Face workflow
9. SFT instruction tuning
10. LoRA / QLoRA
11. Domain data engineering
12. RAG
13. Distillation
14. Evaluation
15. Safety and model cards
16. Quantized deployment
17. Complete domain model project

## The First Chapters to Build

For the first batch, build only these 5 chapters:

1. `01_pytorch_training_intuition`
2. `02_language_modeling`
3. `03_tokenizer_and_dataset`
4. `04_embedding_and_neural_lm`
5. `05_attention`

These five chapters form the smallest loop:

```text
how models train
  -> how text generation is defined
    -> how text becomes numbers
      -> how numbers become vectors
        -> how tokens look at one another
```

After these five chapters, move on to:

- `06_transformer_block`
- `07_mini_gpt`
- `08_huggingface_workflow`
- `09_sft_instruction_tuning`
- `10_lora_qlora`

## Final Graduation Project

Choose one of two graduation projects.

### Track A: Small Model for Legal Contract Review

Input: contract clause.

Output:

- Risk level
- Risk points
- Evidence
- Revision suggestions
- Uncertainty note

Technical combination:

- RAG
- LoRA
- Distillation
- Evaluation
- Model card

### Track B: Small Model for Medical Education QA

Input: user's medical question.

Output:

- Plain-language explanation
- Possible causes
- When to seek care
- Risk reminders
- Not a substitute for physician diagnosis

Technical combination:

- RAG
- SFT
- Safety refusal
- Distillation
- Evaluation
