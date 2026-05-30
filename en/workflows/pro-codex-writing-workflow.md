Languages: [中文](../../workflows/pro-codex-writing-workflow.md) | English | [日本語](../../ja/workflows/pro-codex-writing-workflow.md)

# Pro + Codex Textbook Writing Workflow

This workflow is for reusing the current collaboration pattern:

- ChatGPT Pro handles high-quality course design, article structure, technical review, and rewrite suggestions.
- Codex handles browser control, context organization, local file edits, code implementation, test verification, and preserving engineering results.

Suitable scenarios:

- Writing a systematic textbook.
- Upgrading existing drafts into professional articles.
- Letting Pro act first as chief editor/reviewer, then having Codex land the changes in the repository.
- Continuously maintaining lessons, notebooks, src, tests, reports, and related materials in a local repo.

## 1. Division of Labor

### Pro's Role

Pro is not used to "write files for Codex" directly. It is used for high-level judgment:

- Course positioning.
- Chapter path.
- Problem-chain design.
- Technical-depth calibration.
- Article structure and paragraph rewrites.
- Review: point out where content is shallow, wrong, or missing experiments and acceptance criteria.
- Give Codex an executable revision checklist.

### Codex's Role

Codex does not copy Pro's answer directly into the repository. It is responsible for engineering execution:

- Control the ChatGPT page with Chrome.
- Organize the current repo content into context that Pro can review.
- Control the number of Pro pages to avoid wasting Pro conversations and quota.
- Wait for Pro's full answer instead of frequently opening new pages.
- Extract actionable items from Pro's answer.
- Edit local Markdown, notebooks, Python code, and tests.
- Run `pytest`, `ruff`, and example scripts.
- Write reusable processes back into the workflow.

## 2. Basic Loop

Every writing or revision pass follows this loop:

```text
local draft/code
-> Codex summarizes context
-> send to ChatGPT Pro for review or design
-> wait for Pro's full suggestions
-> Codex extracts a revision checklist
-> Codex edits local files
-> Codex runs tests and checks
-> Codex reports results
```

Do not let Pro and Codex each write separate versions without seeing each other's work. Pro must see the current version; otherwise it can only give generic suggestions.

## 3. Rules for Using Chrome

### Do Not Use MCP as a Substitute for the User's Page

This workflow enforces the following:

- Codex must not use an MCP browser, DevTools MCP, or any other MCP page-control tool as a substitute for the Chrome page already opened by the user.
- If the user explicitly says they already opened a page with `@chrome`, Codex must first acknowledge and reuse that user-page context. It must not open a separate MCP/isolated browser page and then conclude "not logged in" or "unavailable."
- If the current environment has no non-MCP Chrome control capability, Codex must state the limitation directly and fall back to a local-file workflow; it must not quietly switch to MCP.
- When the user explicitly says "do not use MCP," that constraint overrides every browser-automation suggestion in this workflow.

### Prefer Reusing Existing Pages

Do not casually open many Pro pages. Priority order:

1. If there is an existing relevant ChatGPT conversation, continue in that conversation.
2. If the original conversation is stuck but still thinking, keep waiting.
3. Open a new page only when the current page is truly unrecoverable, the topic is completely different, or the user explicitly asks for a new page.

### Control Page Count

- Keep only one Pro conversation page for the same task whenever possible.
- When using Pro to write course content, ask one chapter at a time: submit only one chapter's context, problem chain, code, and acceptance goals per round.
- Do not ask Pro through multiple pages in parallel. Do not open multiple Pro pages for the same batch of chapters.
- Do not ask Pro to write multiple chapters of full text at once. Multi-chapter work must be split into sequential loops: finish landing and verifying Chapter N locally, then ask about Chapter N+1.
- If extra pages are opened by mistake, Codex should close the useless pages and keep the one with the most context.
- After submitting a long context, if ChatGPT shows it as an attachment or pasted file, confirm that the prompt was actually sent.

### Waiting Strategy

Do not resend immediately when Pro is thinking for a long time.

Suggested rhythm:

1. Wait 60-90 seconds after sending.
2. If it only shows "thinking," wait another 60-120 seconds.
3. If a tool read times out, reread the page state before opening a new page.
4. If Pro produces no output for several minutes, ask the user whether to keep waiting or switch to a shorter context.

## 4. Context Format for Pro

Pro's input must include three layers of context:

### Task Goal

Explain what role Pro should play, for example:

```text
You are the Pro-level chief editor of an LLM textbook and a strict technical reviewer.
Please point out where the current textbook content is shallow and give Codex an executable rewrite checklist.
```

### User Preferences

State the user's preferences clearly:

```text
The user already knows Python and does not need a Python introduction.
The writing style should be problem-driven:
real problem -> limitation of the old method -> new mechanism -> boundary -> next-chapter problem.
The goal is not popular science, but a runnable, testable, reproducible professional textbook project.
```

### Current Local Content

Let Pro see the current version. Include at least:

- `README.md`
- `roadmap.md`
- the current `lessons/xx_*.md`
- related `src/**/*.py`
- related `tests/**/*.py`

Send in this format:

```text
Below is the current content:

===== README.md =====
...

===== roadmap.md =====
...

===== lessons/01_xxx.md =====
...

===== src/... =====
...

===== tests/... =====
...
```

## 5. Pro Review Prompt Template

```text
You are now the "Pro-level chief editor of an LLM textbook + strict technical reviewer."

Below is the textbook engineering content Codex has already written. User feedback: it is too simple, more like a beginner skeleton, and not professional enough.

Please do not give a generic new outline. Instead, critique and revise based on the existing content below. Requirements:

1. First list where this version is shallow: route, chapter problem chains, technical depth, experiment design, code quality, and test design.
2. Give Pro-level rewrite principles: keep the problem-questioning style, but raise it enough to support systematic LLM learning.
3. Explain how README and roadmap should be changed.
4. Give a rewrite outline and key paragraphs for the current chapter.
5. Explain how the code should be upgraded: modules, functions, config, history, experiments, tests.
6. Explain which professional acceptance tests should be added.
7. Explain how later chapters should upgrade the problem chain.
8. End with a Codex-executable revision checklist, ordered by priority.

Use Chinese. Critique directly. Do not be polite for politeness' sake.

Below is the current content:
...
```

## 6. Prompt Template for Having Pro Write Article Structure

When the task is not review but asking Pro to produce article structure, use this template:

```text
You are a Chinese technical textbook author. Please write in a "problem-questioning" style.

Topic:
[fill in topic]

Target readers:
They know Python, but are not assumed to know deep learning / probability / information theory / GPU / distributed training.

Core question:
[the real question this chapter must answer]

Running example:
[one example used from beginning to end]

Please output:
1. problem chain: original state -> problem -> new mechanism -> what it solves -> new boundary -> next-chapter question.
2. concept card: core concept, mathematical object, shape, code object, experiment object.
3. outline: chapter structure.
4. draft: full Chinese draft.
5. Codex landing checklist: which files to edit, which tests to write, how to verify.

Writing requirements:
- Explain why it appears before explaining what it is.
- Every key concept must include intuition, mathematical object, shape, code implementation, and failure modes.
- Do not write a glossary.
- Do not stop at popular science; it must guide local implementation and tests.
```

## 7. Codex Landing Rules

After receiving Pro's output, Codex does not copy it directly. It should:

1. Extract the highest-priority changes.
2. Update `README.md` and `roadmap.md` so the project positioning is aligned first.
3. Revise the current chapter body and deepen the problem chain.
4. Update the corresponding `src/` code.
5. Update the corresponding `tests/`.
6. Run verification:

```bash
pytest -q
ruff check .
python -m <module>
```

7. If verification fails, fix the local implementation first. Ask Pro again only when the technical judgment is unclear.

## 8. Textbook Quality Bar

Each chapter must answer:

1. What capability does this chapter add?
2. What is the mathematical object behind this capability?
3. What are the shapes of the inputs, outputs, parameters, and loss?
4. Where is the minimal implementation?
5. What is the success experiment?
6. What is the failure experiment?
7. How do tests prove it is not broken?
8. Why does the next chapter naturally appear?

If a chapter only answers "what the concept is," it is not professional enough.

## 9. Professional Sample Requirements for Chapter 1

Chapter 1 must not be only a toy MLP. It should become the engineering sample for all later chapters.

It must include:

- Training loop.
- Computation graph.
- Local intuition for the chain rule.
- Finite-difference gradient check.
- train/val split.
- overfit tiny experiment.
- initialization.
- learning rate.
- batch size.
- random seeds and reproducible experiments.
- `model.train()` vs `model.eval()`.
- `torch.no_grad()`.
- tests as learning acceptance, not smoke tests.

The code should include at least:

- `TrainingConfig`
- `EpochMetrics`
- `HistoryItem`
- `TrainingHistory`
- `set_seed`
- `split_dataset`
- `make_dataloaders`
- `compute_grad_norm`
- `compute_update_norm`
- `train_one_epoch`
- `evaluate`
- `run_training`
- `run_overfit_tiny_experiment`

Tests should cover at least:

- dataset shape.
- forward shape.
- loss clearly decreases.
- parameters actually update.
- evaluate does not produce gradients.
- fixed seed is reproducible.
- tiny sample can be overfit.
- train/eval dropout behavior differs.
- train/val split has no overlap.
- grad norm and update norm are greater than 0.

## 10. Common Mistakes

### Mistake 1: Not Giving Pro the Current Content

Saying only "help me make this more professional" produces generic outlines. You must send the current README, roadmap, lesson, code, and tests.

### Mistake 2: Opening Too Many Pro Pages

Too many Pro pages fragment context and waste quota. Keep one full-context page for one task.

### Mistake 3: Copying Pro's Output Directly

Pro's output is editorial advice and material, not the final repository state. Codex is responsible for integration, deduplication, and verification.

### Mistake 4: Writing Articles Without Tests

The goal of this textbook project is "readable, runnable, modifiable, reproducible." A chapter without tests is not complete.

### Mistake 5: Treating Wait Timeouts as Failure

A Chrome tool timeout does not mean Pro has stopped. Reread page state before deciding the next step.

## 11. End-of-Round Report Format

Codex reports back to the user:

```text
Completed:
- Which files were changed
- What key review feedback Pro gave
- How I landed that feedback

Verification:
- pytest -q result
- ruff check . result
- example script result

Next:
- Which chapter to continue with
- What Pro should review next
```
