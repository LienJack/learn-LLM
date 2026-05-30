Languages: [中文](../../workflows/gpt-pro-review-prompt.md) | English | [日本語](../../ja/workflows/gpt-pro-review-prompt.md)

# GPT Pro Course Review Prompt

After pasting the full text of `all-articles-for-gpt-review.md` into GPT Pro, use the prompt below.

```markdown
You are a senior Chinese technical textbook editor, LLM engineering course designer, and machine-learning education reviewer. Please review the following 19-article course collection, "From Zero to Small Domain Models with LLMs," and provide systematic editorial feedback and revision suggestions.

## Course Positioning

This is not an API tutorial, and it is not a high-level popular-science article that only explains concepts. It is a Chinese textbook for learners who already know Python but may not have systematically studied deep learning or LLM engineering.

The course goal is to help learners start from a minimal training loop and gradually understand:

training loop -> next-token language modeling -> Tokenizer / Dataset -> Embedding -> Attention -> Transformer -> Mini GPT -> Hugging Face workflow -> SFT -> LoRA / QLoRA -> domain data engineering -> RAG -> distillation -> evaluation -> safety and model cards -> quantized deployment -> legal / medical small domain model projects -> general engineering template for domain models

The core thread of this course is:

> If we want a model to generate reliable answers from context, which capabilities must be added in sequence? How are those capabilities verified through experiments, tests, and failure cases?

## Target Readers

- They know Python and can read basic PyTorch code.
- They are not assumed to have systematically studied deep learning, probabilistic modeling, information theory, GPU training, or distributed systems.
- They want to move from "I can run a demo" toward "I can train, evaluate, deploy, and govern an LLM system."

## Writing Style Requirements

Please review the course as a problem-driven technical concept blog / textbook, not as an encyclopedia entry or academic paper.

Each chapter should prioritize:

1. Explain why the concept appears before explaining what it is.
2. Start from a real learner confusion or capability gap.
3. Use a clear chain of problem evolution:
   original situation -> what problem appears -> what mechanism is introduced -> what it solves -> what new problem remains -> how it leads to the next chapter.
4. Reuse the running example as much as possible, especially "contract-clause risk detection / legal contract review"; use medical QA as a second high-risk domain example.
5. When a term appears, first translate it into plain language, then define its technical boundary.
6. Do not write parallel definition lists. Do not pile up terminology.
7. Keep each chapter engineering-oriented: shapes, inputs/outputs, loss, minimal implementation, experimental observation, failure modes, and test acceptance.
8. Be technically honest: state applicability boundaries, non-goals, common misunderstandings, and high-risk scenarios.

## Please Focus On

### A. Overall Structure

- Is the 19-chapter evolution chain natural?
- Is there a clear progression from Chapter 1 to Chapter 19, rather than a set of parallel topics?
- Which chapter transitions are weak and need "the previous chapter left this problem / this chapter solves it"?
- Which chapter order, titles, or responsibilities need adjustment?
- Are there repetitions, breaks, skipped steps, or concepts that appear before being explained?

### B. Individual Chapter Quality

Please inspect each chapter:

- Does the opening start from a real problem?
- Is "the real problem this chapter solves" sharp enough?
- Does the "problem chain" support the whole chapter?
- Is the Concept Card useful, or is it just a formal table?
- Is the running example consistent, concrete, and runnable?
- Are any conceptual explanations too abstract, too textbook-like, or too much like stacked definitions?
- Where should analogies, counterexamples, boundary notes, or minimal examples be added?
- Does the ending naturally lead into the next chapter?

### C. Beginner Readability

- Which paragraphs move too fast, skip too much, or are too dense with terminology for the target reader?
- Which mathematical objects, shapes, losses, or training details need a gentler ramp?
- Where should "engineering terms" be translated into more intuitive plain language?
- Which diagrams, flowcharts, tables, or minimal code snippets would significantly improve understanding?

### D. Engineering Loop

Please check whether each chapter truly answers:

- What is the current capability gap?
- What is the key mathematical object?
- What are the shapes of the inputs, outputs, parameters, and loss?
- What is the minimal implementation?
- How can it be observed through experiments?
- What are the common failure modes?
- How do tests prove the implementation is not broken?

If a chapter lacks any of these elements, clearly state where it should be added.

### E. Small Domain Model Thread

Please especially check whether the thread from general LLM foundations to legal / medical small domain model engineering is strong enough:

- Does the running example of contract-risk detection keep working from Chapter 1 onward?
- Are the legal and medical projects integrated validations of earlier capabilities, rather than extra appendices?
- Are the boundaries among RAG, SFT, LoRA, distillation, evaluation, safety, and deployment explained clearly?
- Is there a risk of mixing RAG, fine-tuning, and distillation into one vague thing?
- Are refusal, human review, evidence citation, model cards, and rollback in high-risk domains treated seriously enough?

## Output Format

Please do not fully rewrite all 19 articles. First provide a review report and revision suggestions.

Use the following structure:

## 1. Overall Judgment

Use 5-10 sentences to evaluate the course's biggest strengths, biggest risks, and the highest-priority direction for revision.

## 2. Overall Structural Revision Suggestions

List 5-10 cross-chapter suggestions. Each item should include:

- Problem
- Why it affects the learning experience
- Suggested revision
- Affected chapters

## 3. Chapter-by-Chapter Review

Give feedback for Chapters 1-19 one by one. For each chapter, include:

- Whether the chapter's role is clear
- Whether the problem chain holds
- The 1-3 points that most need strengthening
- Suggested additions / deletions / moves
- Whether examples, diagrams, experiments, tests, or failure modes need to be added

## 4. Prioritized Revision Checklist

Use P0 / P1 / P2:

- P0: must be fixed because it affects the course's main thread or technical correctness
- P1: clearly improves the learning experience and article quality
- P2: polish, wording, titles, and local example optimization

## 5. Style Unification Suggestions

Provide a reusable single-chapter writing template for unifying all 19 articles. The template should follow the style of "problem-driven, concept evolution chain, running example, engineering loop."

## 6. Example Rewrites

Choose 2-3 representative paragraphs and demonstrate how to rewrite them from "textbook-like / definition-style expression" into "problem-driven, beginner-friendly, engineering-boundary-aware expression."

## 7. Next Actions for the Author

Give a practical revision order: which chapters to revise first, how to revise them, and how each revision round should be accepted.

## Review Principles

- Technical correctness comes before prose.
- A clear learning path comes before covering more concepts.
- Do not suggest adding too much external knowledge unless it solves a current break or misunderstanding in the articles.
- Do not turn the articles into marketing copy, viral social-media writing, or a literature review.
- Keep the Chinese natural, clear, and low in terminology density, without sacrificing engineering accuracy.
- If some content is already good enough, say so explicitly. Do not invent criticism just to provide suggestions.
```
