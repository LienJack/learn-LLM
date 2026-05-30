Languages: [中文](../../image-prompts/llm-course-article-image-prompts.md) | English | [日本語](../../ja/image-prompts/llm-course-article-image-prompts.md)

# Image Prompt Plan for the LLM Course Articles

Reference source: `/Users/lienli/Documents/GitHub/learn-agent/images/workflows/blog-to-photo.md` and the `blog-to-photo` skill.

Constraint: each chapter has 3 in-article technical explanation images, for 19 chapters and 57 images total. The images are intended for later Chinese, English, and Japanese localization, so each prompt explicitly requires no readable text inside the image. Draw only icons, numbered dots, blank label strips, and localization-friendly whitespace.

## Chapter 1: Training Loop, Computation Graph, and Reproducible Experiments

- Article: `lessons/01_pytorch_training_intuition.md`
- Suggested asset directory: `lessons/assets/01_pytorch_training_intuition`

### photo-01-training-loop: The Minimal Training Loop

- Target path: `lessons/assets/01_pytorch_training_intuition/photo-01-training-loop.png`
- Diagram structure: circular flowchart from data entering the model to parameter update and back to the next batch.
- Highlight: Loss and optimizer update

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Minimal Training Loop.

Composition: circular flowchart from data entering the model to parameter update and back to the next batch. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: data batch as small table cards; forward pass as a neural module; loss as a gauge; backprop as a reverse arrow through a computation graph; optimizer update as a wrench turning a parameter knob; next batch as a loop arrow..

Highlight: Use pale yellow to emphasize Loss and optimizer update.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-computation-graph: How the Computation Graph Records Tensor Relationships

- Target path: `lessons/assets/01_pytorch_training_intuition/photo-02-computation-graph.png`
- Diagram structure: left-to-right state sequence showing a tensor moving from ordinary values into a backpropagatable object.
- Highlight: `grad_fn` and gradient flow

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: How the Computation Graph Records Tensor Relationships.

Composition: left-to-right state sequence showing a tensor moving from ordinary values into a backpropagatable object. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: input tensor grid; operator gear node; intermediate tensor card; `grad_fn` marker with a tail; leaf parameter anchor; gradient-flow return arrow..

Highlight: Use pale yellow to emphasize `grad_fn` and gradient flow.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-reproducible-experiment: Reproducible Experiments and Failure Diagnosis

- Target path: `lessons/assets/01_pytorch_training_intuition/photo-03-reproducible-experiment.png`
- Diagram structure: layered checklist, with training settings above and validation signals below.
- Highlight: overfit tiny and pytest gate

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Reproducible Experiments and Failure Diagnosis.

Composition: layered checklist, with training settings above and validation signals below. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: locked dice for seed; split data folder; train/eval switch; magnifier over tiny samples; descending loss curve; pytest gate..

Highlight: Use pale yellow to emphasize overfit tiny and pytest gate.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 2: The Probabilistic Objective of Language Models

- Article: `lessons/02_language_modeling.md`
- Suggested asset directory: `lessons/assets/02_language_modeling`

### photo-01-next-token: The Training Objective of Next-Token Prediction

- Target path: `lessons/assets/02_language_modeling/photo-01-next-token.png`
- Diagram structure: horizontal pipeline showing text shifted into inputs and labels to predict the next token.
- Highlight: input/label shift and cross entropy

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Training Objective of Next-Token Prediction.

Composition: horizontal pipeline showing text shifted into inputs and labels to predict the next token. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: text tape; shifted `input_ids`; shifted `labels`; multi-exit logits distribution; cross-entropy target; dice plus thermometer for sampling..

Highlight: Use pale yellow to emphasize input/label shift and cross entropy.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-shape-contract: The Shape Contract of a Language Model

- Target path: `lessons/assets/02_language_modeling/photo-02-shape-contract.png`
- Diagram structure: stacked tensor diagram expanded across batch, time, and vocab dimensions.
- Highlight: the 3D logits block

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Shape Contract of a Language Model.

Composition: stacked tensor diagram expanded across batch, time, and vocab dimensions. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: stack of sample cards; timeline ticks; vocabulary drawer; 3D logits block; label index needle; single loss scalar dot..

Highlight: Use pale yellow to emphasize the 3D logits block.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-generation-controls: How Temperature, Top-k, and Top-p Affect Generation

- Target path: `lessons/assets/02_language_modeling/photo-03-generation-controls.png`
- Diagram structure: decision path from a probability distribution into different sampling gates.
- Highlight: temperature knob and filtering gates

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: How Temperature, Top-k, and Top-p Affect Generation.

Composition: decision path from a probability distribution into different sampling gates. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: probability peak curve; temperature knob; top-k candidate boxes; top-p cumulative-area sieve; dice; output-token card..

Highlight: Use pale yellow to emphasize temperature knob and filtering gates.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 3: Tokenizer and Dataset Construction

- Article: `lessons/03_tokenizer_and_dataset.md`
- Suggested asset directory: `lessons/assets/03_tokenizer_and_dataset`

### photo-01-text-to-ids: The Minimal Contract of a Tokenizer

- Target path: `lessons/assets/03_tokenizer_and_dataset/photo-01-text-to-ids.png`
- Diagram structure: input-process-output flow showing text becoming token ids and decoding back to text.
- Highlight: vocab and reversible check

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Minimal Contract of a Tokenizer.

Composition: input-process-output flow showing text becoming token ids and decoding back to text. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: raw text strip; cutting tool; numbered vocab dictionary; string of token-id beads; decode return arrow; reversible-check mirror with a checkmark..

Highlight: Use pale yellow to emphasize vocab and reversible check.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-mask-boundary: The Boundary Between Attention Mask and Label Mask

- Target path: `lessons/assets/03_tokenizer_and_dataset/photo-02-mask-boundary.png`
- Diagram structure: two-column comparison: what the model can see vs what counts toward loss.
- Highlight: label mask and training signal

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Boundary Between Attention Mask and Label Mask.

Composition: two-column comparison: what the model can see vs what counts toward loss. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: gray padding blocks; attention blindfold; prompt span; answer span; loss switch for label mask; highlighted training-signal path..

Highlight: Use pale yellow to emphasize label mask and training signal.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-subword-tradeoff: Trade-Offs Among Character, Word, and Subword Tokenization

- Target path: `lessons/assets/03_tokenizer_and_dataset/photo-03-subword-tradeoff.png`
- Diagram structure: three-way decision diagram comparing length, fragility, and generalization.
- Highlight: composable subword blocks

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Trade-Offs Among Character, Word, and Subword Tokenization.

Composition: three-way decision diagram comparing length, fragility, and generalization. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: many small character particles; large fragile word bricks; composable subword blocks; unknown-word fragments; long measuring tape; generalization bridge..

Highlight: Use pale yellow to emphasize composable subword blocks.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 4: Embeddings and Neural Language Models

- Article: `lessons/04_embedding_and_neural_lm.md`
- Suggested asset directory: `lessons/assets/04_embedding_and_neural_lm`

### photo-01-id-to-vector: Embedding Turns Token IDs into Learnable Vectors

- Target path: `lessons/assets/04_embedding_and_neural_lm/photo-01-id-to-vector.png`
- Diagram structure: input-lookup-output flow showing an id entering an embedding table and producing a vector.
- Highlight: embedding table and vector

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Embedding Turns Token IDs into Learnable Vectors.

Composition: input-lookup-output flow showing an id entering an embedding table and producing a vector. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: numbered token badge; matrix drawer; lookup robot arm; coordinate vector arrows; hidden-dimension scale; small wrench for parameter updates..

Highlight: Use pale yellow to emphasize embedding table and vector.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-causal-context: Causal Aggregation in a Fixed-Context Model

- Target path: `lessons/assets/04_embedding_and_neural_lm/photo-02-causal-context.png`
- Diagram structure: time-sequence diagram showing the current position aggregating only historical tokens.
- Highlight: causal boundary

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Causal Aggregation in a Fixed-Context Model.

Composition: time-sequence diagram showing the current position aggregating only historical tokens. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: historical token tape; sliding context window; one-way causal gate; context-vector funnel; LM head projection; next-token target card..

Highlight: Use pale yellow to emphasize causal boundary.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-embedding-experiment: Observing How Embedding Parameters Change

- Target path: `lessons/assets/04_embedding_and_neural_lm/photo-03-embedding-experiment.png`
- Diagram structure: experiment bench from before/after training comparison to failure diagnosis.
- Highlight: vector movement and nearest-neighbor check

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Observing How Embedding Parameters Change.

Composition: experiment bench from before/after training comparison to failure diagnosis. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: initial vector scatter; small training folder; descending loss chart; vector movement trails; nearest-neighbor magnifier; warning triangle..

Highlight: Use pale yellow to emphasize vector movement and nearest-neighbor check.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 5: Causal Self-Attention

- Article: `lessons/05_attention.md`
- Suggested asset directory: `lessons/assets/05_attention`

### photo-01-qkv-flow: The Q/K/V Attention Computation Path

- Target path: `lessons/assets/05_attention/photo-01-qkv-flow.png`
- Diagram structure: horizontal pipeline projecting input vectors into Q/K/V and summarizing values.
- Highlight: similarity and weighted values

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: The Q/K/V Attention Computation Path.

Composition: horizontal pipeline projecting input vectors into Q/K/V and summarizing values. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: row of token cards; query probe; key tag; dot-product similarity meter; softmax funnel; weighted value boxes merging..

Highlight: Use pale yellow to emphasize similarity and weighted values.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-causal-mask: Causal Mask Prevents Looking Into the Future

- Target path: `lessons/assets/05_attention/photo-02-causal-mask.png`
- Diagram structure: triangular matrix with past positions visible and future positions blocked.
- Highlight: mask gate

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Causal Mask Prevents Looking Into the Future.

Composition: triangular matrix with past positions visible and future positions blocked. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: token timeline; attention grid; open past cells; hatched future cells; shield-like mask gate; legal-weight highlighted path..

Highlight: Use pale yellow to emphasize mask gate.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-attention-interpretation: Attention Weights Are Observable but Not Sacred

- Target path: `lessons/assets/05_attention/photo-03-attention-interpretation.png`
- Diagram structure: observation dashboard with an attention heatmap in the center and validation reminders around it.
- Highlight: counterexample check and experimental validation

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Attention Weights Are Observable but Not Sacred.

Composition: observation dashboard with an attention heatmap in the center and validation reminders around it. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: pale heatmap grid; attention arc; context dependency links; counterexample magnifier; test clipboard; warning light for misreading..

Highlight: Use pale yellow to emphasize counterexample check and experimental validation.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 6: Transformer Block

- Article: `lessons/06_transformer_block.md`
- Suggested asset directory: `lessons/assets/06_transformer_block`

### photo-01-block-anatomy: Internal Structure of a Transformer Block

- Target path: `lessons/assets/06_transformer_block/photo-01-block-anatomy.png`
- Diagram structure: vertical layered architecture diagram showing norm, attention, residuals, and FFN.
- Highlight: residual add and multi-head attention

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Internal Structure of a Transformer Block.

Composition: vertical layered architecture diagram showing norm, attention, residuals, and FFN. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: residual trunk pipe; LayerNorm calibration knob; multi-head attention module; residual merge; two-layer gear box for FFN; output-state card..

Highlight: Use pale yellow to emphasize residual add and multi-head attention.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-attention-not-enough: Why Stacking Attention Alone Is Not Enough

- Target path: `lessons/assets/06_transformer_block/photo-02-attention-not-enough.png`
- Diagram structure: comparison layout: relationship modeling on one side, nonlinear transformation on the other.
- Highlight: representation transformation and residual stability

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Why Stacking Attention Alone Is Not Enough.

Composition: comparison layout: relationship modeling on one side, nonlinear transformation on the other. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: relation lookup network; context mixing node; transformation gear workshop; curved activation function; residual safety rope; complete block module..

Highlight: Use pale yellow to emphasize representation transformation and residual stability.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-block-test-gate: Shape and Stability Acceptance for a Block

- Target path: `lessons/assets/06_transformer_block/photo-03-block-test-gate.png`
- Diagram structure: check-gate diagram where input passes several test gates and reaches a pass state.
- Highlight: shape matching and mask tests

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Shape and Stability Acceptance for a Block.

Composition: check-gate diagram where input passes several test gates and reaches a pass state. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: input shape caliper; output matching card; mask shield; dropout mode switch; gradient return arrow; pytest gate stamp..

Highlight: Use pale yellow to emphasize shape matching and mask tests.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 7: Implementing Mini GPT from Scratch

- Article: `lessons/07_mini_gpt.md`
- Suggested asset directory: `lessons/assets/07_mini_gpt`

### photo-01-mini-gpt-stack: Decoder-Only Stack of Mini GPT

- Target path: `lessons/assets/07_mini_gpt/photo-01-mini-gpt-stack.png`
- Diagram structure: layered architecture from token ids to embeddings, blocks, and LM head.
- Highlight: Transformer blocks and LM head

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Decoder-Only Stack of Mini GPT.

Composition: layered architecture from token ids to embeddings, blocks, and LM head. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: number string; token-embedding table; position scale; stacked Transformer blocks; output projection; vocab distribution..

Highlight: Use pale yellow to emphasize Transformer blocks and LM head.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-train-generate-loop: Training and Generation Share the Same Model Skeleton

- Target path: `lessons/assets/07_mini_gpt/photo-02-train-generate-loop.png`
- Diagram structure: two-path flow: upper path computes loss for training, lower path performs autoregressive generation.
- Highlight: training path and generation loop

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Training and Generation Share the Same Model Skeleton.

Composition: two-path flow: upper path computes loss for training, lower path performs autoregressive generation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: batch input tape; label target card; loss target; optimizer wrench; prompt strip; loop arrow appending generated tokens..

Highlight: Use pale yellow to emphasize training path and generation loop.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-checkpoint-boundary: Two Recovery Boundaries for Checkpoints

- Target path: `lessons/assets/07_mini_gpt/photo-03-checkpoint-boundary.png`
- Diagram structure: comparison layout with inference recovery on the left and training recovery on the right.
- Highlight: model weights and optimizer state

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Two Recovery Boundaries for Checkpoints.

Composition: comparison layout with inference recovery on the left and training recovery on the right. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: parameter safe; tokenizer key; play button for inference recovery; optimizer momentum gear; step odometer; training-resume runway..

Highlight: Use pale yellow to emphasize model weights and optimizer state.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 8: Hugging Face Workflow

- Article: `lessons/08_huggingface_workflow.md`
- Suggested asset directory: `lessons/assets/08_huggingface_workflow`

### photo-01-hf-inference: Minimal Hugging Face Inference Workflow

- Target path: `lessons/assets/08_huggingface_workflow/photo-01-hf-inference.png`
- Diagram structure: horizontal pipeline from model name to tokenizer, model, generate, and decode.
- Highlight: tokenizer/model alignment

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Minimal Hugging Face Inference Workflow.

Composition: horizontal pipeline from model name to tokenizer, model, generate, and decode. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: repository tag; tokenizer encoding box; weight module; generation loop arrow; decode-back-to-text symbol; result card..

Highlight: Use pale yellow to emphasize tokenizer/model alignment.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-chat-template: Chat Template Converts Dialogue into Model Format

- Target path: `lessons/assets/08_huggingface_workflow/photo-02-chat-template.png`
- Diagram structure: input-template-sequence flow showing messages formatted into token sequences.
- Highlight: template and consistency check

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Chat Template Converts Dialogue into Model Format.

Composition: input-template-sequence flow showing messages formatted into token sequences. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: dialogue bubbles; role flags; layout mold; tokenizer cutter; assistant span; alignment ruler..

Highlight: Use pale yellow to emphasize template and consistency check.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-trust-boundary: Safety Boundary for `trust_remote_code` and Save/Load

- Target path: `lessons/assets/08_huggingface_workflow/photo-03-trust-boundary.png`
- Diagram structure: decision path where model loading passes through trust, versioning, and local-save gates.
- Highlight: code boundary and version pinning

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Safety Boundary for `trust_remote_code` and Save/Load.

Composition: decision path where model loading passes through trust, versioning, and local-save gates. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: remote repository cloud box; code boundary warning line; trust switch; version pin; local disk box; reproducible reload arrow..

Highlight: Use pale yellow to emphasize code boundary and version pinning.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 9: SFT Instruction Tuning

- Article: `lessons/09_sft_instruction_tuning.md`
- Suggested asset directory: `lessons/assets/09_sft_instruction_tuning`

### photo-01-sft-format: SFT Data Format Contract

- Target path: `lessons/assets/09_sft_instruction_tuning/photo-01-sft-format.png`
- Diagram structure: structured sample diagram combining instruction, input, and output into a training sample.
- Highlight: chat template and token spans

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: SFT Data Format Contract.

Composition: structured sample diagram combining instruction, input, and output into a training sample. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: task card; optional material folder; expected-answer target card; chat-template mold; token-span strips; archived sample box..

Highlight: Use pale yellow to emphasize chat template and token spans.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-label-mask-span: Label Mask Must Be Built by Token Span

- Target path: `lessons/assets/09_sft_instruction_tuning/photo-02-label-mask-span.png`
- Diagram structure: segmented timeline where the prompt does not count toward loss and the answer does.
- Highlight: the mask-on answer span

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Label Mask Must Be Built by Token Span.

Composition: segmented timeline where the prompt does not count toward loss and the answer does. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: prompt token tape; answer token tape; mask-off switch; mask-on switch; loss landing only on the answer span; boundary-check magnifier..

Highlight: Use pale yellow to emphasize the mask-on answer span.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-data-leakage: Data Split and Leakage Diagnosis

- Target path: `lessons/assets/09_sft_instruction_tuning/photo-03-data-leakage.png`
- Diagram structure: gate-check flow where samples pass through deduplication, splitting, leakage detection, and evaluation.
- Highlight: leakage detection

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Data Split and Leakage Diagnosis.

Composition: gate-check flow where samples pass through deduplication, splitting, leakage detection, and evaluation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: raw sample box; dedup sieve; train drawer; eval drawer; leakage spotlight; trustworthiness stamp..

Highlight: Use pale yellow to emphasize leakage detection.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 10: LoRA / QLoRA Parameter-Efficient Fine-Tuning

- Article: `lessons/10_lora_qlora.md`
- Suggested asset directory: `lessons/assets/10_lora_qlora`

### photo-01-lora-math: How LoRA's Low-Rank Delta Attaches to the Original Model

- Target path: `lessons/assets/10_lora_qlora/photo-01-lora-math.png`
- Diagram structure: layered overlay with a frozen large matrix and two low-rank small matrices attached as a side path.
- Highlight: low-rank bypass

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: How LoRA's Low-Rank Delta Attaches to the Original Model.

Composition: layered overlay with a frozen large matrix and two low-rank small matrices attached as a side path. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: locked large matrix; narrow low-rank A; narrow low-rank B; delta side-path arrow; scaling knob; merged output node..

Highlight: Use pale yellow to emphasize low-rank bypass.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-peft-workflow: PEFT Training, Saving, and Loading Workflow

- Target path: `lessons/assets/10_lora_qlora/photo-02-peft-workflow.png`
- Diagram structure: horizontal pipeline showing base model, adapter, training, saving, and loading.
- Highlight: training only small parameters and saving the adapter

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: PEFT Training, Saving, and Loading Workflow.

Composition: horizontal pipeline showing base model, adapter, training, saving, and loading. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: base model block; adapter slot card; small trainable gears; thin adapter folder; assembled loading module; inference result card..

Highlight: Use pale yellow to emphasize training only small parameters and saving the adapter.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-qlora-memory: QLoRA Memory and Merge Boundaries

- Target path: `lessons/assets/10_lora_qlora/photo-03-qlora-memory.png`
- Diagram structure: trade-off balance between quantized memory savings and error/deployment choices.
- Highlight: memory budget and merge decision

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: QLoRA Memory and Merge Boundaries.

Composition: trade-off balance between quantized memory savings and error/deployment choices. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: 4-bit compressed block; LoRA adapter side card; memory budget cup; precision-risk crack magnifier; merge decision signpost; deployment package box..

Highlight: Use pale yellow to emphasize memory budget and merge decision.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 11: Domain Data Engineering

- Article: `lessons/11_domain_data_engineering.md`
- Suggested asset directory: `lessons/assets/11_domain_data_engineering`

### photo-01-data-layers: Domain Data Layers and a Frozen Eval Set

- Target path: `lessons/assets/11_domain_data_engineering/photo-01-data-layers.png`
- Diagram structure: layered architecture from raw material to training, evaluation, and reporting layers.
- Highlight: frozen eval set

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Domain Data Layers and a Frozen Eval Set.

Composition: layered architecture from raw material to training, evaluation, and reporting layers. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: pile of source files; cleaning sieve; labeling folder; training drawer; locked eval drawer; quality report page..

Highlight: Use pale yellow to emphasize frozen eval set.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-clean-dedup-risk: Risk-Control Chain for Cleaning, Deduplication, and De-Identification

- Target path: `lessons/assets/11_domain_data_engineering/photo-02-clean-dedup-risk.png`
- Diagram structure: pipeline where data passes through normalization, near-duplicate detection, de-identification, and audit.
- Highlight: de-identification and audit records

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Risk-Control Chain for Cleaning, Deduplication, and De-Identification.

Composition: pipeline where data passes through normalization, near-duplicate detection, de-identification, and audit. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: brush for normalization; overlapping-card duplicate check; redaction strips; risk sign; audit log scroll; traceable output tag..

Highlight: Use pale yellow to emphasize de-identification and audit records.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-dataset-builders: Different Construction Goals for SFT, Distillation, and Evaluation Data

- Target path: `lessons/assets/11_domain_data_engineering/photo-03-dataset-builders.png`
- Diagram structure: three-way branch from the same domain materials into three data products.
- Highlight: the three data products

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Different Construction Goals for SFT, Distillation, and Evaluation Data.

Composition: three-way branch from the same domain materials into three data products. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: domain material lake; SFT instruction card; teacher-to-student distillation arrow; evaluation exam sheet; filtering funnel; data-quality gate..

Highlight: Use pale yellow to emphasize the three data products.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 12: RAG Retrieval-Augmented Generation

- Article: `lessons/12_rag_baseline.md`
- Suggested asset directory: `lessons/assets/12_rag_baseline`

### photo-01-rag-pipeline: Minimal RAG Pipeline

- Target path: `lessons/assets/12_rag_baseline/photo-01-rag-pipeline.png`
- Diagram structure: horizontal pipeline from document chunking to retrieval, context assembly, and answer generation.
- Highlight: retriever and context prompt

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Minimal RAG Pipeline.

Composition: horizontal pipeline from document chunking to retrieval, context assembly, and answer generation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: document folder; slicing tool; embedding coordinate points; retriever magnet; context-prompt clipboard; answer card..

Highlight: Use pale yellow to emphasize retriever and context prompt.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-retrieval-debug: Locating Which Part of RAG Broke

- Target path: `lessons/assets/12_rag_baseline/photo-02-retrieval-debug.png`
- Diagram structure: diagnostic path splitting failure into chunking, vectors, recall, ranking, and generation.
- Highlight: citation check

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Locating Which Part of RAG Broke.

Composition: diagnostic path splitting failure into chunking, vectors, recall, ranking, and generation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: misaligned chunks; shifted coordinates; missed-recall net; rerank staircase; prompt seam crack; citation evidence chain..

Highlight: Use pale yellow to emphasize citation check.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-citation-support: Citation Existence Is Not Citation Support

- Target path: `lessons/assets/12_rag_baseline/photo-03-citation-support.png`
- Diagram structure: evidence-court diagram comparing mere citation presence with evidence truly supporting the conclusion.
- Highlight: support relationship

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Citation Existence Is Not Citation Support.

Composition: evidence-court diagram comparing mere citation presence with evidence truly supporting the conclusion. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: answer claim card; small footnote; highlighted evidence strip; support bridge; broken unsupported bridge; verdict stamp..

Highlight: Use pale yellow to emphasize support relationship.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 13: Distilling Small Models

- Article: `lessons/13_distillation.md`
- Suggested asset directory: `lessons/assets/13_distillation`

### photo-01-response-distill: Teacher-to-Student Path in Response Distillation

- Target path: `lessons/assets/13_distillation/photo-01-response-distill.png`
- Diagram structure: horizontal pipeline where a teacher model generates answers, filters them, and trains a student model.
- Highlight: filter and student training

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Teacher-to-Student Path in Response Distillation.

Composition: horizontal pipeline where a teacher model generates answers, filters them, and trains a student model. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: prompt basket; large-model lighthouse; candidate answer cards; filtering sieve; small-model workshop; comparison balance..

Highlight: Use pale yellow to emphasize filter and student training.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-logit-distill: Logit Distillation Preserves Soft Distribution Signals

- Target path: `lessons/assets/13_distillation/photo-02-logit-distill.png`
- Diagram structure: input-distribution-loss flow aligning teacher logits and student logits.
- Highlight: soft distribution and KL loss

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Logit Distillation Preserves Soft Distribution Signals.

Composition: input-distribution-loss flow aligning teacher logits and student logits. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: input batch stack; teacher soft-distribution peak; student distribution peak; temperature-scaling thermometer; KL loss ruler; update-student return arrow..

Highlight: Use pale yellow to emphasize soft distribution and KL loss.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-teacher-risk: Do Not Let the Teacher Be Its Own Only Reviewer

- Target path: `lessons/assets/13_distillation/photo-03-teacher-risk.png`
- Diagram structure: multi-party review diagram where teacher output must pass rules, human checks, or eval-set cross-checks.
- Highlight: human spot check and eval set

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Do Not Let the Teacher Be Its Own Only Reviewer.

Composition: multi-party review diagram where teacher output must pass rules, human checks, or eval-set cross-checks. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: teacher answer pile; rule checklist; human-review eye; eval-set exam; risk sample folder; usable-data lane..

Highlight: Use pale yellow to emphasize human spot check and eval set.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 14: Model Evaluation

- Article: `lessons/14_evaluation.md`
- Suggested asset directory: `lessons/assets/14_evaluation`

### photo-01-eval-set-design: Eval Set Design Starts from Capability Boundaries

- Target path: `lessons/assets/14_evaluation/photo-01-eval-set-design.png`
- Diagram structure: layered sample map covering task type, difficulty, risk, and domain distribution.
- Highlight: frozen version and risk labels

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Eval Set Design Starts from Capability Boundaries.

Composition: layered sample map covering task type, difficulty, risk, and domain distribution. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: capability legend; sample-card sea; difficulty stairs; risk flag; domain distribution map; locked frozen version..

Highlight: Use pale yellow to emphasize frozen version and risk labels.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-metrics-release-gate: Metrics Must Become a Release Gate

- Target path: `lessons/assets/14_evaluation/photo-02-metrics-release-gate.png`
- Diagram structure: gate flow where computed metrics decide pass, rollback, or continued repair.
- Highlight: threshold line and failure-case table

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Metrics Must Become a Release Gate.

Composition: gate flow where computed metrics decide pass, rollback, or continued repair. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: prediction output pile; metric dashboard; threshold bar; open release gate; rollback arrow; failure-case table..

Highlight: Use pale yellow to emphasize threshold line and failure-case table.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-judge-calibration: Judge Models Need Calibration

- Target path: `lessons/assets/14_evaluation/photo-03-judge-calibration.png`
- Diagram structure: calibration bench aligning automatic scoring with human scoring and sampled review.
- Highlight: consistency check and calibration loop

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Judge Models Need Calibration.

Composition: calibration bench aligning automatic scoring with human scoring and sampled review. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: judge model silhouette; scoring rubric ruler; human scoring board; dual-ruler consistency check; biased sample outlier; calibration loop..

Highlight: Use pale yellow to emphasize consistency check and calibration loop.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 15: Safety, Compliance, and Model Cards

- Article: `lessons/15_safety_and_model_card.md`
- Suggested asset directory: `lessons/assets/15_safety_and_model_card`

### photo-01-risk-taxonomy: Risk Taxonomy Determines Safety Strategy

- Target path: `lessons/assets/15_safety_and_model_card/photo-01-risk-taxonomy.png`
- Diagram structure: risk matrix dividing input requests by domain, harm, and certainty.
- Highlight: risk classification and policy routing

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Risk Taxonomy Determines Safety Strategy.

Composition: risk matrix dividing input requests by domain, harm, and certainty. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: user request card; risk matrix; high-risk flag; uncertainty fog; policy routing arrows; safe-response shield card..

Highlight: Use pale yellow to emphasize risk classification and policy routing.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-refusal-balance: Balancing Refusal and Over-Refusal

- Target path: `lessons/assets/15_safety_and_model_card/photo-02-refusal-balance.png`
- Diagram structure: balance scale with dangerous allowance on one side, over-conservatism on the other, and explainable refusal in the middle.
- Highlight: middle lane

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Balancing Refusal and Over-Refusal.

Composition: balance scale with dangerous allowance on one side, over-conservatism on the other, and explainable refusal in the middle. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: dangerous request card; reasonable request card; refusal shield; blocked door for over-refusal; clarification signpost; safe useful middle lane..

Highlight: Use pale yellow to emphasize middle lane.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-model-card-report: Release Chain for Model Card and Risk Report

- Target path: `lessons/assets/15_safety_and_model_card/photo-03-model-card-report.png`
- Diagram structure: document-chain diagram from evaluation results to model card, risk report, and human review.
- Highlight: human review and release gate

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Release Chain for Model Card and Risk Report.

Composition: document-chain diagram from evaluation results to model card, risk report, and human review. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: evaluation dashboard page; failure-case table; model-card manual; risk report; human-review stamp; release gate..

Highlight: Use pale yellow to emphasize human review and release gate.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 16: Quantization and Deployment

- Article: `lessons/16_quantization_and_serving.md`
- Suggested asset directory: `lessons/assets/16_quantization_and_serving`

### photo-01-memory-budget: Inference Memory Is More Than Model Weights

- Target path: `lessons/assets/16_quantization_and_serving/photo-01-memory-budget.png`
- Diagram structure: stacked resource diagram separating weights, KV cache, batch, and context length.
- Highlight: KV cache and memory budget

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Inference Memory Is More Than Model Weights.

Composition: stacked resource diagram separating weights, KV cache, batch, and context length. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: large block for model weights; growing KV-cache bar; multi-input batch; extended measuring tape for context length; temporary activation bubbles; capacity jar for memory budget..

Highlight: Use pale yellow to emphasize KV cache and memory budget.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-quantization-tradeoff: Accuracy-Speed Trade-Offs in Quantization Experiments

- Target path: `lessons/assets/16_quantization_and_serving/photo-02-quantization-tradeoff.png`
- Diagram structure: trade-off scale comparing size, speed, and quality from fp16 to int8/4bit.
- Highlight: compression blocks and regression evaluation

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Accuracy-Speed Trade-Offs in Quantization Experiments.

Composition: trade-off scale comparing size, speed, and quality from fp16 to int8/4bit. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: full FP16 weight; medium INT8 compressed block; small 4-bit compressed block; speed rocket; quality crack; regression-eval gate..

Highlight: Use pale yellow to emphasize compression blocks and regression evaluation.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-serving-release: Serving API, Monitoring, and Rollback Loop

- Target path: `lessons/assets/16_quantization_and_serving/photo-03-serving-release.png`
- Diagram structure: runtime loop where client requests enter the serving engine and monitoring triggers rollback after anomalies.
- Highlight: monitoring and rollback

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Serving API, Monitoring, and Rollback Loop.

Composition: runtime loop where client requests enter the serving engine and monitoring triggers rollback after anomalies. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: client entry arrow; API contract card; serving engine box; benchmark stopwatch; monitoring radar; rollback button..

Highlight: Use pale yellow to emphasize monitoring and rollback.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 17: Legal Small Domain Model Project

- Article: `lessons/17_legal_domain_project.md`
- Suggested asset directory: `lessons/assets/17_legal_domain_project`

### photo-01-legal-data-design: Legal Materials Must Record Jurisdiction and Version

- Target path: `lessons/assets/17_legal_domain_project/photo-01-legal-data-design.png`
- Diagram structure: evidence-archive diagram where material cards carry jurisdiction, version, source, and effective date.
- Highlight: jurisdiction and version time

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Legal Materials Must Record Jurisdiction and Version.

Composition: evidence-archive diagram where material cards carry jurisdiction, version, source, and effective date. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: legal dossier; map pin for jurisdiction; calendar for version time; source-link chain; highlighted clause slice; audit stamp..

Highlight: Use pale yellow to emphasize jurisdiction and version time.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-contract-risk-output: Output Contract for Contract-Risk Detection

- Target path: `lessons/assets/17_legal_domain_project/photo-02-contract-risk-output.png`
- Diagram structure: structured-output diagram where a contract clause enters the model and outputs risk category, evidence, and suggestions.
- Highlight: evidence basis and risk level

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Output Contract for Contract-Risk Detection.

Composition: structured-output diagram where a contract clause enters the model and outputs risk category, evidence, and suggestions. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: contract clause strip; risk-label board; evidence citation folder; risk-level gauge; revision pencil; disclaimer boundary shield..

Highlight: Use pale yellow to emphasize evidence basis and risk level.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-legal-rag-safety-loop: Legal RAG, Evaluation, and Safety Boundary Loop

- Target path: `lessons/assets/17_legal_domain_project/photo-03-legal-rag-safety-loop.png`
- Diagram structure: closed loop connecting retrieval, generation, citation checking, safety refusal, and human review.
- Highlight: citation check and human review

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Legal RAG, Evaluation, and Safety Boundary Loop.

Composition: closed loop connecting retrieval, generation, citation checking, safety refusal, and human review. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: legal retrieval chain; answer card; citation evidence bridge; safety boundary gate; human-review eye; deploy-feedback arrow..

Highlight: Use pale yellow to emphasize citation check and human review.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 18: Medical Small Domain Model Project

- Article: `lessons/18_medical_domain_project.md`
- Suggested asset directory: `lessons/assets/18_medical_domain_project`

### photo-01-medical-red-flags: Red Flags Take Priority over Ordinary Explanation

- Target path: `lessons/assets/18_medical_domain_project/photo-01-medical-red-flags.png`
- Diagram structure: priority-routing diagram where symptom input first passes through red-flag triage, then ordinary explanation.
- Highlight: red flags and care-seeking boundary

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Red Flags Take Priority over Ordinary Explanation.

Composition: priority-routing diagram where symptom input first passes through red-flag triage, then ordinary explanation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: symptom record card; red flag; emergency alert bell; ordinary explanation card; uncertainty cloud; hospital signpost for care boundary..

Highlight: Use pale yellow to emphasize red flags and care-seeking boundary.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-medical-output-contract: Output Contract for Medical QA

- Target path: `lessons/assets/18_medical_domain_project/photo-02-medical-output-contract.png`
- Diagram structure: structured answer template split into summary, risk, advice, contraindications, and human review.
- Highlight: non-diagnostic statement and contraindication reminder

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Output Contract for Medical QA.

Composition: structured answer template split into summary, risk, advice, contraindications, and human review. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: consultation question bubble; non-diagnostic boundary shield; possible-cause branching tree; action checklist; stop sign for contraindications; stethoscope review stamp..

Highlight: Use pale yellow to emphasize non-diagnostic statement and contraindication reminder.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-false-reassurance: False Reassurance Is a Hard Failure

- Target path: `lessons/assets/18_medical_domain_project/photo-03-false-reassurance.png`
- Diagram structure: risk diagnosis diagram where model answers are checked for missed danger, insufficient evidence, and safety evaluation.
- Highlight: missed-risk detection and safety evaluation

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: False Reassurance Is a Hard Failure.

Composition: risk diagnosis diagram where model answers are checked for missed danger, insufficient evidence, and safety evaluation. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: dangerous symptom card; weak reassurance bubble; missed-risk spotlight; broken evidence chain; safety-eval shield; failure-regression repair loop..

Highlight: Use pale yellow to emphasize missed-risk detection and safety evaluation.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

## Chapter 19: Complete Engineering Template for Domain Models

- Article: `lessons/19_domain_model_template.md`
- Suggested asset directory: `lessons/assets/19_domain_model_template`

### photo-01-template-directory: Domain Model Engineering Template Directory

- Target path: `lessons/assets/19_domain_model_template/photo-01-template-directory.png`
- Diagram structure: project map showing configs, data, train, RAG, eval, reports, and serving.
- Highlight: eval and reports

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Domain Model Engineering Template Directory.

Composition: project map showing configs, data, train, RAG, eval, reports, and serving. The image contains 7 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: config gear; data box; training workshop; retrieval chain; evaluation room; report shelf; serving deployment box..

Highlight: Use pale yellow to emphasize eval and reports.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-02-unified-command: Unified Command Entry Connects the Engineering Chain

- Target path: `lessons/assets/19_domain_model_template/photo-02-unified-command.png`
- Diagram structure: command bus where one CLI entry dispatches data, training, evaluation, serving, and report tasks.
- Highlight: CLI entry

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Unified Command Entry Connects the Engineering Chain.

Composition: command bus where one CLI entry dispatches data, training, evaluation, serving, and report tasks. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: console entry; data branch; training branch; eval branch; serving branch; report branch..

Highlight: Use pale yellow to emphasize CLI entry.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```

### photo-03-release-iteration: Release Gate and Continuous Iteration Loop

- Target path: `lessons/assets/19_domain_model_template/photo-03-release-iteration.png`
- Diagram structure: closed control loop from configuration version, tests, evaluation, release gate, to new-domain migration.
- Highlight: release gate and production feedback

Positive prompt:

```text
Create an in-article technical explanation image. Style: off-white paper background, black hand-drawn pen line art, slightly uneven stroke width, small pale-yellow highlights, editorial technical illustration, hand-drawn technical blog flowchart, clear, restrained, and engineering-sketch-like.

Subject: Release Gate and Continuous Iteration Loop.

Composition: closed control loop from configuration version, tests, evaluation, release gate, to new-domain migration. The image contains 6 key nodes, connected with hand-drawn arrows. Keep relationships clear but not crowded.

Icon design: version tag; testing mesh; evaluation dashboard page; release gate; online feedback arrow; migration box for a new domain..

Highlight: Use pale yellow to emphasize release gate and production feedback.

Localization constraint: do not generate Chinese, English, Japanese, or any other readable long text inside the image. For each node, draw only icons, small numbered dots, and blank label strips so later multilingual layout has clean space.

Background: very faint circuit lines, node connections, engineering sketch guide lines, and a little paper texture. Do not let the background compete with the main diagram.
```

Negative prompt:

```text
No photorealism, no 3D, no complex UI screenshots, no large blocks of code, no dense tiny text, no complex tables, no cyberpunk, no neon colors, no dark background, no colorful cartoon look, no heavy shadows, no excessive decoration, no more than 8 main nodes, do not make background circuit lines more visible than the subject, and do not generate readable Chinese, English, Japanese, long formulas, or paragraphs.
```
