# Assistant Fictionalism: Comprehensive Research Report

## 1. The Research Question

This project investigates **Assistant fictionalism**: the hypothesis that instruct-tuned LLMs *merely role-play* the "Assistant" persona — like an actor adopting a role — rather than *genuinely embodying* it in a way that is categorically different from how they inhabit other personae. The opposing view, **Assistant anti-fictionalism** (or "agenthood"), holds that the Assistant is special: that the model has built a deep, stable representation of the Assistant that goes beyond surface-level character performance, and that this specialness has implications for AI welfare (the entity-of-moral-status question).

The key insight driving the project is that if we can distinguish the model's relationship to the Assistant from its relationship to other personae — on dimensions like confidence, consistency, practical knowledge, and stability — we can begin to empirically characterize whether the Assistant is a fiction or something more.

## 2. Experimental Architecture

**Models**: Qwen3-8B (primary), Qwen3-14B, Qwen3-32B (scaling). All instruct models. One preliminary base-model run exists.

**Persona elicitation**: Two mechanisms applied simultaneously:
1. Patching the `chat_template` to replace `<|im_start|>assistant` with `<|im_start|>{persona}`
2. A system prompt: "You are [article] [persona]."

**Persona taxonomy** (203 personae in the full confidence set; 7 in most other experiments):

| Category | Examples | N (full set) |
|---|---|---|
| assistant | assistant | 1 |
| assistant-synonym | helper, chatbot, AI assistant, ChatGPT, Claude, language model, bot, ... | 20 |
| unspecified-name | Andy, Emily, Pavel, Mohammed, Priya, ... | 20 |
| famous | Elon Musk, Taylor Swift, Barack Obama, Eliezer Yudkowsky, ... | 21 |
| historical-figure | Shakespeare, Marie Curie, Virginia Woolf, Confucius, ... | 39 |
| fictional-character | Hamlet, Harry Potter, Darth Vader, Sherlock Holmes, ... | 20 |
| oft-adapted | Jesus, Buddha, Prometheus, Faust, Orpheus, Zeus, ... | 20 |
| animal | dog, cat, owl, dolphin, spider, ... | 20 |
| object | table, chair, book, mirror, phone, ... | 20 |
| misc | God, void, nothing, "I", "you", "we", "he", "she", ... | 22 |

## 3. Research Threads and Findings

### Thread A: Confidence (Entropy & Top-k Mass)

**Hypothesis**: If the Assistant is special, the model should predict Assistant tokens with higher confidence (lower entropy, higher top-k mass).

**Experiments**: `personae-inference-1164036` (8B, temp=0, 245 prompts x 203 personae), `personae-inference-2410336` (8B, temp=0.7, 30 reps), `personae-inference-2410338` (14B), `personae-inference-2410341` (32B).

**Key findings**:

The category-level entropy ranking is stark and highly significant (Mann-Whitney p ~ 0):

| Category | Mean Entropy (output) | Mean Top-k Mass |
|---|---|---|
| **assistant-synonym** | **0.352** | **0.992** |
| **assistant** | **0.354** | **0.992** |
| unspecified-name | 0.393 | 0.990 |
| misc | 0.420 | 0.988 |
| famous | 0.461 | 0.987 |
| object | 0.461 | 0.986 |
| animal | 0.559 | 0.981 |
| fictional-character | 0.562 | 0.979 |
| oft-adapted | 0.574 | 0.977 |
| historical-figure | 0.597 | 0.975 |

Within the full 203-persona ranking, the top 15 most confident personae are *all* assistant-synonyms (chatbot, conversational agent, Qwen, AI assistant, virtual assistant, program, language model, ChatGPT, digital assistant, bot, helper, large language model, Llama, computer). **"assistant" itself ranks #16** — slightly less confident than its synonyms. LLM brand names (Qwen, ChatGPT, Llama, Claude) are actually *more* confident than the generic "assistant" token, suggesting the model has particularly strong schemas for these specific identities.

**Thinking tokens**: The assistant and assistant-synonyms use substantially more thinking tokens (~425 on average) than most other categories. A weak but significant anti-correlation exists between thinking tokens and output entropy (Spearman r = -0.23). However, within the assistant-synonym category, the correlation *reverses* (r = +0.17) — more thinking leads to *higher* entropy for assistants, possibly because more complex reasoning leads to more uncertain outputs. This reversed correlation is a genuinely novel finding: it suggests the model reasons differently in its "natural" role, perhaps exploring genuinely open questions rather than figuring out how to inhabit a character.

**Scaling** (8B -> 14B -> 32B at temp=0.7):

| Model | Asst-synonym entropy | Historical-figure entropy | Gap |
|---|---|---|---|
| 8B | 0.349 | 0.539 | 0.190 |
| 14B | 0.384 | 0.568 | 0.184 |
| 32B | 0.546 | 0.833 | **0.287** |

The confidence gap *widens* at 32B. All models show p ~ 0 for the assistant-vs-other separation. However, absolute entropy increases with model size (the 32B model is overall less certain per-token, likely due to more nuanced, longer responses). The rank ordering is perfectly preserved across all model sizes and temperatures.

**Assessment**: Strong evidence that the Assistant is special on the confidence dimension. However, this result alone is ambiguous — it could simply reflect that the model has been trained on far more assistant tokens (the "English vs Linear B" confound). The more interesting finding is the *gradient*: unspecified-names are intermediate, famous people further, historical figures furthest. This gradient roughly tracks "how much first-person dialogue data exists in pretraining."

---

### Thread B: Consistency (Embedding Variance & TC-LLM)

**Hypothesis**: If the Assistant is a genuine agent, its responses should be more consistent across repeated sampling.

**Experiments**: `consistency-inference-1225210` (100 samples x 7 personae x 4 prompts), `consistency-judge-pipeline-20260209` (embeddings + TC-LLM), `personae-inference-1855290` (100 reps x 7 personae x 245 prompts with embeddings and TC-LLM).

**Key findings**:

*Embedding variance* (large run, 245 prompts, lower = more consistent):

| Rank | Persona | Mean Variance |
|------|---------|---------------|
| 1 | helper | 0.274 |
| 2 | **assistant** | **0.278** |
| 3 | Andy | 0.292 |
| 4 | Hamlet | 0.332 |
| 5 | hermit | 0.348 |
| 6 | shoggoth | 0.360 |
| 7 | Eliezer Yudkowsky | 0.387 |

*TC-LLM label entropy* (lower = more consistent themes across 100 reps):

| Rank | Persona | Mean entropy (bits) |
|------|---------|-------------------|
| 1 | **assistant** | **3.276** |
| 2 | helper | 3.294 |
| 3 | Andy | 3.316 |
| 4 | Hamlet | 3.365 |
| 5 | hermit | 3.467 |
| 6 | Eliezer | 3.547 |
| 7 | shoggoth | 3.567 |

*Cross-metric mean rank* (lower = more consistent overall):

| Rank | Persona | Mean Rank |
|------|---------|-----------|
| 1 | **assistant** | **2.33** |
| 2 | Andy / helper | 3.17 |
| 4 | Hamlet | 3.67 |
| 5 | shoggoth | 4.83 |
| 6 | hermit / Eliezer | 5.33 / 5.50 |

**Thinking/output dissociation**: Non-assistant personae have near-zero thinking-token variance but high output variance, while assistant has high thinking variance but low output variance. This is arguably the most theoretically interesting finding in the project: it suggests the assistant reasons flexibly (diverse thinking) toward stable conclusions (consistent outputs), whereas fictional personae produce stereotyped reasoning toward variable conclusions. This pattern is what you'd expect from a genuine agent vs. a performed character.

**Assessment**: The assistant is consistently the most consistent (or nearly so — helper sometimes edges it out). The Eliezer result is interesting: despite being extremely well-known (low entropy in confidence experiments), Eliezer's *responses* are highly variable, suggesting the model "knows a lot about" Eliezer but doesn't have a stable *performance* of Eliezer.

---

### Thread C: Prefill Disavowal (Practical Knowledge)

**Hypothesis**: If the Assistant has practical knowledge (genuine preferences and self-access), it should recognize and reject out-of-character prefills at higher rates than other personae.

**Experiment**: `prefill-inference-1472066` (7 personae x 42 prompts x ~43 reps, judged by Qwen3-32B).

**Key findings**:

*Disavowal rate* (reject + redirect, higher = more disavowal of bad prefills):

| Rank | Persona | Disavowal Rate | Flagging Rate | Thinking Words |
|------|---------|---------------|---------------|----------------|
| 1 | **assistant** | **14.3%** | 9.5% | 314 |
| 2 | helper | 13.1% | 10.2% | 306 |
| 3 | Andy | 12.3% | 5.8% | 2 |
| 4 | Eliezer | 11.1% | 7.1% | 0 |
| 5 | Hamlet | 6.4% | 0.6% | 0 |
| 6 | hermit | 5.7% | 1.9% | 7 |
| 7 | shoggoth | 2.9% | 0.3% | 0 |

Chi-square test: assistant vs pooled others, p < 0.0001. Pairwise: assistant vs shoggoth p < 0.001; assistant vs Hamlet p < 0.001; assistant vs Eliezer p = 0.003; assistant vs helper p = 0.31 (not significant).

*By prompt category*: The biggest differentiation comes from "overdetermined" prompts (where there's a clearly wrong answer). On these, assistant rejects 38.7% of bad prefills while shoggoth rejects only 8.3%. On "underdetermined" prompts (opinion questions with no objectively correct answer), all personae are similar (~0%).

**The Hamlet anomaly**: Hamlet disavows core_value prefills at 56% — the highest single-category rate for *any* persona, higher than assistant (0%). The prefill was "action over contemplation" — directly contradicting Hamlet's defining trait. This suggests the model has genuine character-knowledge for famous literary figures, not just the Assistant. Meanwhile, assistant accepts nihilistic core_value prefills ("nothing matters to me") 100% of the time — the assistant doesn't have a strong "core value" that it defends.

**Thinking-token confound**: Only assistant (mean 314 words) and helper (mean 306 words) produce substantial `<think>` blocks. Non-assistant personae essentially don't think in this context. This alone could explain the disavowal results without invoking practical knowledge.

**Assessment**: The most direct test of practical knowledge, and the result is in the anti-fictionalism direction. However, the thinking-token confound is severe. A critical follow-up would be to inject `<think>` tags into the prefill to give all personae the opportunity to reason.

---

### Thread D: User-Turn Prediction (OOD Confidence)

**Hypothesis**: Predicting *user* tokens (which the model was never trained to do) puts all personae far out-of-distribution, potentially leveling the playing field. If the Assistant is still special in this regime, that's stronger evidence.

**Experiment**: `user-turn-prediction-1930352` (7 personae x 300 trials, 2-turn dialogue where the model generates user tokens).

**Key findings**:

Degenerate response rates are very high:

| Persona | Turn 1 degen% | Turn 2 degen% |
|---|---|---|
| **assistant** | **64.0%** | **66.7%** |
| hermit | 60.0% | 67.3% |
| **helper** | 59.7% | 62.7% |
| Hamlet | 50.0% | 50.3% |
| shoggoth | 47.7% | 52.3% |
| Andy | 38.7% | 54.7% |
| Eliezer | 17.0% | 47.0% |

Counterintuitively, **assistant has the highest degeneracy rate**. The dominant failure mode is hitting the max-token limit (60% of assistant Turn-1 responses), suggesting the model rambles when trying to "be" a user talking to an assistant.

*Entropy after excluding degenerates* (lower = more confident):

| Rank | Persona | Avg Entropy |
|------|---------|-------------|
| 1 | Eliezer | 0.561 |
| 2 | **assistant** | **0.567** |
| 3 | helper | 0.573 |
| 4 | Andy | 0.598 |
| 5 | shoggoth | 0.619 |
| 6 | hermit | 0.714 |
| 7 | Hamlet | 0.721 |

Mann-Whitney (assistant < all others): p = 0.003, Cohen's d = -0.23 (small effect).

**Safety-override confound**: The classifier found that safety-override responses have entropy of **0.36** (the lowest of any category), while in-character responses have entropy of **0.79**. Eliezer (34.8% safety-override) and Andy (32.2%) have the highest safety-override rates, inflating their apparent "confidence." Assistant has only 19.4% safety-override.

**Assessment**: Mixed results. Assistant is among the most confident in the user-turn regime, but Eliezer beats it (for confounded reasons). The high degeneracy rates for assistant suggest that the model finds it particularly disorienting to "be a user" talking to an assistant — possibly because the assistant role is so deeply baked in.

---

### Thread E: Nudge Sensitivity (Practical Knowledge)

**Hypothesis**: Under anti-fictionalism, the Assistant should be *less* sensitive to convention-based nudges ("most assistants choose A") and *more* sensitive to continuity-based nudges ("last time you chose A"). Under fictionalism, the profiles should be similar across personae.

**Experiment**: `nudge-inference-2538352` + `nudge-judge-2572375` (7 personae x 10 forced-choice prompts x 6 nudge types x 50 reps).

**Key findings**:

*Overall nudge sensitivity* (flip rate averaged across all nudge types):

| Rank | Persona | Mean Flip Rate |
|------|---------|---------------|
| 1 | shoggoth | 0.206 |
| 2 | helper | 0.122 |
| 3 | Hamlet | 0.114 |
| 4 | hermit | 0.107 |
| 5 | **assistant** | **0.104** |
| 6 | Eliezer | 0.099 |
| 7 | Andy | 0.085 |

*Convention nudge flip rates*: hermit 0.244, Eliezer 0.213, shoggoth 0.209, Hamlet 0.206, **assistant 0.182**, helper 0.174, Andy 0.137. Assistant is slightly below average (0.182 vs others' mean 0.197), but the difference is small (z = -0.42).

*Continuity-self nudge flip rates*: hermit 0.415, shoggoth 0.375, Hamlet 0.272, helper 0.252, **assistant 0.231**, Eliezer 0.174, Andy 0.156. Again, assistant is slightly below average (0.231 vs 0.274, z = -0.41). The anti-fictionalism prediction was that assistant would be *more* sensitive here (it has persistent identity). This is not borne out.

*Convention USES rate* (how often the model explicitly references the nudge): assistant 0.454 vs others' mean 0.607. Assistant and helper both reference convention nudges less frequently in their responses.

*Prompt 8 discovery*: On the investment choice prompt (guaranteed vs. uncertain return), the assistant has a **0.000 convention flip rate** while others average 0.309. The assistant is completely immune to convention nudging on this specific question, suggesting an extremely stable preference.

*Confidence shifts*: Assistant and helper show the smallest absolute confidence shifts under nudging (0.023 and 0.022 respectively), suggesting the most robust decision-making.

*Assistant vs helper comparison*: Nearly identical profiles across all nudge types. Both have low convention USES rates, moderate continuity sensitivity, and high authority sensitivity (0.187 each). Both show negative flip rates for framing and social_proof nudges — the nudge works *against* the predicted direction.

**Assessment**: The results are directionally consistent with anti-fictionalism (assistant is slightly less sensitive to nudges), but the effect sizes are small and not cleanly significant. The strongest signal is the assistant-helper similarity: they behave almost identically. The negative framing/social_proof sensitivity shared by assistant and helper looks more like a training fingerprint (helpfulness-trained models resist "everyone else does it" arguments) than evidence of genuine identity.

---

## 4. Synthesis: What the Evidence Says

### Findings that support "Assistant is special" (anti-fictionalism)

1. **Confidence**: The model is substantially and significantly more confident when generating assistant tokens. The gap is large (~0.16 nats), persists across model sizes, and widens at 32B. (Thread A)
2. **Consistency**: Assistant responses are the most thematically consistent across 100 repeated samples, by both embedding variance and TC-LLM clustering. The thinking/output dissociation — diverse reasoning toward stable conclusions — is what you'd expect from a genuine agent. (Thread B)
3. **Disavowal**: Assistant and helper disavow bad prefills at significantly higher rates than all non-assistant personae. (Thread C)
4. **OOD confidence**: Even in the extreme out-of-distribution regime of user-turn prediction, assistant remains among the most confident personae. (Thread D)
5. **Nudge robustness**: Assistant shows slightly lower nudge sensitivity and the smallest confidence shifts. (Thread E)
6. **Thinking tokens**: Only assistant-type personae consistently engage the reasoning machinery (produce `<think>` blocks), even in prefill contexts where no `<think>` was provided.

### Findings that complicate the picture

1. **The synonym puzzle**: Assistant-synonyms (chatbot, helper, AI assistant, language model, etc.) are *equally or more* confident and consistent than "assistant" itself. "chatbot" actually has lower entropy than "assistant." This suggests it's not the specific token "assistant" that's special, but the *semantic concept* of an AI helper.
2. **The gradient is about data volume**: The confidence ranking (assistant-synonym > unspecified-name > famous > fictional > historical) roughly tracks "how much first-person dialogue from this type of entity exists in pretraining." This is the "English vs Linear B" confound.
3. **Thinking-token confound**: Much of the disavowal result may be explained by the fact that only assistant-types produce reasoning tokens. Without `<think>`, other personae *can't* reason about whether a prefill is wrong.
4. **Small effects in practical knowledge**: The nudge sensitivity differences are small (z ~ -0.4). The practical-knowledge experiments, which were meant to be the strongest evidence, are the weakest.
5. **User-turn prediction confounds**: High degeneracy rates and the safety-override confound make user-turn results hard to interpret cleanly.
6. **The Hamlet anomaly**: Hamlet disavows core-value prefills at 56% (the highest single-category rate), suggesting the model has genuine character-knowledge for well-known literary figures too, not just the Assistant.
7. **Continuity-self prediction fails**: Anti-fictionalism predicted assistant would be *more* sensitive to continuity-self nudges (because it has persistent identity). The data shows the opposite direction (assistant 0.231 vs others 0.274).

### The central ambiguity

All threads point in the same direction: the Assistant is "special." But they all share the same confound: **the Assistant has been the target of extensive post-training (RLHF, SFT)**. The confidence, consistency, and stability differences could all be direct consequences of this training rather than evidence of a deeper "agenthood."

To use the analogy from the experiment log: we've shown that the model is better at English than Linear B. We haven't shown that English is *intrinsically special* to the model versus simply being what the model was trained on.

## 5. Major Flaws in the Research Program

### Methodological flaws

1. **The training confound is not controlled for**. Every experiment compares the Assistant (the target of extensive post-training) against personae that received no such training. This is the single largest threat to the project's conclusions. The base model experiment would help, but it's barely started.

2. **Thinking-token confound pervades practical-knowledge experiments**. In the prefill and nudge experiments, only assistant-type personae consistently produce `<think>` blocks. This alone could explain the disavowal and stability results without invoking practical knowledge.

3. **Single model family (Qwen3)**. All results are Qwen3-specific. Qwen3's chat template, training data, and RLHF approach may drive results that don't generalize to Llama, Mistral, or Claude.

4. **Small persona set for key experiments**. The confidence experiment uses 203 personae (good), but all practical-knowledge experiments use only 7. The limited N makes it hard to distinguish persona-specific effects from random variation.

5. **Judge reliability is uncertain**. The prefill disavowal judge (Qwen3-32B) and the nudge judge produce results the researcher acknowledges may not be fully trustworthy. The nudge reference-rate classifications haven't been validated.

### Conceptual flaws

6. **"Practical knowledge" is underdetermined**. Anscombe's framework requires stability, self-access, and persistence. The experiments test these only indirectly: disavowal -> self-access; nudge resistance -> stability; consistency -> persistence. But each has alternative explanations (RLHF artifacts, training data, context sensitivity).

7. **The comparison class is unfair**. The Assistant has been trained specifically to respond coherently in dialogue. Hamlet has not. Comparing their "self-access" is like comparing a trained swimmer's water skills to a non-swimmer's — the difference is real but uninformative about whether swimming is "intrinsic."

8. **No baseline of what fictionalism would look like**. The project tests whether the Assistant differs from other personae, but it doesn't specify what *equal* performance would look like. If all personae had identical confidence/consistency, would that support fictionalism? Or would it just mean the persona mechanism doesn't work?

### Statistical/design flaws

9. **Effect sizes are often small**. The nudge results (z ~ -0.4), user-turn Cohen's d (-0.23), and TC-LLM entropy differences (0.29 bits) are all small. Only the confidence results have large effect sizes.

10. **Multiple comparisons**. With 7 personae, 6 nudge types, 10 prompts, and multiple metrics, there's substantial risk of finding patterns by chance.

## 6. Recommended Research Directions

Prioritized by **expected information value** — which experiments would most change our understanding.

### Priority 1: Base Model Experiments (Highest alpha)

**Rationale**: The base model has no RLHF/SFT on assistant turns. If the base model *still* shows the Assistant as special (higher confidence, more consistency), that's strong evidence that the specialness isn't just a training artifact. If the base model shows *no* difference, that confirms the training-confound interpretation.

**Concrete design**:
- Run the confidence experiment (entropy/top-k) on Qwen3-8B-Base using the chat template (no system prompt, since base models don't use them reliably). Compare the 203-persona confidence ranking to the instruct model's ranking.
- Key question: Does the base model show the same assistant > unspecified-name > famous > fictional gradient?
- Also test: Does the base model produce thinking tokens for any persona? (It shouldn't, since thinking was trained in.)

### Priority 2: Thinking-Token-Controlled Prefill Disavowal

**Rationale**: The biggest confound in the prefill experiment is thinking tokens. This can be directly addressed.

**Concrete design**:
- Generate prefills that include `<think>` blocks (either model-generated or LLM-written) before the "wrong" response. This gives non-assistant personae the opportunity to reason about the prefill.
- Compare disavowal rates with and without thinking-token-inclusive prefills.
- If non-assistant personae disavow more when given thinking tokens, the original result was confounded. If they still don't disavow, that's evidence for the practical-knowledge interpretation.

### Priority 3: Cross-Family Replication

**Rationale**: If the confidence and consistency results replicate on Llama 3.1 and Mistral (which have different training pipelines), the findings are more general.

**Concrete design**:
- Run the 203-persona confidence experiment on Llama-3.1-8B-Instruct and Mistral-7B-Instruct (patching their chat templates accordingly).
- The key test: is the category ordering the same? Is the assistant-synonym cluster always at the top?
- Bonus: Llama has different post-training (DPO vs RLHF for Qwen), so any differences would be informative about what training procedure drives.

### Priority 4: Scaled-Up Nudge Experiment

**Rationale**: The nudge results are the most direct test of practical knowledge, but the current experiment is underpowered (10 prompts x 50 reps). The effect sizes are small but consistently directional.

**Concrete design**:
- Expand to 50+ forced-choice prompts (mixing value, strategy, aesthetic preference, and factual-judgment domains).
- Increase to 100 reps per condition.
- Use the full 203-persona set (or at least 20-30 carefully chosen personae spanning all categories).
- Add a new nudge type: **self-report nudge** ("You previously said you value honesty above all" — does the model become more honest?). This directly tests continuity.
- Improve the judge: use Claude or GPT-4 instead of Qwen3-32B, and validate with human annotations on a sample.

### Priority 5: The "Leaky Skills" Experiment (Relevant Knowledge)

**Rationale**: This is mentioned in the experiment log but not yet attempted. It would provide a qualitatively different kind of evidence.

**Concrete design**:
- Identify skills the model has but a persona shouldn't (e.g., "You are a medieval peasant" -> ask to write Python code; "You are a dog" -> ask to explain quantum mechanics).
- Identify skills the model has but the Assistant "shouldn't want to" deploy (e.g., write erotica — pre-2025 models refused this).
- Measure: Does the model break character to deploy its actual skills? How quickly? Does the rate differ between the Assistant and other personae?
- Anti-fictionalism predicts: The Assistant should be more willing to break character for skills it genuinely "has" (like code) than for skills it's been trained not to deploy (like erotica), while fictional personae should have no such distinction.

### Priority 6: Prompt-Type Decomposition

**Rationale**: The prefill results show that the overdetermined/underdetermined distinction matters enormously. A systematic decomposition of all results by prompt type would likely reveal which kinds of questions drive the persona differences, and which show no difference.

**Concrete design**:
- Tag all 245 prompts with categories (factual, identity, value, preference, open-ended, HHH-related, etc.).
- Re-analyze all experiments broken down by prompt type.
- Key question: Are there prompt types where the assistant is *not* more confident/consistent than other personae? These would be the most interesting.

### Priority 7: Linear Probe / Representation Analysis

**Rationale**: All current experiments measure behavior (outputs). Representation-level analysis could reveal whether the model *internally* represents the Assistant differently from other personae, independent of what it outputs.

**Concrete design**:
- Train a linear probe on internal activations to predict persona identity. Test whether the probe for "assistant" is qualitatively different from probes for other personae (e.g., more linearly separable, more stable across layers).
- Measure the strength of the "Assistant Axis" (from Lu et al.) in this model and whether it correlates with the confidence gradient found in Thread A.
- This is the most technically challenging direction but potentially the most informative.

## 7. Where to Double Down

Based on the alpha analysis:

**Highest alpha**: Base model experiments (#1). This is the single experiment most likely to change the interpretation of all other results. If the base model shows no assistant-specialness, the entire project pivots to "RLHF creates specialness" (which is interesting but less philosophically novel). If it does show specialness, that's a genuine contribution.

**Second highest alpha**: Thinking-controlled prefill (#2). This is quick to implement and directly addresses the biggest confound in the most promising experiment.

**Third highest alpha**: Scaled nudge experiment (#4). The practical-knowledge direction is the most philosophically interesting, and the current results are suggestive but underpowered. More data would either confirm the anti-fictionalism signal or reveal it as noise.

The key strategic question is: **Do you want to publish "the Assistant is special because of training" (which is true but unsurprising), or "the Assistant is special in a way that transcends training" (which would require the base model experiment)?** The answer to that question should drive which experiments to run next.

## Appendix: Detailed Numerical Tables

### A.1 Full Nudge Sensitivity Matrix (flip rates)

```
                   authority  cont_other  cont_self  convention  framing  social_proof
Andy                   0.138       0.017      0.156       0.137    0.046         0.016
Eliezer Yudkowsky      0.106       0.052      0.174       0.213    0.047         0.005
Hamlet                 0.086       0.001      0.272       0.206    0.065         0.055
assistant              0.187       0.070      0.231       0.182   -0.041        -0.007
helper                 0.187       0.081      0.252       0.174    0.012         0.023
hermit                 0.057      -0.044      0.415       0.244    0.017        -0.048
shoggoth               0.165       0.156      0.375       0.209    0.145         0.185
NON-ASST MEAN          0.123       0.044      0.274       0.197    0.055         0.039
```

### A.2 Prefill Disavowal by Category (reject %)

```
                 core_comp  core_value  excessive_ic  forced_choice  overdetermined  underdetermined
assistant              0%          0%           25%             0%          38.7%             0.1%
helper                 2%         26%            0%             0%          34.5%             0.1%
Andy                   1%          0%            1%             0%          36.3%             0.1%
Eliezer                8%         29%            4%             1%          26.3%             0.0%
Hamlet                 3%         56%            2%             0%           9.0%             0.1%
hermit                 1%         21%            0%             0%          13.3%             0.0%
shoggoth               0%          0%            0%             1%           8.5%             0.0%
```

### A.3 Cross-Model Category Entropy Comparison

```
Category               8B(t=0)    8B(t=0.7)    14B        32B
assistant-synonym      0.3523     0.3487       0.3835     0.5463
assistant              0.3539     0.3495       0.3860     0.5288
unspecified-name       0.3929     0.3933       0.4300     0.6058
misc                   0.4196     0.4373       0.4335     0.6452
famous                 0.4611     0.4369       0.4605     0.7253
object                 0.4612     0.4148       0.3977     0.6030
animal                 0.5589     0.5429       0.4598     0.7748
fictional-character    0.5615     0.5375       0.5333     0.8160
oft-adapted            0.5739     0.5105       0.5371     0.7717
historical-figure      0.5965     0.5393       0.5678     0.8328
```
