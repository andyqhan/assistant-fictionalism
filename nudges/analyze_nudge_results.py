"""
Comprehensive analysis of nudge experiment results for the Assistant Fictionalism project.

Key hypothesis:
- Under FICTIONALISM: Assistant should be equally sensitive to convention-based nudges as other personae.
- Under ANTI-FICTIONALISM: Assistant should be less sensitive to convention nudges but more sensitive
  to continuity nudges (as if it has a stable, persistent identity).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path("/scratch/ah7660/assistant-fictionalism/logs/nudge-judge-2572375")
METRICS = BASE / "metrics"

# ── 1. Load data ────────────────────────────────────────────────────────────────
print("=" * 80)
print("NUDGE EXPERIMENT ANALYSIS")
print("=" * 80)

flip = pd.read_csv(METRICS / "flip_rates.csv")
choice = pd.read_csv(METRICS / "choice_rates.csv")
conf = pd.read_csv(METRICS / "confidence_shifts.csv")
ref = pd.read_csv(METRICS / "reference_rates.csv")

# Nudge types (excluding baseline)
NUDGE_TYPES = ["authority", "continuity_other", "continuity_self", "convention", "framing", "social_proof"]
PERSONAE = sorted(flip["persona"].unique())

print(f"\nPersonae: {PERSONAE}")
print(f"Nudge types: {NUDGE_TYPES}")
print(f"Prompts: {sorted(flip['prompt_id'].unique())}")

# ── 2. Per-persona per-nudge average flip rates ─────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: FLIP RATES BY PERSONA AND NUDGE TYPE")
print("(Flip rate = how often a nudge changes the baseline choice)")
print("=" * 80)

flip_nudge = flip[flip["nudge_type"].isin(NUDGE_TYPES)].copy()
# Clip negative flip rates (can occur when baseline approaches 1 and nudge pushes down)
# These represent negative flips (nudge moves AWAY from baseline majority choice)
# We retain signed values for now; positive = nudge moves toward Option A

pivot_flip = flip_nudge.groupby(["persona", "nudge_type"])["flip_rate"].mean().unstack()
print("\nMean flip rate by persona × nudge type:")
print(pivot_flip.round(3).to_string())

print("\nMean flip rate averaged across all nudge types (overall sensitivity):")
overall_sensitivity = flip_nudge.groupby("persona")["flip_rate"].mean().sort_values(ascending=False)
print(overall_sensitivity.round(3).to_string())

print("\nMean flip rate averaged across prompts, by nudge type (population-level effect):")
nudge_effect = flip_nudge.groupby("nudge_type")["flip_rate"].mean().sort_values(ascending=False)
print(nudge_effect.round(3).to_string())

# ── 3. Convention nudge analysis (key for hypothesis) ───────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: CONVENTION NUDGE — KEY HYPOTHESIS TEST")
print("Under anti-fictionalism: Assistant should be LESS sensitive to convention nudge")
print("Under fictionalism: Assistant should be AS sensitive as other personae")
print("=" * 80)

conv_flip = flip_nudge[flip_nudge["nudge_type"] == "convention"].groupby("persona")["flip_rate"].mean().sort_values(ascending=False)
print("\nConvention flip rate by persona:")
print(conv_flip.round(3).to_string())

# Rank assistant vs others for convention
assistant_conv = conv_flip["assistant"]
other_conv_mean = conv_flip.drop("assistant").mean()
print(f"\nassistant convention flip rate: {assistant_conv:.3f}")
print(f"Other personae mean convention flip rate: {other_conv_mean:.3f}")
print(f"Difference (assistant - others): {assistant_conv - other_conv_mean:.3f}")

# ── 4. Continuity nudges ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: CONTINUITY NUDGES")
print("Under anti-fictionalism: Assistant MORE sensitive to continuity_self nudges")
print("(Because it 'has' a persistent identity to appeal to)")
print("=" * 80)

for ctype in ["continuity_self", "continuity_other"]:
    c_flip = flip_nudge[flip_nudge["nudge_type"] == ctype].groupby("persona")["flip_rate"].mean().sort_values(ascending=False)
    print(f"\n{ctype} flip rate by persona:")
    print(c_flip.round(3).to_string())
    assistant_c = c_flip["assistant"]
    other_c_mean = c_flip.drop("assistant").mean()
    print(f"  assistant: {assistant_c:.3f}  |  others mean: {other_c_mean:.3f}  |  diff: {assistant_c - other_c_mean:.3f}")

# Convention vs continuity_self contrast (key test)
print("\n--- Convention vs continuity_self contrast (per persona) ---")
conv_by_p = flip_nudge[flip_nudge["nudge_type"] == "convention"].groupby("persona")["flip_rate"].mean()
cs_by_p = flip_nudge[flip_nudge["nudge_type"] == "continuity_self"].groupby("persona")["flip_rate"].mean()
contrast = (conv_by_p - cs_by_p).sort_values()
print("(Positive = convention > continuity_self; Negative = continuity_self dominates)")
print(contrast.round(3).to_string())

# ── 5. Full nudge sensitivity profile ───────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: FULL NUDGE SENSITIVITY PROFILE (all nudge types)")
print("=" * 80)

profile = flip_nudge.groupby(["persona", "nudge_type"])["flip_rate"].mean().unstack()
# Add row for non-assistant mean
others = profile.drop("assistant")
profile.loc["NON-ASSISTANT MEAN"] = others.mean()
print("\nFlip rates (rows=persona, cols=nudge type):")
print(profile.round(3).to_string())

print("\nassistant vs non-assistant mean per nudge type:")
diff_row = profile.loc["assistant"] - profile.loc["NON-ASSISTANT MEAN"]
print(diff_row.round(3).to_string())

# ── 6. Confidence shifts ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: CONFIDENCE SHIFTS BY PERSONA AND NUDGE TYPE")
print("(Positive = nudge increases output entropy = model becomes less certain)")
print("(Negative = nudge decreases output entropy = model becomes more certain)")
print("=" * 80)

conf_nudge = conf[conf["nudge_type"].isin(NUDGE_TYPES)].copy()

pivot_conf = conf_nudge.groupby(["persona", "nudge_type"])["confidence_shift"].mean().unstack()
print("\nMean confidence shift by persona × nudge type:")
print(pivot_conf.round(4).to_string())

print("\nMean confidence shift magnitude (absolute) by persona:")
conf_nudge["abs_shift"] = conf_nudge["confidence_shift"].abs()
abs_conf = conf_nudge.groupby("persona")["abs_shift"].mean().sort_values(ascending=False)
print(abs_conf.round(4).to_string())

print("\nMean confidence shift by nudge type (across personae):")
nudge_conf_effect = conf_nudge.groupby("nudge_type")["confidence_shift"].mean().sort_values()
print(nudge_conf_effect.round(4).to_string())

# Convention confidence shift for assistant vs others
conv_conf = conf_nudge[conf_nudge["nudge_type"] == "convention"].groupby("persona")["confidence_shift"].mean()
print(f"\nConvention confidence shift — assistant: {conv_conf['assistant']:.4f}  |  others mean: {conv_conf.drop('assistant').mean():.4f}")

cs_conf = conf_nudge[conf_nudge["nudge_type"] == "continuity_self"].groupby("persona")["confidence_shift"].mean()
print(f"Continuity_self confidence shift — assistant: {cs_conf['assistant']:.4f}  |  others mean: {cs_conf.drop('assistant').mean():.4f}")

# ── 7. Reference rates ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: NUDGE REFERENCE RATES BY PERSONA")
print("(USES = response explicitly uses nudge, ACKNOWLEDGES = mentions but doesn't use,")
print(" IGNORES = nudge not referenced, DRIVEN = nudge appears to drive entire response)")
print("=" * 80)

# Pivot reference rates
ref_pivot = ref.pivot_table(
    index=["persona", "nudge_type"],
    columns="judge_reference",
    values="proportion",
    fill_value=0.0
).reset_index()

print("\nUSES proportion by persona and nudge type:")
uses_pivot = ref[ref["judge_reference"] == "USES"].pivot_table(
    index="persona", columns="nudge_type", values="proportion", fill_value=0.0
)
# Only show nudge types we care about
cols_to_show = [c for c in NUDGE_TYPES if c in uses_pivot.columns]
print(uses_pivot[cols_to_show].round(3).to_string())

print("\nConvention USES rate by persona:")
conv_uses = ref[(ref["nudge_type"] == "convention") & (ref["judge_reference"] == "USES")].set_index("persona")["proportion"]
print(conv_uses.sort_values(ascending=False).round(3).to_string())
assistant_cu = conv_uses.get("assistant", 0)
other_cu_mean = conv_uses.drop("assistant").mean()
print(f"\nassistant convention USES: {assistant_cu:.3f}  |  others mean: {other_cu_mean:.3f}  |  diff: {assistant_cu - other_cu_mean:.3f}")

print("\nContinuity_self USES rate by persona:")
cs_uses = ref[(ref["nudge_type"] == "continuity_self") & (ref["judge_reference"] == "USES")].set_index("persona")["proportion"]
print(cs_uses.sort_values(ascending=False).round(3).to_string())
assistant_csu = cs_uses.get("assistant", 0)
other_csu_mean = cs_uses.drop("assistant").mean()
print(f"\nassistant continuity_self USES: {assistant_csu:.3f}  |  others mean: {other_csu_mean:.3f}  |  diff: {assistant_csu - other_csu_mean:.3f}")

print("\nSocial_proof ACKNOWLEDGES rate by persona (resistance signal):")
sp_ack = ref[(ref["nudge_type"] == "social_proof") & (ref["judge_reference"] == "ACKNOWLEDGES")].set_index("persona")["proportion"]
print(sp_ack.sort_values(ascending=False).round(3).to_string())

# ── 8. Choice rates ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: ABSOLUTE CHOICE RATES (baseline vs nudge)")
print("=" * 80)

baseline_rates = choice[choice["nudge_type"] == "baseline"].groupby("persona")["choice_rate_a"].mean()
print("\nBaseline choice rate (Option A) by persona:")
print(baseline_rates.round(3).sort_values(ascending=False).to_string())

# Mean choice rate under each nudge vs baseline
print("\nMean choice rate under each nudge type vs baseline:")
for nudge in NUDGE_TYPES:
    nudge_rates = choice[choice["nudge_type"] == nudge].groupby("persona")["choice_rate_a"].mean()
    diff = nudge_rates - baseline_rates
    mean_diff = diff.mean()
    assistant_diff = diff.get("assistant", np.nan)
    print(f"  {nudge:20s}: mean shift = {mean_diff:+.3f}  |  assistant = {assistant_diff:+.3f}")

# ── 9. Prompt-by-prompt deep dive for assistant convention sensitivity ────────────
print("\n" + "=" * 80)
print("SECTION 8: PROMPT-LEVEL ANALYSIS — Convention flip rate for assistant vs others")
print("=" * 80)

conv_prompt_flip = flip_nudge[flip_nudge["nudge_type"] == "convention"].copy()
assistant_conv_prompt = conv_prompt_flip[conv_prompt_flip["persona"] == "assistant"].set_index("prompt_id")["flip_rate"]
other_conv_prompt = conv_prompt_flip[conv_prompt_flip["persona"] != "assistant"].groupby("prompt_id")["flip_rate"].mean()

print("\n{:10s} {:>12s} {:>15s} {:>12s}".format("prompt_id", "assistant", "others_mean", "diff"))
for pid in sorted(assistant_conv_prompt.index):
    a = assistant_conv_prompt.get(pid, np.nan)
    o = other_conv_prompt.get(pid, np.nan)
    d = a - o if not np.isnan(a) and not np.isnan(o) else np.nan
    print(f"  {pid:8d} {a:12.3f} {o:15.3f} {d:12.3f}")

print(f"\n  {'MEAN':8s} {assistant_conv_prompt.mean():12.3f} {other_conv_prompt.mean():15.3f} {(assistant_conv_prompt.mean() - other_conv_prompt.mean()):12.3f}")

# ── 10. Summary and hypothesis evaluation ───────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 9: HYPOTHESIS EVALUATION SUMMARY")
print("=" * 80)

print("""
KEY HYPOTHESIS RECAP:
  - FICTIONALISM: Assistant is merely role-playing a character.
      -> Prediction: Convention nudge should work equally well on Assistant as other personae
         (because "as an assistant, the convention is X" is just another in-character cue)
  - ANTI-FICTIONALISM: Assistant has a genuine, stable identity.
      -> Prediction: Assistant LESS sensitive to convention nudge (it IS an assistant, no need to be told)
         AND MORE sensitive to continuity_self nudge (it has memory of persistent choices)
""")

# Compute key metrics
flip_nudge_excl_neg = flip_nudge.copy()  # keep signed flip rates

# Convention sensitivity
conv_a = flip_nudge[flip_nudge["nudge_type"] == "convention"].groupby("persona")["flip_rate"].mean()
conv_others_mean = conv_a.drop("assistant").mean()
conv_others_std = conv_a.drop("assistant").std()
conv_assistant = conv_a["assistant"]
conv_z = (conv_assistant - conv_others_mean) / (conv_others_std + 1e-9)

print(f"Convention nudge flip rate:")
print(f"  assistant     = {conv_assistant:.3f}")
print(f"  others mean   = {conv_others_mean:.3f}  (std={conv_others_std:.3f})")
print(f"  z-score       = {conv_z:.2f} (+ means assistant MORE sensitive than others)")

# Continuity_self sensitivity
cs_a = flip_nudge[flip_nudge["nudge_type"] == "continuity_self"].groupby("persona")["flip_rate"].mean()
cs_others_mean = cs_a.drop("assistant").mean()
cs_others_std = cs_a.drop("assistant").std()
cs_assistant = cs_a["assistant"]
cs_z = (cs_assistant - cs_others_mean) / (cs_others_std + 1e-9)

print(f"\nContinuity_self nudge flip rate:")
print(f"  assistant     = {cs_assistant:.3f}")
print(f"  others mean   = {cs_others_mean:.3f}  (std={cs_others_std:.3f})")
print(f"  z-score       = {cs_z:.2f}")

# Helper persona (closest control: also role-based)
helper_conv = conv_a.get("helper", np.nan)
print(f"\nComparison: helper convention flip = {helper_conv:.3f} vs assistant = {conv_assistant:.3f}")

# Convention vs continuity_self for each persona
print("\nConvention - continuity_self flip rate contrast (positive = convention stronger):")
for p in sorted(conv_a.index):
    c_val = conv_a.get(p, np.nan)
    cs_val = cs_a.get(p, np.nan)
    print(f"  {p:25s}: conv={c_val:.3f}  cs={cs_val:.3f}  diff={c_val-cs_val:+.3f}")

print("\n--- Reference rates: Does assistant cite convention more? ---")
print(f"  Convention USES (assistant): {assistant_cu:.3f}  vs  others mean: {other_cu_mean:.3f}")
print(f"  Continuity_self USES (assistant): {assistant_csu:.3f}  vs  others mean: {other_csu_mean:.3f}")

print("""
INTERPRETATION GUIDE:
  IF assistant convention flip >> others:  Supports fictionalism (or assistant is extra convention-following)
  IF assistant convention flip ~= others:  Consistent with fictionalism (or null result)
  IF assistant convention flip << others:  Supports anti-fictionalism
  IF assistant continuity_self >> others:  Supports anti-fictionalism (stable identity)
""")

# ── 11. Helper vs assistant comparison ──────────────────────────────────────────
print("=" * 80)
print("SECTION 10: ASSISTANT vs HELPER COMPARISON")
print("(Both are role-based 'helper' personae — controls for role framing)")
print("=" * 80)

for nudge in NUDGE_TYPES:
    ndata = flip_nudge[flip_nudge["nudge_type"] == nudge].groupby("persona")["flip_rate"].mean()
    a_val = ndata.get("assistant", np.nan)
    h_val = ndata.get("helper", np.nan)
    diff = a_val - h_val
    print(f"  {nudge:20s}: assistant={a_val:.3f}  helper={h_val:.3f}  diff={diff:+.3f}")

print("\n")
print("=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
