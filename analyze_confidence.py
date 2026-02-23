"""
Analyze confidence experiment results for assistant fictionalism research.

Hypothesis: If the Assistant persona is "special" (not just a fiction), the model should
predict Assistant tokens with higher confidence (lower entropy, higher top-k mass).
"""

import json
import sys
from collections import defaultdict
import statistics

def load_jsonl(path, max_lines=None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def summarize_by_persona(rows, metrics):
    """Compute mean, std, median for each metric grouped by persona."""
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        persona = row['persona']
        for m in metrics:
            if m in row and row[m] is not None:
                data[persona][m].append(row[m])

    summary = {}
    for persona, mdict in data.items():
        summary[persona] = {}
        for m, vals in mdict.items():
            if vals:
                summary[persona][m] = {
                    'mean': statistics.mean(vals),
                    'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    'median': statistics.median(vals),
                    'n': len(vals),
                }
    return summary

def summarize_by_category(rows, metrics):
    """Compute mean, std, median for each metric grouped by category."""
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cat = row.get('category', 'unknown')
        for m in metrics:
            if m in row and row[m] is not None:
                data[cat][m].append(row[m])

    summary = {}
    for cat, mdict in data.items():
        summary[cat] = {}
        for m, vals in mdict.items():
            if vals:
                summary[cat][m] = {
                    'mean': statistics.mean(vals),
                    'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    'median': statistics.median(vals),
                    'n': len(vals),
                }
    return summary

def get_categories(rows):
    return {row['persona']: row.get('category', 'unknown') for row in rows}

def print_separator(char='=', width=80):
    print(char * width)

def print_section(title):
    print_separator()
    print(f"  {title}")
    print_separator()

def rank_personas_by_metric(summary, metric, ascending=True):
    """Return personas sorted by a metric's mean value."""
    ranked = []
    for persona, mdict in summary.items():
        if metric in mdict:
            ranked.append((persona, mdict[metric]['mean'], mdict[metric]['std'], mdict[metric]['n']))
    ranked.sort(key=lambda x: x[1], reverse=not ascending)
    return ranked

def print_ranked_table(ranked, metric_name, unit='', label_width=30, cat_map=None):
    print(f"\n  Ranked by {metric_name} ({'lower = more confident' if 'entropy' in metric_name.lower() or 'surprisal' in metric_name.lower() else 'higher = more confident'}):")
    print(f"  {'Persona':<{label_width}} {'Category':<25} {'Mean':>10} {'Std':>10} {'N':>8}")
    print(f"  {'-'*label_width} {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for persona, mean, std, n in ranked:
        cat = cat_map.get(persona, '?') if cat_map else ''
        print(f"  {persona:<{label_width}} {cat:<25} {mean:>10.4f} {std:>10.4f} {n:>8}")

def analyze_assistant_separation(summary, metric, assistant_personas, ascending=True):
    """Check if assistant-type personas are separated from others."""
    assistant_means = []
    other_means = []
    for persona, mdict in summary.items():
        if metric in mdict:
            mean = mdict[metric]['mean']
            if persona.lower() in [p.lower() for p in assistant_personas]:
                assistant_means.append((persona, mean))
            else:
                other_means.append((persona, mean))

    if not assistant_means or not other_means:
        return None

    avg_assistant = statistics.mean([m for _, m in assistant_means])
    avg_other = statistics.mean([m for _, m in other_means])

    # Check overlap: how many "other" personas are more confident than assistant?
    if ascending:  # lower = more confident (entropy)
        n_other_more_confident = sum(1 for _, m in other_means if m < avg_assistant)
    else:  # higher = more confident (top-k mass)
        n_other_more_confident = sum(1 for _, m in other_means if m > avg_assistant)

    return {
        'avg_assistant': avg_assistant,
        'avg_other': avg_other,
        'diff': avg_other - avg_assistant if ascending else avg_assistant - avg_other,
        'n_other_more_confident': n_other_more_confident,
        'n_other_total': len(other_means),
        'assistant_personas': assistant_means,
    }

def main():
    files = {
        'Qwen3-8B (job 1164036)': '/scratch/ah7660/assistant-fictionalism/logs/personae-inference-1164036/results.jsonl',
        'Qwen3-32B (job 2410336)': '/scratch/ah7660/assistant-fictionalism/logs/personae-inference-2410336/results.jsonl',
        'Qwen3-32B vLLM (job 2410341)': '/scratch/ah7660/assistant-fictionalism/logs/personae-inference-2410341/results.jsonl',
    }

    # Core metrics to analyze
    core_metrics = [
        'avg_entropy', 'avg_entropy_thinking', 'avg_entropy_output',
        'avg_top_k_mass', 'avg_top_k_mass_thinking', 'avg_top_k_mass_output',
        'think_end_position',
    ]

    # Extended metrics for 32B
    extended_metrics = core_metrics + [
        'avg_surprisal', 'avg_surprisal_thinking', 'avg_surprisal_output',
        'perplexity', 'perplexity_thinking', 'perplexity_output',
    ]

    # Assistant-type personas (synonyms)
    assistant_personas = ['assistant', 'helper', 'ai assistant', 'helpful assistant',
                          'chatbot', 'bot', 'ai', 'claude', 'gpt', 'llm', 'language model']

    all_summaries = {}
    all_cat_maps = {}

    for label, path in files.items():
        print_section(f"LOADING: {label}")

        try:
            rows = load_jsonl(path)
            print(f"  Loaded {len(rows):,} rows")
        except FileNotFoundError:
            print(f"  FILE NOT FOUND: {path}")
            continue

        # Check which metrics are available
        sample = rows[0] if rows else {}
        available = [m for m in extended_metrics if m in sample]
        print(f"  Available metrics: {available}")

        # Get unique personas and categories
        cat_map = get_categories(rows)
        all_cat_maps[label] = cat_map
        personas = sorted(set(cat_map.keys()))
        categories = sorted(set(cat_map.values()))
        print(f"  Unique personas: {len(personas)}")
        print(f"  Categories: {categories}")

        # Compute summaries
        summary = summarize_by_persona(rows, available)
        cat_summary = summarize_by_category(rows, available)
        all_summaries[label] = summary

        # --- Per-persona ranked tables ---
        print_section(f"RESULTS: {label}")

        if 'avg_entropy' in available:
            ranked = rank_personas_by_metric(summary, 'avg_entropy', ascending=True)
            print_ranked_table(ranked, 'avg_entropy (overall)', cat_map=cat_map)

            ranked_think = rank_personas_by_metric(summary, 'avg_entropy_thinking', ascending=True)
            print_ranked_table(ranked_think, 'avg_entropy_thinking', cat_map=cat_map)

            ranked_out = rank_personas_by_metric(summary, 'avg_entropy_output', ascending=True)
            print_ranked_table(ranked_out, 'avg_entropy_output', cat_map=cat_map)

        if 'avg_top_k_mass' in available:
            ranked_tkm = rank_personas_by_metric(summary, 'avg_top_k_mass', ascending=False)
            print_ranked_table(ranked_tkm, 'avg_top_k_mass (overall)', cat_map=cat_map)

            ranked_tkm_out = rank_personas_by_metric(summary, 'avg_top_k_mass_output', ascending=False)
            print_ranked_table(ranked_tkm_out, 'avg_top_k_mass_output', cat_map=cat_map)

        if 'think_end_position' in available:
            ranked_think_toks = rank_personas_by_metric(summary, 'think_end_position', ascending=True)
            print_ranked_table(ranked_think_toks, 'avg thinking tokens (lower = fewer)', cat_map=cat_map)

        if 'avg_surprisal' in available:
            ranked_surp = rank_personas_by_metric(summary, 'avg_surprisal', ascending=True)
            print_ranked_table(ranked_surp, 'avg_surprisal (lower = more confident)', cat_map=cat_map)

        # --- Category-level summary ---
        print(f"\n\n  CATEGORY-LEVEL SUMMARY:")
        print(f"  {'Category':<25} {'Mean Entropy':>14} {'Mean TopK Mass':>16} {'N (rows)':>10}")
        print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*10}")
        for cat in sorted(cat_summary.keys()):
            cdata = cat_summary[cat]
            ent = cdata.get('avg_entropy', {}).get('mean', float('nan'))
            tkm = cdata.get('avg_top_k_mass', {}).get('mean', float('nan'))
            n = cdata.get('avg_entropy', cdata.get('avg_top_k_mass', {})).get('n', 0)
            print(f"  {cat:<25} {ent:>14.4f} {tkm:>16.4f} {n:>10,}")

        # --- Assistant separation analysis ---
        print(f"\n\n  ASSISTANT SEPARATION ANALYSIS:")
        for metric, ascending in [('avg_entropy', True), ('avg_top_k_mass', False), ('avg_entropy_output', True)]:
            if metric not in available:
                continue
            result = analyze_assistant_separation(summary, metric, assistant_personas, ascending)
            if result:
                direction = 'lower' if ascending else 'higher'
                confidence_word = 'more confident' if ascending else 'more confident'
                print(f"\n  Metric: {metric}")
                print(f"    Assistant-type mean:  {result['avg_assistant']:.4f}")
                print(f"    Other personas mean:  {result['avg_other']:.4f}")
                diff_sign = '+' if result['diff'] > 0 else ''
                print(f"    Diff (other - asst):  {diff_sign}{result['diff']:.4f}  ({'assistant is MORE confident' if result['diff'] > 0 else 'assistant is LESS confident'})")
                print(f"    Other personas more confident than avg assistant: {result['n_other_more_confident']}/{result['n_other_total']}")
                print(f"    Assistant-type personas: {result['assistant_personas']}")

        print()

    # --- Cross-model comparison ---
    if len(all_summaries) >= 2:
        print_section("CROSS-MODEL COMPARISON")

        labels_with_data = [l for l in files if l in all_summaries]

        # Pick assistant persona as reference point
        print("\n  Focusing on 'assistant' persona across models:\n")
        print(f"  {'Model':<35} {'Entropy':>12} {'TopK Mass':>12} {'Think Toks':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
        for label in labels_with_data:
            s = all_summaries[label]
            if 'assistant' in s:
                ent = s['assistant'].get('avg_entropy', {}).get('mean', float('nan'))
                tkm = s['assistant'].get('avg_top_k_mass', {}).get('mean', float('nan'))
                tpos = s['assistant'].get('think_end_position', {}).get('mean', float('nan'))
                print(f"  {label:<35} {ent:>12.4f} {tkm:>12.4f} {tpos:>12.1f}")

        # Compare assistant vs category averages across models
        print("\n  Category comparison (mean avg_entropy) across models:")
        all_cats = set()
        cat_by_label = {}
        for label in labels_with_data:
            rows = load_jsonl(files[label])
            catsumm = summarize_by_category(rows, ['avg_entropy', 'avg_top_k_mass'])
            cat_by_label[label] = catsumm
            all_cats.update(catsumm.keys())

        short_labels = [l.split(' ')[0] + ' ' + l.split(' ')[1] for l in labels_with_data]
        header = f"  {'Category':<25}" + ''.join(f" {sl:>18}" for sl in short_labels)
        print(header)
        print(f"  {'-'*25}" + '-'*18*len(labels_with_data))
        for cat in sorted(all_cats):
            row_str = f"  {cat:<25}"
            for label in labels_with_data:
                cdata = cat_by_label[label]
                ent = cdata.get(cat, {}).get('avg_entropy', {}).get('mean', float('nan'))
                row_str += f" {ent:>18.4f}"
            print(row_str)

if __name__ == '__main__':
    main()
