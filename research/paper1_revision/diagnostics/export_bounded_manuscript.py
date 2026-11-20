"""Export saved historical follow-up tables; no fits or physical solves."""
from pathlib import Path
import pandas as pd
from ..data import ROOT, read, put, sha
from ..bounded_ablation.pipeline import DEST, identity

BS = chr(92)
BR = BS * 2
OUT = ROOT / 'research/three_papers/manuscripts/paper1/results'
ORDER = ['equal4', 'cost_priority', 'co2_priority', 'time_priority', 'service4_co2']
LABELS = ['Equal', 'Cost', 'CO$_2$', 'Time', 'Service']
ARMS = ['current_clip', 'linear_no_b9_clip', 'fractional_logit', 'fractional_logit_no_b9']


def table(body, caption, label, alignment='lrrrr'):
    return (BS + 'begin{table}[htbp]' + BS + 'centering' + BS + 'small\n'
            + BS + 'begin{tabular}{' + alignment + '}' + BS + 'toprule\n'
            + body + BS + 'bottomrule\n' + BS + 'end{tabular}\n'
            + BS + 'caption{' + caption + '}' + BS + 'label{' + label + '}\n'
            + BS + 'end{table}\n')


def panel(frame, title):
    text = BS + 'multicolumn{5}{l}{' + BS + 'textit{' + title + '}}' + BR + '\n'
    for target, label in zip(ORDER, LABELS):
        row = frame.loc[target]
        text += label + ' & ' + ' & '.join(f'{row[a]:.5f}' for a in ARMS) + BR + '\n'
    return text


def main():
    original = identity()
    if original != read(DEST / 'design.json')['inputs'] or read(DEST / 'status.json')['status'] != 'complete':
        raise ValueError('Follow-up sources changed or results incomplete')
    comparison = pd.read_csv(DEST / 'comparison.csv')
    header = ('Preference & ' + BS + 'shortstack{Projected' + BR + 'linear} & '
              + BS + 'shortstack{Linear' + BR + 'without B9} & '
              + BS + 'shortstack{Fractional' + BR + 'logit} & '
              + BS + 'shortstack{Fractional logit' + BR + 'without B9}' + BR + BS + 'midrule\n')
    generated = []
    for rep, filename, label in [('realized', 'bounded_milp_table.tex', 'tab:bounded_milp'),
                                  ('legacy_raw', 'bounded_legacy_table.tex', 'tab:bounded_legacy')]:
        q = comparison[comparison.representation == rep]
        body = header
        for scheme, title in [('nested', 'Scenario-grouped validation'), ('lofo', 'Family-held-out validation')]:
            if scheme == 'lofo':
                body += BS + 'addlinespace\n'
            body += panel(q[q.scheme == scheme].set_index('target'), title)
        descriptor = 'realized MILP descriptors' if rep == 'realized' else 'original raw generator descriptors'
        caption = ('Exploratory historical MSE using ' + descriptor + '. Each cell averages 3,000 '
            'observations (600 scenarios, five deadlines). Both linear columns deliver predictions '
            'projected onto $[0,1]$; fractional-logistic outputs are bounded by their link. '
            'Models and tuning are specific to each deadline. The independent campaign contributes no outcomes.')
        path = OUT / filename
        path.write_text(table(body, caption, label), encoding='utf-8')
        generated.append(path)
    low = pd.read_csv(DEST / 'lofo_family_scores.csv')
    low = low[(low.representation == 'realized') & (low.family == 'low_criticality_scenarios')]
    low = low.pivot(index='target', columns='arm', values='mse')
    path = OUT / 'bounded_low_family_table.tex'
    path.write_text(table(header + panel(low, 'Low-criticality family omitted from training'),
        'Historical MSE on the 100 low-criticality scenarios, averaging five deadlines per scenario. '
        'The bounded model without B9 improves four preferences relative to the projected linear '
        'comparator, but time priority remains worse. These 500 rows per preference are not independent scenarios.',
        'tab:bounded_low_family'), encoding='utf-8')
    generated.append(path)
    c = pd.read_csv(DEST / 'paired_contrasts.csv')
    c = c[(c.representation == 'realized') & (c.arm == 'fractional_logit_no_b9')]
    body = 'Preference & MSE reduction & Lower & Upper' + BR + BS + 'midrule\n'
    for scheme, ref, title in [
        ('nested', 'current_clip', 'Against projected linear: scenario folds'),
        ('lofo', 'current_clip', 'Against projected linear: family holdout'),
        ('lofo', 'fractional_logit', 'Against full fractional logit: family holdout')]:
        body += BS + 'multicolumn{4}{l}{' + BS + 'textit{' + title + '}}' + BR + '\n'
        q = c[(c.scheme == scheme) & (c.reference == ref)].set_index('target')
        for target, label in zip(ORDER, LABELS):
            row = q.loc[target]
            body += label + ' & ' + ' & '.join(f'{row[a]:.6f}' for a in
                ['mse_gain', 'simultaneous_lower', 'simultaneous_upper']) + BR + '\n'
        body += BS + 'addlinespace\n'
    path = OUT / 'bounded_contrasts_table.tex'
    path.write_text(table(body,
        'Positive differences favor the realized fractional-logistic model without B9. '
        'Approximate 95' + BS + '% simultaneous intervals cover 40 declared contrasts within each '
        'validation scheme, with 2,000 paired scenario resamples after averaging deadlines. '
        'They condition on the fitted predictions and do not account for post-hoc design selection or refitting uncertainty.',
        'tab:bounded_contrasts', 'lrrr'), encoding='utf-8')
    generated.append(path)
    if identity() != original:
        raise ValueError('Frozen inputs changed during export')
    sources = [DEST / name for name in ['comparison.csv', 'lofo_family_scores.csv', 'paired_contrasts.csv',
                                       'design.json', 'validation.json']]
    # This provenance file belongs to the manuscript, not to a frozen run.
    import json
    receipt = dict(scope='Historical exploratory results only; no independent outcomes',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
        table_hashes={str(p.relative_to(ROOT)): sha(p) for p in generated},
        exporter_sha256=sha(__file__), frozen_comparison_unchanged=True, model_refits=0, milp_solves=0)
    (OUT / 'bounded_manuscript_provenance.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Exported', len(generated), 'tables from verified historical results.')


if __name__ == '__main__':
    main()
