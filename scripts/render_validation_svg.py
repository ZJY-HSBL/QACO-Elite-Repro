"""Generate editable SVG graphics directly from recorded paired-validation CSV."""
import argparse
import csv
from pathlib import Path
from html import escape

METHODS = ('Plain-ACO', 'Q-ACO', 'Q-ACO-Elite')
COLORS = {'Plain-ACO': '#d97845', 'Q-ACO': '#387daf', 'Q-ACO-Elite': '#238d79'}


def _records(path):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def draw_lengths(summary, target):
    w, h = 930, 480
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Repeated random-map mean path lengths">',
              '<rect width="100%" height="100%" fill="#fcfdff"/>',
              '<text x="42" y="44" fill="#182b42" font-size="23" font-family="Arial" font-weight="bold">Mean Path Length · Independent 6-Seed Validation</text>',
              '<text x="42" y="70" fill="#657489" font-size="13" font-family="Arial">Paired 20×20 random maps; lower is better. Not the authors’ maps.</text>']
    plot_top, plot_bottom = 115, 390
    scale = (plot_bottom - plot_top) / 50
    for t in range(0, 51, 10):
        y = plot_bottom - t * scale
        output += [f'<line x1="85" x2="890" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e9f0"/>',
                   f'<text x="68" y="{y+5:.1f}" text-anchor="end" fill="#546477" font-size="12" font-family="Arial">{t}</text>']
    for di, density in enumerate((.2, .3, .4)):
        xcenter = 220 + di * 270
        for j, method in enumerate(METHODS):
            r = next(r for r in summary if float(r['density']) == density and r['method'] == method)
            length = float(r['length_mean'])
            x = xcenter + (j - 1) * 55 - 23
            y = plot_bottom - length * scale
            output.append(f'<rect x="{x}" y="{y:.2f}" width="46" height="{length*scale:.2f}" rx="4" fill="{COLORS[method]}"/>')
            output.append(f'<text x="{x+23}" y="{y-8:.1f}" font-size="12" font-family="Arial" text-anchor="middle" fill="#283648">{length:.2f}</text>')
        output.append(f'<text x="{xcenter}" y="420" text-anchor="middle" font-family="Arial" fill="#253951">Density {density:.1f}</text>')
    for j, method in enumerate(METHODS):
        x = 200 + j * 230
        output += [f'<rect x="{x}" y="446" width="14" height="14" fill="{COLORS[method]}" rx="2"/>',
                   f'<text x="{x+23}" y="458" font-size="13" font-family="Arial" fill="#42516a">{escape(method)}</text>']
    output.append('</svg>')
    Path(target).write_text('\n'.join(output) + '\n', encoding='utf-8')


def draw_effects(effects, target):
    rows = [r for r in effects if r['metric'] == 'best_length']
    w, h = 1040, 760
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Paired path-length improvements with bootstrap confidence intervals">',
              '<rect width="100%" height="100%" fill="#fcfdff"/>',
              '<text x="38" y="42" fill="#152c47" font-size="23" font-weight="bold" font-family="Arial">Paired Path-Length Differences · 95% Bootstrap CIs</text>',
              '<text x="38" y="67" fill="#60738d" font-size="13" font-family="Arial">Baseline minus candidate; positive = candidate shorter. N=6 maps per density, Holm-adjusted p-values in CSV.</text>']
    left, width, x_min, x_max = 475, 510, -1.5, 9
    xcoord = lambda v: left + (v-x_min)*width/(x_max-x_min)
    for tick in (-1, 0, 2, 4, 6, 8):
        x = xcoord(tick)
        output += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="95" y2="710" stroke="{("#8f9fb0" if tick==0 else "#e4eaf2")}" stroke-dasharray="{("5 4" if tick==0 else "0")}"/>',
                   f'<text x="{x:.1f}" y="733" text-anchor="middle" font-size="12" font-family="Arial" fill="#52657e">{tick}</text>']
    for i, r in enumerate(rows):
        y = 116 + i*48
        lo, hi, mean = (float(r[k]) for k in ('ci_low','ci_high','improvement_mean'))
        color = '#238d79' if lo > 0 else '#c3843e' if hi < 0 else '#60738d'
        label = r['before'] + ' → ' + r['after']
        output.append(f'<text x="34" y="{y+5}" font-family="Arial" font-size="13" fill="#284059">{escape(str(r["density"]))} · {escape(label)}</text>')
        output += [f'<line x1="{xcoord(lo):.1f}" x2="{xcoord(hi):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                   f'<line x1="{xcoord(lo):.1f}" x2="{xcoord(lo):.1f}" y1="{y-7}" y2="{y+7}" stroke="{color}" stroke-width="2"/>',
                   f'<line x1="{xcoord(hi):.1f}" x2="{xcoord(hi):.1f}" y1="{y-7}" y2="{y+7}" stroke="{color}" stroke-width="2"/>',
                   f'<circle cx="{xcoord(mean):.1f}" cy="{y}" r="5" fill="{color}"/>']
        if i in (3, 7):
            output.append(f'<line x1="30" x2="985" y1="{y+22}" y2="{y+22}" stroke="#dfe6ee"/>')
    output += ['<text x="730" y="751" font-family="Arial" text-anchor="middle" font-size="12" fill="#536a86">Mean paired reduction in grid units</text>','</svg>']
    Path(target).write_text('\n'.join(output) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/validation'))
    parser.add_argument('--output', type=Path, default=Path('assets'))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    draw_lengths(_records(args.input / 'summary.csv'), args.output / 'validation_mean_lengths.svg')
    draw_effects(_records(args.input / 'paired_effects.csv'), args.output / 'validation_paired_effects.svg')


if __name__ == '__main__':
    main()
