"""Render lightweight, editable SVG evidence from the actual stored experiment data."""
from pathlib import Path
import json
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
COLORS = {'Plain-ACO':'#e8793a','Q-ACO':'#3178a5','Q-ACO-Elite':'#27926d'}
METHODS = tuple(COLORS)


def header(w,h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">',
            f'<rect width="{w}" height="{h}" fill="#f7f9fc"/>']


def text(x,y,value,size=15,color='#193650',weight='400'):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-family="Arial,sans-serif" font-weight="{weight}">{value}</text>'


def render_paths(density, trajectories):
    obst = np.load(RESULTS / f'map_{density}.npy')
    data = trajectories[density]
    out = header(960, 340)
    out.append(text(24,35,f'AGV routes | 20x20 | target obstacle fraction {density}',21,weight='700'))
    cell = 11
    blocks = ' '.join(f'M{c*cell} {r*cell}h11v11h-11z' for r,c in np.argwhere(obst))
    out.append(f'<defs><path id="obstacles" d="{blocks}" fill="#273747"/></defs>')
    for j, name in enumerate(METHODS):
        ox,oy = 30 + j*320,70
        out.append(text(ox,oy-9,name,16,weight='700'))
        out.append(f'<rect x="{ox}" y="{oy}" width="220" height="220" fill="white" stroke="#c8d3df"/>')
        out.append(f'<use href="#obstacles" x="{ox}" y="{oy}"/>')
        path = data[name]['path']
        coords = []
        for s in path:
            r,c=divmod(s,20)
            coords.append(f'{(c+0.5)*cell:g},{(r+0.5)*cell:g}')
        if coords:
            out.append(f'<g transform="translate({ox} {oy})"><polyline points="{" ".join(coords)}" stroke="{COLORS[name]}" stroke-width="3" stroke-linejoin="round" fill="none"/></g>')
        out.append(f'<circle cx="{ox+5.5}" cy="{oy+5.5}" r="5" fill="#20aa5a"/>')
        out.append(f'<circle cx="{ox+214.5}" cy="{oy+214.5}" r="5" fill="#e14242"/>')
        out.append(text(ox,309,'green: start   red: goal',12,'#566c80'))
    out.append('</svg>')
    (ASSETS / f'paths_{density}.svg').write_text('\n'.join(out),encoding='utf-8')


def render_convergence(density, trajectories):
    data=trajectories[density]
    W,H=960,360
    out=header(W,H)
    out.append(text(38,36,f'Best-so-far convergence | obstacle density {density}',21,weight='700'))
    xs=[v for m in METHODS for v in data[m]['best_so_far'] if v is not None]
    lo=min(xs)-1; hi=max(xs)+1
    x1,x2,y1,y2=80,925,65,290
    out.append(f'<path d="M{x1} {y1}V{y2}H{x2}" stroke="#8a9aab" fill="none" stroke-width="1.5"/>')
    for j in range(5):
        y=y2-j*(y2-y1)/4; val=lo+j*(hi-lo)/4
        out.append(f'<path d="M{x1} {y:g}H{x2}" stroke="#d6dfe8"/>')
        out.append(text(20,y+5,f'{val:.1f}',12,'#52687b'))
    for m in METHODS:
        vals=data[m]['best_so_far']
        parts=[]; prev=None
        for i,v in enumerate(vals):
            if v is None:
                continue
            x=x1+i*(x2-x1)/99; y=y2-(v-lo)/(hi-lo)*(y2-y1)
            if prev is None:
                parts.append(f'M{x:.1f} {y:.1f}')
            elif v != prev:
                parts.append(f'H{x:.1f}V{y:.1f}')
            prev=v
        if prev is not None:
            parts.append(f'H{x2}')
        out.append(f'<path d="{" ".join(parts)}" stroke="{COLORS[m]}" stroke-width="2.5" fill="none"/>')
    for j,m in enumerate(METHODS):
        out.append(f'<rect x="{95+j*275}" y="316" width="16" height="8" fill="{COLORS[m]}"/>')
        out.append(text(116+j*275,326,m,14))
    out.append('</svg>')
    (ASSETS / f'convergence_{density}.svg').write_text('\n'.join(out),encoding='utf-8')


def render_summary():
    with (RESULTS / 'results.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    out=header(960,360)
    out.append(text(22,37,'Independent AGV benchmark | path length (lower is better)',21,weight='700'))
    maxv=max(float(r['best_length']) for r in rows)
    for i,density in enumerate(('0.2','0.3','0.4')):
        ox=70+i*310; oy=62
        out.append(text(ox+36,oy+8,f'obstacle p={density}',15,weight='700'))
        for j,m in enumerate(METHODS):
            r=next(x for x in rows if x['density']==density and x['algorithm']==m)
            height=float(r['best_length'])/maxv*214
            x=ox+j*67+5
            out.append(f'<rect x="{x}" y="{294-height:.1f}" width="48" height="{height:.1f}" fill="{COLORS[m]}" rx="2"/>')
            out.append(text(x+4,286-height,f'{float(r["best_length"]):.2f}',12))
        out.append(f'<path d="M{ox} 294H{ox+232}" stroke="#9cabb9"/>')
    for i,m in enumerate(METHODS):
        out.append(f'<rect x="{90+i*275}" y="328" width="15" height="8" fill="{COLORS[m]}"/>')
        out.append(text(111+i*275,338,m,13))
    out.append('</svg>')
    (ASSETS / 'benchmark_summary.svg').write_text('\n'.join(out),encoding='utf-8')


if __name__=='__main__':
    trajectories=json.loads((RESULTS/'trajectories.json').read_text(encoding='utf-8'))
    for density in ('0.2','0.3','0.4'):
        render_paths(density, trajectories)
        render_convergence(density, trajectories)
    render_summary()
    print('SVG images written to', ASSETS)
