"""Create an editable SVG illustrating the method (no external dependencies)."""
from pathlib import Path

SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="440" viewBox="0 0 1000 440" role="img" aria-label="Q-ACO-Elite method diagram">
<defs>
  <linearGradient id="bg" x2="1" y2="1"><stop stop-color="#f9fbfd"/><stop offset="1" stop-color="#eef3f9"/></linearGradient>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0 0L9 3L0 6Z" fill="#546579"/></marker>
</defs>
<rect width="1000" height="440" rx="18" fill="url(#bg)"/>
<text x="45" y="52" font-family="Arial,sans-serif" font-size="26" font-weight="700" fill="#142b44">Q-ACO-Elite: Reproduction Workflow</text>
<text x="45" y="79" font-family="Arial,sans-serif" font-size="14" fill="#607387">2026 AGV path planning paper | 20 × 20 connected grid | 8-neighbor, no corner cutting</text>
<g font-family="Arial,sans-serif" text-anchor="middle">
<rect x="40" y="130" width="190" height="120" rx="12" fill="#e4eefb" stroke="#aac5e8"/>
<text x="135" y="169" font-size="18" font-weight="700" fill="#215489">Grid world</text>
<text x="135" y="198" font-size="14" fill="#345879">Obstacle probabilities</text>
<text x="135" y="218" font-size="14" fill="#345879">0.2 / 0.3 / 0.4</text>
<rect x="280" y="130" width="200" height="120" rx="12" fill="#e7f4ea" stroke="#aad9bc"/>
<text x="380" y="166" font-size="18" font-weight="700" fill="#276344">Improved Q-learning</text>
<text x="380" y="192" font-size="14" fill="#3d7056">Boltzmann annealing</text>
<text x="380" y="214" font-size="14" fill="#3d7056">+ Euclidean heuristic</text>
<rect x="530" y="130" width="200" height="120" rx="12" fill="#fff3e1" stroke="#e9c990"/>
<text x="630" y="166" font-size="18" font-weight="700" fill="#81501d">Q-guided ACO</text>
<text x="630" y="192" font-size="14" fill="#87602d">Q(s,a) → positive prior</text>
<text x="630" y="214" font-size="14" fill="#87602d">τᵅ · ηᵝ · priorᵛ</text>
<rect x="780" y="130" width="180" height="120" rx="12" fill="#f1e9fb" stroke="#d2b9eb"/>
<text x="870" y="166" font-size="18" font-weight="700" fill="#60438d">Elite update</text>
<text x="870" y="192" font-size="14" fill="#73549a">Global-best deposit</text>
<text x="870" y="214" font-size="14" fill="#73549a">E / L_best</text>
</g>
<g stroke="#546579" stroke-width="2.7" fill="none" marker-end="url(#arrow)">
<path d="M231 190H273"/><path d="M480 190H523"/><path d="M730 190H773"/>
<path d="M870 256V302H630V255" stroke-dasharray="6 5"/>
</g>
<rect x="230" y="335" width="540" height="54" rx="10" fill="white" stroke="#b7c7d9"/>
<text x="500" y="358" font-family="Arial,sans-serif" text-anchor="middle" font-size="17" font-weight="700" fill="#253b52">Evaluation: path length · first-best iteration · turning cost</text>
<text x="500" y="376" font-family="Arial,sans-serif" text-anchor="middle" font-size="12" fill="#536b85">Plain-ACO vs Q-ACO vs Q-ACO-Elite | Dijkstra reference optimum</text>
<path d="M870 306V315H500V330" fill="none" stroke="#546579" stroke-width="2.3" marker-end="url(#arrow)"/>
</svg>'''

if __name__ == '__main__':
    dest = Path(__file__).resolve().parents[1] / 'assets' / 'method_flowchart.svg'
    dest.write_text(SVG, encoding='utf-8')
    print(dest)
