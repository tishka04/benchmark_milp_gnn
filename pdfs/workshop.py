import os, textwrap, subprocess, json, math, re, shutil, sys
from pathlib import Path

work = Path("/mnt/data/powdev_beamer_like")
work.mkdir(exist_ok=True)

# 1) Extract useful page images from the manuscript PDF
pdf = "/mnt/data/MILP_GNN_V2.pdf"
pages = [6,9,13,15,16,26,28,29,30,31,39,40,42,63,65]  # 1-indexed pages from manuscript
extract_py = work / "extract_pages.py"
extract_py.write_text(textwrap.dedent(f"""
import fitz, os
from pathlib import Path
pdf = r"{pdf}"
out = Path(r"{work/'assets'}")
out.mkdir(exist_ok=True)
doc = fitz.open(pdf)
pages = {pages}
for p in pages:
    page = doc[p-1]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
    pix.save(str(out / f"page_{{p}}.png"))
print("done", len(list(out.glob("*.png"))))
"""))
subprocess.run([sys.executable, str(extract_py)], check=True)

# 2) Create the PPTX with PptxGenJS
js = work / "make_deck.js"
js.write_text(textwrap.dedent(f"""
const pptxgen = require('pptxgenjs');
const {{
  imageSizingContain,
  imageSizingCrop,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
}} = require('/home/oai/skills/slides/pptxgenjs_helpers');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'OpenAI';
pptx.company = 'OpenAI';
pptx.subject = 'MILP–GNN–EBM for PowDev';
pptx.title = 'Hybrid MILP–GNN–EBM for Large-Scale Scenario Exploration';
pptx.lang = 'en-US';
pptx.theme = {{
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'en-US'
}};

const W = 13.333;
const H = 7.5;

const C = {{
  navy: '10243E',
  blue: '1F5AA6',
  teal: '1C8C8C',
  orange: 'E6862A',
  green: '2F8F5B',
  red: 'B94A48',
  purple: '6B59A3',
  ink: '243447',
  gray: '667085',
  light: 'F6F8FB',
  line: 'D8DEE8',
  white: 'FFFFFF',
  paleBlue: 'EAF1FB',
  paleTeal: 'E9F7F6',
  paleOrange: 'FFF2E8',
  paleGreen: 'EDF8F1',
  paleRed: 'FCEEEE',
}};

const assets = '{(work/"assets").as_posix()}';

function addBg(slide, title, kicker='PowDev project | Manuscript-based overview') {{
  slide.background = {{ color: C.white }};
  slide.addShape(pptx.ShapeType.rect, {{ x:0, y:0, w:W, h:H, fill:{{color:C.white}}, line:{{color:C.white}} }});
  slide.addShape(pptx.ShapeType.rect, {{ x:0, y:0, w:W, h:0.22, fill:{{color:C.navy}}, line:{{color:C.navy}} }});
  slide.addText(kicker, {{
    x:0.5, y:0.32, w:4.8, h:0.2, fontSize:11, color:C.blue, bold:false
  }});
  slide.addText(title, {{
    x:0.5, y:0.62, w:12.2, h:0.45, fontSize:26, bold:true, color:C.navy
  }});
  slide.addShape(pptx.ShapeType.line, {{ x:0.5, y:1.1, w:12.2, h:0, line:{{color:C.line, width:1.1}} }});
}}

function addFooter(slide, txt='Source: manuscript and LaTeX source provided by the authors') {{
  slide.addText(txt, {{
    x:0.5, y:7.08, w:12.2, h:0.18, fontSize:8.5, color:'7A8797', italic:true, align:'right'
  }});
}}

function bullets(slide, items, x, y, w, h, fs=18, color=C.ink) {{
  const runs = [];
  items.forEach((t) => {{
    runs.push({{ text: t, options: {{ bullet: {{ indent: 16 }}, hanging: 3, breakLine: true }} }});
  }});
  slide.addText(runs, {{
    x, y, w, h, fontSize:fs, color, breakLine:false, paraSpaceAfterPt:9, valign:'top'
  }});
}}

function chip(slide, text, x, y, w, fill, color='FFFFFF') {{
  slide.addShape(pptx.ShapeType.roundRect, {{
    x, y, w, h:0.34, rectRadius:0.06,
    fill:{{color:fill}}, line:{{color:fill}}
  }});
  slide.addText(text, {{ x:x+0.08, y:y+0.06, w:w-0.16, h:0.16, fontSize:10.5, color, bold:true, align:'center' }});
}}

function metricBox(slide, x, y, w, h, title, value, sub, fill='F7F9FC', accent=C.blue) {{
  slide.addShape(pptx.ShapeType.roundRect, {{
    x, y, w, h, rectRadius:0.08, fill:{{color:fill}}, line:{{color:C.line, width:1}}
  }});
  slide.addShape(pptx.ShapeType.rect, {{ x, y, w:0.08, h, fill:{{color:accent}}, line:{{color:accent}} }});
  slide.addText(title, {{ x:x+0.18, y:y+0.16, w:w-0.25, h:0.2, fontSize:12.5, color:C.gray, bold:true }});
  slide.addText(value, {{ x:x+0.18, y:y+0.46, w:w-0.25, h:0.35, fontSize:28, color:C.navy, bold:true }});
  slide.addText(sub, {{ x:x+0.18, y:y+0.9, w:w-0.25, h:0.34, fontSize:10.5, color:C.gray }});
}}

function addNotes(slide, lines) {{
  if (slide.addNotes) {{
    slide.addNotes(`[Sources]\\n${{lines.join('\\n')}}`);
  }}
}}

function addPageImg(slide, page, x, y, w, h, contain=true) {{
  const path = `${{assets}}/page_${{page}}.png`;
  if (contain) slide.addImage({{ path, ...imageSizingContain(path, x, y, w, h) }});
  else slide.addImage({{ path, ...imageSizingCrop(path, x, y, w, h) }});
}}

//
// Main deck
//

// 1 Title
{{
  const s = pptx.addSlide();
  s.background = {{ color: C.light }};
  s.addShape(pptx.ShapeType.rect, {{ x:0, y:0, w:W, h:H, fill:{{color:C.light}}, line:{{color:C.light}} }});
  s.addShape(pptx.ShapeType.rect, {{ x:0, y:0, w:4.1, h:H, fill:{{color:C.navy}}, line:{{color:C.navy}} }});
  s.addText('Hybrid MILP–GNN–EBM for Large-Scale Scenario Exploration', {{
    x:4.55, y:1.0, w:7.9, h:1.4, fontSize:27, bold:true, color:C.navy
  }});
  s.addText('PowDev presentation for a multidisciplinary audience\\n(climate, maths, power systems, economics, project management)', {{
    x:4.58, y:2.55, w:7.7, h:0.9, fontSize:17, color:C.ink
  }});
  chip(s, 'MILP', 4.58, 3.7, 0.95, C.blue);
  chip(s, 'GNN', 5.7, 3.7, 0.95, C.teal);
  chip(s, 'EBM', 6.82, 3.7, 0.95, C.orange);
  chip(s, 'Power-system flexibility', 7.96, 3.7, 2.15, C.green);
  s.addText('Based on the manuscript by Théotime Coudray & Stéphane Goutte', {{
    x:4.58, y:4.45, w:6.9, h:0.25, fontSize:14, color:C.gray
  }});
  s.addText('Main message', {{
    x:4.58, y:5.15, w:2.0, h:0.22, fontSize:13, bold:true, color:C.blue
  }});
  s.addText('Use exact optimisation to learn a fast exploration engine, then recover feasible operating plans with optimisation again.', {{
    x:4.58, y:5.45, w:7.9, h:0.85, fontSize:19, color:C.navy, bold:true
  }});
  addPageImg(s, 6, 0.32, 0.78, 3.38, 5.95, false);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF: MILP_GNN_V2.pdf, especially pipeline figure on manuscript page 6.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 2 Context & motivation
{{
  const s = pptx.addSlide();
  addBg(s, '1. Context and motivation');
  bullets(s, [
    'Power systems now combine variable renewables, storage, demand response, and cross-border exchanges across many spatial scales.',
    'Stakeholders increasingly ask “what-if” questions rather than only one optimal schedule: extreme weather, congestion, topology changes, or new flexibility portfolios.',
    'Repeated MILP solves remain rigorous and auditable, but become expensive when thousands of scenarios must be explored quickly.',
    'The goal is not to replace optimisation, but to make large scenario exploration fast enough to support decision-making.'
  ], 0.7, 1.45, 6.0, 4.8, 18);
  metricBox(s, 7.2, 1.55, 2.3, 1.45, 'Simple scenario', '39 zones', '44,568 vars | Crit = 0.12', C.paleGreen, C.green);
  metricBox(s, 9.7, 1.55, 2.3, 1.45, 'Critical scenario', '119 zones', '139,128 vars | Crit = 0.78', C.paleRed, C.red);
  addPageImg(s, 9, 7.1, 3.25, 5.3, 3.15, true);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Introduction and Table/Figure comparing simple vs critical scenarios (manuscript pages 1–2 and 9).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 3 Literature & gap
{{
  const s = pptx.addSlide();
  addBg(s, '2. Literature snapshot and research gap');
  s.addText('Existing strengths', {{ x:0.7, y:1.45, w:3.6, h:0.25, fontSize:17, bold:true, color:C.blue }});
  bullets(s, [
    'MILP is still the reference for high-fidelity scheduling and planning.',
    'Hierarchical / multi-layer MILP captures coordination across buildings, districts, regions.',
    'GNNs can exploit topology for OPF surrogates, warm starts, and risk assessment.',
    'Energy-based models offer controlled exploration of discrete solution spaces.'
  ], 0.7, 1.8, 3.8, 4.6, 16.5);
  s.addText('What is still missing', {{ x:4.75, y:1.45, w:3.7, h:0.25, fontSize:17, bold:true, color:C.orange }});
  bullets(s, [
    'Large scenario sweeps remain solver-bound.',
    'Most surrogates do not guarantee feasible end-to-end operating plans.',
    'Prediction is easier than controlled exploration of many discrete candidates.',
    'Economic consistency and interactivity are still weak.'
  ], 4.75, 1.8, 3.8, 4.6, 16.5);
  metricBox(s, 9.0, 1.8, 3.0, 1.45, 'Research question', 'Can a hybrid MILP–GNN–EBM stack enable near-real-time exploration and remain useful as complexity increases?', 'From exact optimisation to an exploration engine', C.paleBlue, C.blue);
  s.addShape(pptx.ShapeType.roundRect, {{ x:9.0, y:3.75, w:3.0, h:2.0, rectRadius:0.08, fill:{{color:'FAFBFD'}}, line:{{color:C.line}} }});
  s.addText('Six contributions', {{ x:9.22, y:3.95, w:2.5, h:0.2, fontSize:14, bold:true, color:C.navy }});
  bullets(s, [
    'diverse scenario generator',
    'multi-layer MILP oracle',
    'graph projection',
    'topology-consistent encoder',
    'EBM + feasibility-aware decoding',
    'evaluation protocol across complexity regimes'
  ], 9.18, 4.18, 2.55, 1.4, 13.2);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Literature Review / Research Gap / Contributions sections (manuscript pages 2–4).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 4 Method overview
{{
  const s = pptx.addSlide();
  addBg(s, '3. Method overview');
  addPageImg(s, 6, 0.7, 1.45, 7.2, 5.5, true);
  s.addShape(pptx.ShapeType.roundRect, {{ x:8.35, y:1.65, w:4.25, h:4.95, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Intuition in one sentence', {{ x:8.6, y:1.95, w:3.7, h:0.22, fontSize:15, bold:true, color:C.blue }});
  s.addText('We solve a diverse training set exactly, learn a structured representation of scenarios, sample promising binary decisions quickly, then reconstruct feasible dispatch with optimisation.', {{
    x:8.6, y:2.25, w:3.6, h:1.05, fontSize:18, color:C.navy, bold:true
  }});
  bullets(s, [
    'Scenario generator creates diverse “what-if” cases.',
    'MILP oracle provides gold / silver labels.',
    'Graph Builder + HTE encode topology and time couplings.',
    'EBM + Langevin sampler propose several binary candidates.',
    'Decoder + LP worker return complete, feasible operating plans.'
  ], 8.55, 3.55, 3.7, 2.55, 15.2);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, workflow figure on manuscript page 6.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 5 Scenario generator
{{
  const s = pptx.addSlide();
  addBg(s, '4. Method block 1 — Scenario generator');
  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:1.45, w:4.25, h:4.95, rectRadius:0.08, fill:{{color:C.paleBlue}}, line:{{color:C.line}} }});
  s.addText('What goes into one scenario?', {{ x:0.95, y:1.7, w:3.6, h:0.25, fontSize:17, bold:true, color:C.navy }});
  bullets(s, [
    'network structure: regions, zones, interties, neighbouring countries',
    'asset portfolio: thermal, nuclear, hydro, wind, solar, batteries, DR',
    'economic policies: CO₂ price, caps, import/export rules',
    'technical scalers: ramps, min generation, storage E/P ratio, DR constraints',
    'exogenous drivers: weather, demand profiles, inflows'
  ], 0.95, 2.0, 3.7, 3.7, 15.5);
  s.addShape(pptx.ShapeType.roundRect, {{ x:5.2, y:1.45, w:3.2, h:2.25, rectRadius:0.08, fill:{{color:C.paleTeal}}, line:{{color:C.line}} }});
  s.addText('Diversity logic', {{ x:5.45, y:1.7, w:2.4, h:0.2, fontSize:16, bold:true, color:C.teal }});
  bullets(s, ['Latin Hypercube Sampling for key continuous drivers', 'greedy k-center selection to avoid near-duplicates', 'stratification so rare but critical cases are not drowned out'], 5.45, 2.0, 2.5, 1.4, 14.5);
  s.addShape(pptx.ShapeType.roundRect, {{ x:5.2, y:3.95, w:3.2, h:2.45, rectRadius:0.08, fill:{{color:C.paleOrange}}, line:{{color:C.line}} }});
  s.addText('Criticality index', {{ x:5.45, y:4.2, w:2.4, h:0.2, fontSize:16, bold:true, color:C.orange }});
  s.addText('Crit(s) = α · Stress(s) + (1 − α) · Hard(s)', {{ x:5.45, y:4.55, w:2.45, h:0.26, fontSize:18, bold:true, color:C.navy, align:'center' }});
  bullets(s, ['Stress = VRE variability, load, flexibility margins, spatial tension', 'Hardness = size, non-convexity, temporal and spatial couplings'], 5.4, 4.95, 2.55, 1.15, 13.8);
  addPageImg(s, 9, 8.7, 1.55, 3.9, 4.85, true);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Scenario Generator and Criticality sections (manuscript pages 5–9).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 6 MILP block
{{
  const s = pptx.addSlide();
  addBg(s, '5. Method block 2 — Multi-layer MILP oracle');
  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:1.45, w:4.0, h:5.0, rectRadius:0.08, fill:{{color:'FAFBFE'}}, line:{{color:C.line}} }});
  s.addText('Role in the pipeline', {{ x:0.95, y:1.72, w:2.4, h:0.2, fontSize:17, bold:true, color:C.red }});
  bullets(s, [
    'Solve each generated scenario with a high-fidelity multi-period unit commitment / dispatch model.',
    'Return auditable schedules: binaries (commitment, activation, modes) + continuous dispatch.',
    'Create gold labels if solved to optimality; silver labels if the solver stops at the time limit.'
  ], 0.95, 2.0, 3.4, 2.0, 15.2);
  s.addText('Why it is hard', {{ x:0.95, y:4.45, w:2.0, h:0.2, fontSize:17, bold:true, color:C.navy }});
  bullets(s, ['binary explosion', 'inter-temporal coupling', 'network and storage constraints', 'cross-border logic'], 0.95, 4.72, 3.2, 1.2, 15.2);

  s.addShape(pptx.ShapeType.roundRect, {{ x:5.0, y:1.45, w:3.15, h:5.0, rectRadius:0.08, fill:{{color:C.paleRed}}, line:{{color:C.line}} }});
  s.addText('Main variable families', {{ x:5.25, y:1.72, w:2.4, h:0.2, fontSize:17, bold:true, color:C.red }});
  bullets(s, [
    'Binary: thermal on/off and start-up, DR activation, storage modes, import/export mode',
    'Continuous: generation, curtailment, shedding, SOC, hydro levels, flows, imports/exports'
  ], 5.25, 2.0, 2.45, 1.5, 15.2);
  s.addText('Typical size range', {{ x:5.25, y:4.15, w:2.2, h:0.2, fontSize:17, bold:true, color:C.navy }});
  s.addText('~45k → ~139k variables\\n~58k → ~181k constraints', {{
    x:5.25, y:4.47, w:2.4, h:0.8, fontSize:22, bold:true, color:C.navy, align:'center'
  }});

  addPageImg(s, 13, 8.45, 1.52, 4.1, 5.0, true);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, MILP oracle and illustrative MILP outputs (manuscript pages 10–13).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 7 Graph builder + HTE
{{
  const s = pptx.addSlide();
  addBg(s, '6. Method block 3 — Graph Builder + Hierarchical Temporal Encoder');
  addPageImg(s, 15, 0.72, 1.52, 4.25, 2.35, true);
  addPageImg(s, 16, 0.72, 4.0, 4.25, 2.2, true);
  s.addShape(pptx.ShapeType.roundRect, {{ x:5.25, y:1.52, w:7.2, h:4.72, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Why graphs?', {{ x:5.55, y:1.8, w:1.7, h:0.2, fontSize:17, bold:true, color:C.teal }});
  bullets(s, [
    'Scenarios are naturally hierarchical: nation → regions → zones → assets.',
    'The number of nodes varies from one scenario to another.',
    'Transmission and weather influence are relational; time couplings matter too.'
  ], 5.55, 2.08, 3.15, 1.5, 15.2);
  s.addText('Why a Hierarchical Temporal Encoder?', {{ x:5.55, y:3.65, w:3.2, h:0.2, fontSize:17, bold:true, color:C.purple }});
  bullets(s, [
    'Bottom-up spatial aggregation collects local detail.',
    'A small temporal Transformer models the 24-hour sequence at system level.',
    'Top-down decoding gives each asset / zone both local and global context.',
    'Training is self-supervised: no MILP labels needed at this stage.'
  ], 5.55, 3.95, 6.0, 1.95, 15.2);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, graph diagrams and HTE description (manuscript pages 14–17).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 8 EBM + Langevin
{{
  const s = pptx.addSlide();
  addBg(s, '7. Method block 4 — EBM + Langevin sampler');
  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:1.5, w:5.0, h:4.85, rectRadius:0.08, fill:{{color:C.paleOrange}}, line:{{color:C.line}} }});
  s.addText('Intuition', {{ x:0.98, y:1.8, w:1.2, h:0.2, fontSize:17, bold:true, color:C.orange }});
  s.addText('Instead of predicting one schedule, learn an “energy landscape” over binary decisions and sample several promising candidates.', {{
    x:0.98, y:2.15, w:4.15, h:0.85, fontSize:21, bold:true, color:C.navy
  }});
  bullets(s, [
    'Good schedules should have low energy.',
    'Multiple low-energy basins allow multimodality.',
    'The sampler explores the space and returns several candidates.'
  ], 0.98, 3.3, 4.0, 1.2, 15.5);

  s.addShape(pptx.ShapeType.roundRect, {{ x:6.0, y:1.5, w:6.4, h:2.2, rectRadius:0.08, fill:{{color:C.paleBlue}}, line:{{color:C.line}} }});
  s.addText('Langevin in logit space', {{ x:6.25, y:1.8, w:2.3, h:0.2, fontSize:17, bold:true, color:C.blue }});
  s.addText('z(k+1) = z(k) − η ∇E + noise', {{
    x:6.25, y:2.18, w:5.7, h:0.34, fontSize:24, bold:true, color:C.navy, align:'center'
  }});
  bullets(s, ['work in continuous logits, then convert to probabilities / binaries', 'start noisy for exploration, then cool down for refinement'], 6.25, 2.7, 5.5, 0.75, 15.0);

  s.addShape(pptx.ShapeType.roundRect, {{ x:6.0, y:4.02, w:6.4, h:2.33, rectRadius:0.08, fill:{{color:C.paleGreen}}, line:{{color:C.line}} }});
  s.addText('Two-step training', {{ x:6.25, y:4.3, w:2.0, h:0.2, fontSize:17, bold:true, color:C.green }});
  bullets(s, [
    'Gold pre-training: push down MILP-optimal solutions, push up bad samples.',
    'Silver fine-tuning: use LP-based pairwise preferences to align energy with economic cost.'
  ], 6.25, 4.62, 5.6, 1.2, 15.2);

  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Energy-Based Learning section (manuscript pages 17–21).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 9 Evaluation protocol
{{
  const s = pptx.addSlide();
  addBg(s, '8. Method block 5 — Evaluation protocol');
  metricBox(s, 0.7, 1.55, 2.5, 1.35, 'Training set', '5,000', 'independent synthetic scenarios', C.paleBlue, C.blue);
  metricBox(s, 3.45, 1.55, 2.5, 1.35, 'Test set', '300', '100 low | 100 medium | 100 high', C.paleGreen, C.green);
  metricBox(s, 6.2, 1.55, 2.5, 1.35, 'Inference', '5 chains', 'best-of-5 candidate selection', C.paleOrange, C.orange);
  metricBox(s, 8.95, 1.55, 3.0, 1.35, 'Metrics', 'gap | speedup | slack | LP stage', 'all measured against the MILP oracle', C.paleTeal, C.teal);

  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:3.25, w:5.2, h:3.05, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Evaluation families', {{ x:0.98, y:3.55, w:2.1, h:0.2, fontSize:17, bold:true, color:C.navy }});
  bullets(s, [
    'Low criticality: 4–20 zones, moderate demand, MILP usually < 2 s.',
    'Medium criticality: 30–80 zones, elevated demand, MILP 1–120 s.',
    'High criticality: 80–160 zones, stressed demand, MILP can reach the time limit.'
  ], 0.98, 3.88, 4.3, 1.8, 15.2);

  s.addShape(pptx.ShapeType.roundRect, {{ x:6.2, y:3.25, w:6.15, h:3.05, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('What “success” means here', {{ x:6.48, y:3.55, w:2.8, h:0.2, fontSize:17, bold:true, color:C.blue }});
  bullets(s, [
    'For easy cases, the exact MILP should remain the best tool.',
    'For hard cases, we want comparable quality, guaranteed feasibility, and bounded runtime.',
    'The scientific test is whether the hybrid stack becomes more useful as complexity grows.'
  ], 6.48, 3.88, 5.15, 1.8, 15.2);

  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Experimental setup and evaluation protocol (manuscript pages 24–25).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 10 Results overview families
{{
  const s = pptx.addSlide();
  addBg(s, '9. Results — Low / medium / high criticality families');
  s.addText('Headline', {{ x:0.7, y:1.42, w:1.4, h:0.2, fontSize:15, bold:true, color:C.blue }});
  s.addText('The hybrid pipeline becomes relatively more attractive as combinatorial hardness increases.', {{
    x:0.7, y:1.72, w:8.2, h:0.45, fontSize:22, bold:true, color:C.navy
  }});

  const cols = [
    ['Low', 'median gap = 12.6%', 'median speedup = 0.12×', 'hard-fix = 90%', C.green, C.paleGreen],
    ['Medium', 'median gap = 18.5%', 'median speedup = 0.46×', 'hard-fix = 65%', C.orange, C.paleOrange],
    ['High', 'median gap = −0.04%', 'median speedup = 0.88×', 'hard-fix = 40%', C.red, C.paleRed],
  ];
  [0.7, 4.15, 7.6].forEach((x, i) => {{
    const c = cols[i];
    s.addShape(pptx.ShapeType.roundRect, {{ x, y:2.55, w:2.9, h:2.6, rectRadius:0.08, fill:{{color:c[5]}}, line:{{color:C.line}} }});
    s.addText(c[0], {{ x:x+0.18, y:2.78, w:2.3, h:0.24, fontSize:20, bold:true, color:c[4], align:'center' }});
    s.addText(c[1], {{ x:x+0.18, y:3.35, w:2.45, h:0.22, fontSize:18, bold:true, color:C.navy, align:'center' }});
    s.addText(c[2], {{ x:x+0.18, y:3.85, w:2.45, h:0.22, fontSize:18, bold:true, color:C.navy, align:'center' }});
    s.addText(c[3], {{ x:x+0.18, y:4.35, w:2.45, h:0.22, fontSize:16, color:C.ink, align:'center' }});
  }});
  s.addShape(pptx.ShapeType.roundRect, {{ x:10.65, y:2.55, w:1.95, h:2.6, rectRadius:0.08, fill:{{color:'FAFBFD'}}, line:{{color:C.line}} }});
  s.addText('Feasibility', {{ x:10.83, y:2.82, w:1.55, h:0.2, fontSize:16, bold:true, color:C.blue, align:'center' }});
  s.addText('100%', {{ x:10.83, y:3.28, w:1.55, h:0.4, fontSize:28, bold:true, color:C.navy, align:'center' }});
  s.addText('near-zero slack\\nacross all 300 test scenarios', {{ x:10.85, y:3.95, w:1.5, h:0.55, fontSize:13.5, color:C.gray, align:'center' }});

  addPageImg(s, 26, 0.7, 5.35, 12.0, 1.4, true);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, main results and Table 4 on manuscript pages 25–27, plus Figure 6 on page 26.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 11 Results crossover and interpretation
{{
  const s = pptx.addSlide();
  addBg(s, '10. Results — Where does the pipeline start to pay off?');
  addPageImg(s, 29, 0.7, 1.5, 6.6, 4.7, true);
  s.addShape(pptx.ShapeType.roundRect, {{ x:7.6, y:1.5, w:4.8, h:4.7, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Three regimes', {{ x:7.9, y:1.82, w:2.0, h:0.2, fontSize:17, bold:true, color:C.blue }});
  bullets(s, [
    'MILP < 10 s: the exact solver is faster and better.',
    'MILP ≈ 10–100 s: transition regime, mixed results.',
    'MILP > 100 s: the hybrid pipeline is consistently faster.'
  ], 7.9, 2.15, 3.8, 1.55, 15.2);
  s.addText('Interpretation', {{ x:7.9, y:4.08, w:1.8, h:0.2, fontSize:17, bold:true, color:C.orange }});
  bullets(s, [
    'The neural overhead is almost constant (~4 s).',
    'The LP worker becomes the main bottleneck, not the encoder or EBM.',
    'This supports a criticality-aware routing strategy: exact MILP for easy cases, hybrid pipeline for hard ones.'
  ], 7.9, 4.38, 4.0, 1.5, 15.2);
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, timing and crossover figures on manuscript pages 28–30.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 12 Persona results
{{
  const s = pptx.addSlide();
  addBg(s, '11. Results — Persona-based evaluation');
  s.addText('Why personas?', {{ x:0.7, y:1.45, w:1.7, h:0.2, fontSize:16, bold:true, color:C.blue }});
  s.addText('Different stakeholders stress different aspects of the same power system.', {{
    x:0.7, y:1.75, w:6.0, h:0.35, fontSize:20, bold:true, color:C.navy
  }});
  addPageImg(s, 40, 0.7, 2.25, 5.8, 4.0, true);

  const px = 6.9;
  s.addShape(pptx.ShapeType.roundRect, {{ x:px, y:1.55, w:5.45, h:4.9, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Three personas, three behaviors', {{ x:px+0.25, y:1.83, w:3.2, h:0.2, fontSize:17, bold:true, color:C.teal }});
  bullets(s, [
    'VRE / Battery developer: MILP remains fast; the pipeline is slower but fairly accurate.',
    'Network operator: strongest case for the pipeline; many time-limited MILP runs and clear speedup.',
    'Mathematician: hardness-dominated cases expose generalisation weaknesses, but the MILP also struggles.'
  ], px+0.25, 2.15, 4.65, 1.9, 15.0);
  s.addText('Median gap / mean speedup', {{ x:px+0.25, y:4.58, w:2.4, h:0.2, fontSize:14, bold:true, color:C.gray }});
  s.addText('VRE/Battery: 4.7% / 0.2×\\nNetwork operator: 30.0% / 5.5×\\nMathematician: −100.6% / 2.6×', {{
    x:px+0.25, y:4.86, w:4.75, h:1.0, fontSize:18, bold:true, color:C.navy
  }});
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, persona design and persona results on manuscript pages 38–40 (Table 12, Figures 19–20).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 13 OOD
{{
  const s = pptx.addSlide();
  addBg(s, '12. Results — Out-of-distribution extreme criticality');
  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:1.5, w:4.7, h:4.9, rectRadius:0.08, fill:{{color:C.paleRed}}, line:{{color:C.line}} }});
  s.addText('Setup', {{ x:0.98, y:1.8, w:0.9, h:0.2, fontSize:17, bold:true, color:C.red }});
  bullets(s, [
    '100 scenarios deliberately pushed beyond the 95th percentile of the training distribution.',
    'All of them hit the MILP time limit.',
    'The test asks whether the pipeline still provides feasible, bounded-time answers under extreme stress.'
  ], 0.98, 2.12, 4.0, 1.75, 15.2);
  metricBox(s, 0.95, 4.55, 1.95, 1.45, 'Feasibility', '100%', 'near-zero slack', 'FFFFFF', C.green);
  metricBox(s, 3.05, 4.55, 1.95, 1.45, 'Median speedup', '5.4×', 'pipeline faster in all cases', 'FFFFFF', C.blue);

  s.addShape(pptx.ShapeType.roundRect, {{ x:5.75, y:1.5, w:6.6, h:4.9, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
  s.addText('Take-away', {{ x:6.0, y:1.8, w:1.5, h:0.2, fontSize:17, bold:true, color:C.blue }});
  s.addText('The pipeline preserves structure and throughput, but quality deteriorates well beyond the training distribution.', {{
    x:6.0, y:2.12, w:5.7, h:0.72, fontSize:20, bold:true, color:C.navy
  }});
  bullets(s, [
    'Median cost gap rises to 119.6%.',
    'Yet hard-fix remains 48%, suggesting partial transfer of learned patterns.',
    'This motivates richer extreme-stress data and stronger repair / decoder mechanisms.'
  ], 6.0, 3.25, 5.5, 1.55, 15.2);
  s.addText('Reference comparison', {{ x:6.0, y:5.18, w:2.0, h:0.2, fontSize:14, bold:true, color:C.gray }});
  s.addText('Extreme OOD vs high-criticality reference\\nmedian gap: 119.6% vs −0.04%\\nmedian speedup: 5.4× vs 0.88×', {{
    x:6.0, y:5.48, w:5.3, h:0.65, fontSize:17, color:C.navy, bold:true
  }});
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, extreme OOD assessment and Table 13 on manuscript pages 41–42.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 14 Conclusion
{{
  const s = pptx.addSlide();
  addBg(s, '13. Conclusion');
  s.addText('What we showed', {{ x:0.7, y:1.45, w:2.0, h:0.2, fontSize:17, bold:true, color:C.blue }});
  bullets(s, [
    'A hybrid MILP–GNN–EBM stack can turn exact optimisation into a structured scenario-exploration engine.',
    'Feasibility is preserved across all evaluation scenarios thanks to decoder + LP reconstruction.',
    'The relative value of the pipeline increases with problem hardness: easy scenarios still belong to MILP, hard scenarios are the sweet spot.',
    'Persona and OOD tests reveal where the method is already useful and where it still needs work.'
  ], 0.7, 1.78, 6.4, 3.6, 17.0);
  metricBox(s, 7.55, 1.7, 2.25, 1.45, 'Key result', 'ρ = +0.224', 'criticality vs speedup', C.paleBlue, C.blue);
  metricBox(s, 10.0, 1.7, 2.25, 1.45, 'Key result', '100%', 'feasible solutions on the 300-scenario benchmark', C.paleGreen, C.green);
  s.addShape(pptx.ShapeType.roundRect, {{ x:7.5, y:3.55, w:4.8, h:2.05, rectRadius:0.08, fill:{{color:C.paleOrange}}, line:{{color:C.line}} }});
  s.addText('Strategic message for PowDev', {{ x:7.78, y:3.85, w:3.2, h:0.2, fontSize:17, bold:true, color:C.orange }});
  s.addText('The methodology is promising not because it beats MILP everywhere, but because it becomes useful precisely where scenario exploration becomes operationally difficult.', {{
    x:7.78, y:4.18, w:3.95, h:1.0, fontSize:19, bold:true, color:C.navy
  }});
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, conclusion and main results sections (manuscript pages 25–45).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 15 Limitations
{{
  const s = pptx.addSlide();
  addBg(s, '14. Current limitations');
  const bx = [0.7, 4.15, 7.6, 10.2];
  const data = [
    ['L1', 'Generalisation to large topologies', 'High-criticality scenarios still trigger many full-soft LP repairs.'],
    ['L2', 'Greedy decoder sub-optimality', 'The decoder guarantees feasibility but can degrade cost quality.'],
    ['L3', 'Fixed overhead', 'The neural stack costs ~4 s even when the MILP solves in <2 s.'],
    ['L4', 'Deterministic single-day setting', 'No rolling horizon, stochasticity, or N−1 security yet.']
  ];
  bx.forEach((x,i)=>{{
    const d=data[i];
    s.addShape(pptx.ShapeType.roundRect, {{ x, y:1.8, w:2.4, h:2.2, rectRadius:0.08, fill:{{color:'FBFCFE'}}, line:{{color:C.line}} }});
    chip(s, d[0], x+0.12, 1.95, 0.46, C.red);
    s.addText(d[1], {{ x:x+0.18, y:2.4, w:2.0, h:0.42, fontSize:16, bold:true, color:C.navy, align:'center' }});
    s.addText(d[2], {{ x:x+0.18, y:3.0, w:2.0, h:0.7, fontSize:13.2, color:C.ink, align:'center' }});
  }});
  s.addShape(pptx.ShapeType.roundRect, {{ x:1.15, y:4.45, w:11.0, h:1.35, rectRadius:0.08, fill:{{color:C.paleRed}}, line:{{color:C.line}} }});
  s.addText('Bottom line', {{ x:1.45, y:4.76, w:1.3, h:0.2, fontSize:16, bold:true, color:C.red }});
  s.addText('This is already a strong methodological proof of concept, but not yet a production-grade decision engine for real-system operations.', {{
    x:2.4, y:4.68, w:8.95, h:0.45, fontSize:21, bold:true, color:C.navy
  }});
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Limitations section (manuscript pages 43–44).']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 16 Next steps / climate application
{{
  const s = pptx.addSlide();
  addBg(s, '15. Next steps for PowDev');
  s.addShape(pptx.ShapeType.roundRect, {{ x:0.7, y:1.55, w:5.75, h:4.85, rectRadius:0.08, fill:{{color:C.paleTeal}}, line:{{color:C.line}} }});
  s.addText('Methodological continuation', {{ x:0.98, y:1.85, w:2.8, h:0.2, fontSize:17, bold:true, color:C.teal }});
  bullets(s, [
    'Scale training toward larger and rarer high-criticality scenarios.',
    'Replace or augment the greedy decoder with a learned feasibility-aware module.',
    'Warm-start the LP worker more intelligently.',
    'Move toward routing policies: exact MILP for easy cases, hybrid pipeline for hard ones.'
  ], 0.98, 2.18, 4.9, 2.0, 15.5);

  s.addShape(pptx.ShapeType.roundRect, {{ x:6.75, y:1.55, w:5.55, h:4.85, rectRadius:0.08, fill:{{color:C.paleBlue}}, line:{{color:C.line}} }});
  s.addText('Application paper idea: extreme climate scenarios', {{ x:7.02, y:1.85, w:4.0, h:0.2, fontSize:17, bold:true, color:C.blue }});
  bullets(s, [
    'Work with Anastasia: build climate-conditioned stress scenarios (heatwaves, Dunkelflaute, correlated demand and inflow shocks).',
    'Use the current hybrid stack as a fast exploration engine for large batches of climate extremes.',
    'Translate technical outcomes into system-level insights for resilience, flexibility procurement, and planning.'
  ], 7.02, 2.18, 4.65, 2.1, 15.5);
  s.addShape(pptx.ShapeType.roundRect, {{ x:1.5, y:5.0, w:10.0, h:0.9, rectRadius:0.08, fill:{{color:C.paleOrange}}, line:{{color:C.line}} }});
  s.addText('Proposed transition: from a methodology paper to an application paper on climate-stressed power-system flexibility.', {{
    x:1.8, y:5.24, w:9.45, h:0.35, fontSize:22, bold:true, color:C.navy, align:'center'
  }});
  addFooter(s);
  addNotes(s, ['User-provided manuscript PDF, Research avenues section (manuscript pages 44–45). Climate-application wording added from the user request.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// 17 Appendix divider
{{
  const s = pptx.addSlide();
  s.background = {{ color: C.navy }};
  s.addText('Appendix', {{ x:0.8, y:2.0, w:3.0, h:0.7, fontSize:30, bold:true, color:C.white }});
  s.addText('Additional equations, figures, tables, and diagnostic material', {{ x:0.8, y:2.85, w:6.0, h:0.35, fontSize:19, color:'D8E5FF' }});
  s.addText('Useful for technical discussion after the main talk', {{ x:0.8, y:3.35, w:5.6, h:0.25, fontSize:15, color:'C6D3EA' }});
  addNotes(s, ['Appendix divider.']);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}}

// Appendix slides
const appendix = [
  ['A1. Full pipeline figure', 6, 'Pipeline from scenario generation to benchmarking'],
  ['A2. Simple vs critical scenarios', 9, 'Illustrative scenario comparison'],
  ['A3. MILP oracle outputs for a critical case', 13, 'Dispatch, storage, curtailment, DR, binaries'],
  ['A4. Spatial graph schema', 15, 'Spatial heterogeneous graph'],
  ['A5. Temporal graph schema', 16, 'Supra-graph with temporal edges'],
  ['A6. Central result figure', 26, 'Solve time, speedup, and cost-gap envelopes'],
  ['A7. LP stage distribution and timing', 28, 'Repair stages and pipeline timing'],
  ['A8. Crossover / Pareto view', 30, 'Quality–speed trade-off'],
  ['A9. Time-limited hard scenarios', 31, 'Hard subset where MILP hits time limit'],
  ['A10. Persona fingerprints and dashboard', 40, 'Persona results'],
  ['A11. Extreme OOD table and discussion', 42, 'OOD robustness'],
  ['A12. Decoder ablation (quality vs speed)', 63, 'No-decoder appendix'],
  ['A13. GNN dispatch ablation', 65, 'Fast but less exact screening alternative'],
];
appendix.forEach((a) => {{
  const s = pptx.addSlide();
  addBg(s, a[0], a[2]);
  addPageImg(s, a[1], 0.65, 1.4, 12.05, 5.85, true);
  addFooter(s);
  addNotes(s, [`User-provided manuscript PDF, manuscript page ${'{'}a[1]{'}'}.`]);
  warnIfSlideHasOverlaps(s, pptx); warnIfSlideElementsOutOfBounds(s, pptx);
}});

pptx.writeFile({{ fileName: '{(work/"powdev_milp_gnn_ebm_presentation.pptx").as_posix()}' }});
"""))

subprocess.run(["node", str(js)], check=True)

# 3) Also create a concise Beamer .tex source matching the structure requested by the user
tex = work / "powdev_milp_gnn_ebm_beamer.tex"
tex.write_text(textwrap.dedent(r"""
\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\title{Hybrid MILP--GNN--EBM for Large-Scale Scenario Exploration}
\subtitle{PowDev project presentation}
\author{Based on the manuscript by Th\'eotime Coudray and St\'ephane Goutte}
\date{}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}{Context and motivation}
\begin{itemize}
\item Power systems combine VRE, storage, demand response and cross-border exchanges across many scales.
\item Stakeholders increasingly ask ``what-if'' questions under extreme weather, congestion and topology changes.
\item MILP remains rigorous and auditable, but repeated solves become expensive when scenario sets are large.
\item Goal: turn exact optimisation into a fast scenario-exploration engine.
\end{itemize}
\end{frame}

\begin{frame}{Literature and research gap}
\begin{itemize}
\item MILP remains the reference for high-fidelity scheduling and planning.
\item GNNs exploit topology well, but feasibility is often fragile.
\item Energy-based models provide controlled exploration of discrete solution spaces.
\item Gap: little support for \textbf{interactive}, \textbf{feasible}, \textbf{economically consistent} large-scale scenario exploration.
\end{itemize}
\end{frame}

\begin{frame}{Research question}
Can a hybrid \textbf{MILP--GNN--EBM} methodology enable near-real-time exploration of multi-scale flexibility scenarios, and remain effective as scenario complexity increases?
\end{frame}

\begin{frame}{Method overview}
\begin{enumerate}
\item Generate diverse scenarios and compute criticality.
\item Solve a subset exactly with a multi-layer MILP oracle.
\item Convert each scenario into a heterogeneous temporal graph.
\item Learn compact scenario embeddings with a Hierarchical Temporal Encoder.
\item Learn an energy landscape over binary decisions with an EBM.
\item Sample candidates with Langevin dynamics.
\item Repair and reconstruct feasible dispatch with decoder + LP worker.
\end{enumerate}
\end{frame}

\begin{frame}{Scenario generator}
\begin{itemize}
\item Samples network structure, assets, policies, technical scalers and exogenous drivers.
\item Uses Latin Hypercube Sampling, greedy k-center selection and stratification.
\item Defines an ex-ante criticality index:
\[
\mathrm{Crit}(s) = \alpha\,\mathrm{Stress}(s) + (1-\alpha)\,\mathrm{Hard}(s)
\]
\end{itemize}
\end{frame}

\begin{frame}{Multi-layer MILP oracle}
\begin{itemize}
\item Multi-period unit commitment and economic dispatch.
\item Binary variables: commitment, start-up, DR activation, storage modes.
\item Continuous variables: generation, flows, storage levels, curtailment, shedding.
\item Produces gold and silver labels for downstream learning.
\end{itemize}
\end{frame}

\begin{frame}{Graph Builder + Hierarchical Temporal Encoder}
\begin{itemize}
\item Build a heterogeneous graph: nation, regions, zones, assets, weather.
\item Add temporal edges for SOC, ramping and cooldown constraints.
\item HTE uses bottom-up aggregation, a temporal Transformer, then top-down decoding.
\item Training is self-supervised.
\end{itemize}
\end{frame}

\begin{frame}{EBM + Langevin sampler}
\begin{itemize}
\item The EBM assigns low energy to good binary schedules.
\item Langevin sampling explores several candidate schedules in logit space.
\item Training: contrastive divergence first, then LP-guided preference fine-tuning.
\item This supports multi-candidate exploration rather than one point prediction.
\end{itemize}
\end{frame}

\begin{frame}{Evaluation protocol}
\begin{itemize}
\item 300 out-of-sample test scenarios: low, medium, high criticality.
\item Metrics: cost gap, speedup, slack, LP repair stage.
\item Best-of-5 sampling policy at inference time.
\end{itemize}
\end{frame}

\begin{frame}{Results on low / medium / high families}
\begin{itemize}
\item Low: MILP clearly dominates (median gap 12.6\%, median speedup 0.12$\times$).
\item Medium: transition regime (median gap 18.5\%, median speedup 0.46$\times$).
\item High: the pipeline becomes attractive (median gap $-0.04\%$, median speedup 0.88$\times$).
\item Feasibility is 100\% across all 300 evaluation scenarios.
\end{itemize}
\end{frame}

\begin{frame}{Main empirical message}
\begin{itemize}
\item The relative advantage of the hybrid stack increases with complexity.
\item Easy scenarios still belong to exact MILP.
\item Hard scenarios are the sweet spot for the pipeline.
\end{itemize}
\end{frame}

\begin{frame}{Persona results}
\begin{itemize}
\item VRE / Battery developer: low speedup but relatively good quality.
\item Network operator: strongest case for the pipeline.
\item Mathematician: hardness-dominated cases expose generalisation limits.
\end{itemize}
\end{frame}

\begin{frame}{OOD extreme criticality}
\begin{itemize}
\item All extreme OOD scenarios hit the MILP time limit.
\item The pipeline remains feasible and faster (median speedup 5.4$\times$).
\item Quality deteriorates: median cost gap rises to 119.6\%.
\end{itemize}
\end{frame}

\begin{frame}{Conclusion}
\begin{itemize}
\item Exact optimisation can be used to train a fast exploration engine.
\item The stack is promising because it becomes more useful as hardness increases.
\item Feasibility is structurally preserved by decoder + LP reconstruction.
\end{itemize}
\end{frame}

\begin{frame}{Limitations}
\begin{itemize}
\item Generalisation to large topologies.
\item Greedy decoder sub-optimality.
\item Fixed overhead on easy cases.
\item Deterministic single-horizon setting.
\end{itemize}
\end{frame}

\begin{frame}{Next steps}
\begin{itemize}
\item Scale the training distribution toward harder and rarer cases.
\item Learn a better feasibility decoder / warm start.
\item Build an application paper on extreme climate scenarios with Anastasia.
\end{itemize}
\end{frame}

\end{document}
"""))

print("Created:")
print(work / "powdev_milp_gnn_ebm_presentation.pptx")
print(tex)
