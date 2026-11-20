"""Render manuscript figures from saved CSVs; never fit models or run solvers."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'research/paper1_weights/runs/weights_v2_variants'
OUT = Path(__file__).resolve().parent / 'results'
POLICIES = ['equal4', 'cost_priority', 'co2_priority', 'time_priority', 'service4_co2']
LABELS = ['Equal weights', 'Cost priority', r'CO$_2$ priority', 'Time priority', 'Service priority']
DEADLINES = [5, 10, 30, 60, 120]
COMPONENTS = ['A'+str(i) for i in range(1, 13)] + ['B'+str(i) for i in range(1, 11)]
EXPECTED = {
 'weights_by_deadline.csv': '6ea1474cb426528c36c7813713632e5ad8b001358085f2dd828ad52d17fc120e',
 'paired_predictive_gains.csv': '79fe90e67cf2a1fd58ed808aa13d6b73eab497998c81fe38ed120363b5f3a89e',
}

def read(name):
    p = RUN / 'report' / name
    if hashlib.sha256(p.read_bytes()).hexdigest() != EXPECTED[name]:
        raise ValueError('Source changed: '+name)
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

weights = read('weights_by_deadline.csv')
gains = read('paired_predictive_gains.csv')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7.5, 'axes.titlesize':8,
    'axes.labelsize':7.5, 'xtick.labelsize':7, 'ytick.labelsize':7,
    'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':0.5, 'savefig.facecolor':'white'})
OUT.mkdir(exist_ok=True)
outputs = []
def save(fig, name):
    pdf = OUT / (name+'.pdf')
    fig.savefig(pdf, metadata={'Title':name, 'Creator':'Saved-result visualization; no statistical refit'})
    fig.savefig(OUT / (name+'.png'), dpi=200)
    outputs.append(pdf.name)
    plt.close(fig)

def five_panels(figsize):
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5])]
    return fig, axes

# Absolute paired differences use the stored point estimates and confidence limits.
fig, axes = five_panels((6.30, 4.15))
for p, label, ax in zip(POLICIES, LABELS, axes):
    rows = [next(r for r in gains if r['policy']==p and r['deadline']==str(b)
                 and r['baseline']=='common_positive' and r['candidate']=='separate_deadlines') for b in DEADLINES]
    y = np.array([float(r['gain_mse'])*1000 for r in rows])
    lo = np.array([float(r['lower'])*1000 for r in rows])
    hi = np.array([float(r['upper'])*1000 for r in rows])
    ax.errorbar(range(5), y, yerr=[y-lo,hi-y], fmt='o', ms=3.5, capsize=2,
                lw=1, color='#155F83', zorder=3)
    ax.axhline(0, color='#4B5563', lw=0.7)
    ax.grid(axis='y', color='#E2E8F0', lw=0.5)
    ax.set_xticks(range(5), DEADLINES)
    ax.set_ylim(-1, 44)
    ax.set_title(label, loc='left', pad=8)
fig.supylabel('Reduction in held-out MSE (x 1,000)', x=0.015, fontsize=8)
fig.supxlabel('Independent execution deadline (s)', y=0.015, fontsize=8)
fig.subplots_adjust(left=.11, right=.985, bottom=.13, top=.94, hspace=.40, wspace=.46)
save(fig, 'weights_variants_predictive_gain')

# Show all 22 components: constant features are unavailable, not learned zeroes.
fig, axes = five_panels((6.30, 7.10))
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('#C6CBD1')
norm = Normalize(vmin=0, vmax=65)
for p, label, ax in zip(POLICIES, LABELS, axes):
    data = np.empty((22,5))
    for i,c in enumerate(COMPONENTS):
        for j,b in enumerate(DEADLINES):
            r = next(r for r in weights if r['policy']==p and r['deadline']==str(b) and r['component'].split('_')[0]==c)
            data[i,j] = np.nan if r['status'].startswith('constant') else 100*float(r['applied_index_weight'])
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    ax.set_title(label, loc='left', pad=7)
    ax.set_xticks(range(5), DEADLINES)
    ax.tick_params(axis='x', labelbottom=True, length=0, pad=4)
    ax.set_yticks(range(22), COMPONENTS)
    ax.tick_params(axis='y', labelleft=True, length=0, pad=3, labelsize=6.5)
    ax.axhline(11.5, color='#64748B', lw=.8)
    ax.set_xticks(np.arange(-.5,5,1), minor=True)
    ax.set_yticks(np.arange(-.5,22,1), minor=True)
    ax.grid(which='minor', color='white', lw=.25)
    ax.tick_params(which='minor', length=0)
    for i in range(22):
        for j in range(5):
            v = data[i,j]
            if np.isfinite(v) and v >= 5:
                ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=5.5,
                        color='white' if v>35 else '#17324D')
    for sp in ax.spines.values(): sp.set_visible(False)
fig.subplots_adjust(left=.065, right=.985, bottom=.168, top=.955, hspace=.21, wspace=.45)
cax = fig.add_axes([.22,.082,.60,.015])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0,10,20,30,40,50,60])
cb.set_label('Normalized component weight (%)', fontsize=8)
fig.text(.5,.130,'Independent execution deadline (s)',ha='center',fontsize=8)
fig.text(.5,.012,'Grey: constant, weight not identifiable. Cell labels: weights of at least 5%, rounded.',
         ha='center',fontsize=6.5,color='#475569')
save(fig, 'weights_variants_deadline_heatmap')

# Pointwise uncertainty for the largest point estimate at each deadline.
fig, axes = five_panels((6.30,4.50))
for p,label,ax in zip(POLICIES,LABELS,axes):
    for i,b in enumerate(DEADLINES):
        r=max((r for r in weights if r['policy']==p and r['deadline']==str(b)),
              key=lambda r:float(r['applied_index_weight']))
        v,lo,hi=[100*float(r[k]) for k in ['applied_index_weight','weight_lower','weight_upper']]
        ax.hlines(i,lo,hi,color='#155F83',lw=1.25)
        ax.plot(v,i,'o',ms=3.5,color='#155F83')
        ax.text(hi+2,i,r['component'].split('_')[0],va='center',fontsize=7,color='#17324D')
    ax.set_yticks(range(5),[str(b)+' s' for b in DEADLINES])
    ax.set_ylim(4.6,-.6)
    ax.set_xlim(0,95)
    ax.set_xticks([0,20,40,60,80])
    ax.set_title(label,loc='left',pad=8)
    ax.grid(axis='x',color='#E2E8F0',lw=.5)
fig.supxlabel('Largest normalized weight (%) and pointwise 95% interval', y=.025,fontsize=8)
fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.94,hspace=.43,wspace=.60)
save(fig, 'weights_variants_leading_intervals')

provenance = {'source_run':'research/paper1_weights/runs/weights_v2_variants',
    'manifest_sha256': '3311d30a960cdb904af5a2d785d3667148294b32a83f155d7c2ffc8f3fdfdb45',
    'reported_profiles':POLICIES, 'source_sha256':EXPECTED, 'operation':'Rendering saved estimates and intervals only; no refitting, resampling, or solver execution',
    'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'figure_sha256':{p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in outputs}}
(OUT/'weights_figures_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figures':outputs,'sources_checked':len(EXPECTED)},indent=2))
