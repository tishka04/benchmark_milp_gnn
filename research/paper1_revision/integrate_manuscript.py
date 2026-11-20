"""One-time structural manuscript revision; generated results remain separate."""
from .data import ROOT,RUN

def run():
    path=ROOT/'research/three_papers/manuscripts/paper1_criticality.tex'
    text=path.read_text(encoding='utf-8')
    if r'\input{paper1/results/revision_results}' in text:
        return
    (RUN/'manuscript_before_structural_edit.tex').write_text(text,encoding='utf-8')
    abstract=r'''Energy-system optimization must deliver useful dispatches within operational
deadlines. We examine a 22-component criticality indicator on 600 constructed
scenarios, separating generator parameters, realized MILP data and descriptive
proxies. Nested scenario-grouped validation evaluates economic and joint
economic--service success under five independent deadlines from 5 to 120 seconds.
The original scalar adds information beyond size in seven of ten historical
endpoint--deadline comparisons, while separate components improve prediction.
For five operational preferences, a new intermediate model separates a common
nonnegative weight direction from deadline-specific amplitudes. Rescaling the
common direction reduces mean squared error by 11.9--40.0\%; separate directions
reduce it by another 24.3--36.1\%. Twenty-three of 25 direction contrasts retain
positive lower bounds in simultaneous intervals over the 50 primary contrasts.
However, leaving a complete family out increases the separate model's pooled
MSE by factors of 2.72--3.63. An audit identifies an almost duplicated demand
proxy, and reduced representations retain similar predictive performance.
Within held-out portfolios, adaptive time allocation improves observed joint
success over uniform allocation: 415 versus 341 successes at 30 seconds reserved
per scenario on average. A three-component allocation performs similarly,
showing that decision utility need not require the richest predictor.
These development analyses support deadline- and preference-conditioned
descriptions within the sampled regimes, while exposing limits of transfer.
The prepared independent physical campaign has not been executed.'''
    start=text.index('\\begin{abstract}')+len('\\begin{abstract}')
    stop=text.index('\\end{abstract}',start);text=text[:start]+'\n'+abstract+'\n'+text[stop:]
    # Move exhaustive ablations and weight tables to the supplement, preserving labels.
    a=text.index('\\subsection{Component contributions without family conditioning}')
    b=text.index('\\subsection{Discrimination and calibration}',a)
    ablations=text[a:b];text=text[:a]+text[b:]
    a=text.index('\\subsection{Learned weights under five operational preferences}')
    b=text.index('\\section{Discussion}',a)
    weights=text[a:b];text=text[:a]+r'\input{paper1/results/revision_results}'+'\n\n'+text[b:]
    old=r'''The consistent predictive improvement establishes that the tested common
coefficient vector is restrictive on this cohort. It does not by itself
exclude every common index: separate models change both relative weights
and total coefficient amplitude. A common direction with deadline-specific
scales has not been tested. The fitted vectors are therefore conditional
predictive candidates for declared preferences and deadlines, rather than
a universal replacement for the historical criticality formula.'''
    weights=weights.replace(old,r'''These earlier contrasts vary scale and direction together.
The revision's intermediate model in Section~\ref{sec:revision_scale} now
separates those freedoms. Individual weight tables remain conditional
descriptions, not universal physical rankings.''')
    appendix=r'\appendix'+'\n'+r'\input{paper1/technical_appendix}'+'\n'+r'\section{Historical component and interaction ablations}'+'\n'+ablations+'\n'+r'\section{Historical weight estimates and conditional intervals}'+'\n'+weights
    text=text.replace(r'\bibliographystyle{unsrtnat}',appendix+'\n'+r'\bibliographystyle{unsrtnat}')
    old=r'''untested intermediate model with shared weights and deadline-specific
slopes could explain part of the gain.'''
    text=text.replace(old,r'''intermediate model with shared weights and deadline-specific
slopes explains part, but not all, of the gain: the remaining pooled
reduction is 24.3--36.1\%, with 23 of 25 deadline-specific contrasts positive
under the simultaneous analysis.''')
    text=text.replace('Grouping related descriptors\nis a plausible refinement, but requires fresh predictive comparison before\nreplacing the reference definition.',
        'Training-fold redundancy reduction now yields nearly unchanged predictive\nloss, while fixed grouping has small costs. The historical definition remains\nthe common reference; alternative representations are explicitly versioned.')
    text=text.replace('predictive evidence from the still open question of its decision value.',
        'predictive evidence from cross-method selection. Within-method budget\nallocation has now been evaluated in Section~\\ref{sec:revision_allocation};\nit does not identify which alternative optimization method to choose.')
    text=text.replace('pointwise intervals do not correct for the many comparisons.',
        'the historical pointwise intervals do not correct for the many comparisons.\nThe revision adds joint simultaneous intervals for its 50 declared primary\nscale/direction contrasts, while secondary sensitivities remain exploratory.')
    conclusion=r'''\section{Conclusion}
Scenario criticality contains useful information about timely economic and
service outcomes within the six sampled regimes. The numerical indicator
must nevertheless be distinguished from the underlying deterministic
scenario and its realized MILP. In particular, the audited A3 proxy repeats
demand-scale information and does not measure a realized peak-to-valley ratio.

Deadline-specific rescaling explains part of the improvement previously
attributed jointly to scale and weights. Separate weight directions still
reduce pooled MSE by 24.3--36.1\% beyond this intermediate model, but
nonlinear predictions are more accurate and redundant components can be
removed with little loss. Leaving a complete family out multiplies MSE by
2.72--3.63, materially restricting transportability claims.

Retrospective allocation within held-out portfolios establishes a limited
decision use: 415 joint successes versus 341 under uniform allocation at a
30-second average reserve. A compact structural representation performs
similarly, so the most accurate continuous predictor is not automatically
the best choice for this decision. These findings remain conditional on
historical observations, HiGHS and the declared preferences. The independent
test campaign, stronger references and additional instrumented repetitions
are prepared but unexecuted; no corresponding empirical conclusions are drawn.

'''
    a=text.index('\\section{Conclusion}');b=text.index('\\appendix',a)
    text=text[:a]+conclusion+text[b:]
    # Position the study using distinct objects and evaluation criteria.
    related=r'''\begin{table}[htbp]\centering\small
\begin{tabular}{p{.23\linewidth}p{.31\linewidth}p{.35\linewidth}}\toprule
Research strand & Object being measured & Relation to this study\\\midrule
Physical flexibility \citep{lannoye2012,ulbig2015,terca2026flexibility} & Ability to accommodate residual-load variation over time & Physical descriptors do not by themselves measure algorithm runtime\\
Empirical algorithm performance \citep{hutter2013} & Outcome of a specified method on an instance & Delivery, economic quality and service are explicit endpoints\\
Composite indicators \citep{oecd2008} & Aggregation under normalization and weighting conventions & Arithmetic sensitivity is separated from predictive weight estimation\\
Selection and resource allocation \citep{rice1976,bischl2016} & Decisions improving a portfolio's outcomes & The present decision experiment allocates time to HiGHS; cross-method selection is distinct\\\bottomrule
\end{tabular}
\caption{Positioning: physical adequacy, computational prediction, aggregation and decisions answer different questions.}
\end{table}
Feature-based benchmark selection, as used by MIPLIB \citep{miplibselection},
also makes the sampling design an explicit part of evaluation. General MIP
instances cannot directly replace energy scenarios requiring these physical
descriptors. PGLib-UC \citep{pglibuc} offers a complementary unit-commitment
benchmark, but its system-wide demand formulation does not reproduce every
multizone storage and load-shifting component studied here; an external
adapter would need to identify the common formulation and unavailable features.

'''
    text=text.replace(r'\section{Methods}',related+r'\section{Methods}',1)
    text=text.replace('vectors for each deadline. Scenario-grouped nested validation separates',
        'vectors for each deadline, including a shared-direction intermediate\nmodel. We then test transfer between families and the utility of time\nallocation within held-out portfolios. Scenario-grouped nested validation separates')
    path.write_text(text,encoding='utf-8')

if __name__=='__main__':run()
