"""Generate all revision tables, figures and manuscript results from saved outputs."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .data import RUN,ROOT,HOME,WEIGHTS,COMP,load,read,put,sha,digest
from .analysis import collect,specifications

LABEL={'equal4':'Equal','cost_priority':'Cost','co2_priority':'Direct thermal CO2','time_priority':'Time','service4_co2':'Service'}
DEST=ROOT/'research/three_papers/manuscripts/paper1/results'


def summary(p):
    p=p.copy();p['squared_loss']=(p.y-p.p)**2
    rows=[]
    for (target,model),q in p.groupby(['target','model']):
        for b,s in [('all',q)]+list(q.groupby('budget')):
            rows.append(dict(target=target,model=model,budget=b,n=len(s),mse=float(s.squared_loss.mean()),
                outside=float(((s.p<0)|(s.p>1)).mean()),mae=float(np.mean(abs(s.p-s.y)))))
    return pd.DataFrame(rows)


def contrasts(p,cfg):
    vectors=[];keys=[]
    for target in cfg['policies']:
        for budget in cfg['deadlines']:
            q=p[(p.target==target)&(p.budget==budget)].copy()
            q['loss']=(q.y-q.p)**2;wide=q.pivot(index='group',columns='model',values='loss').sort_index()
            for left,right in [('all_common','all_rank1'),('all_rank1','all_separate')]:
                vectors.append((wide[left]-wide[right]).to_numpy());keys.append((target,budget,left,right))
    values=np.array(vectors).T;mean=values.mean(axis=0);n=len(values)
    rng=np.random.default_rng(cfg['seed']);boot=np.empty((cfg['bootstrap_samples'],values.shape[1]))
    for i in range(len(boot)):
        boot[i]=values[rng.integers(0,n,n)].mean(axis=0)
    se=boot.std(axis=0,ddof=1);se[se<1e-15]=1e-15
    critical=np.quantile(np.max(np.abs((boot-mean)/se),axis=1),.95)
    rows=[]
    for j,(target,budget,left,right) in enumerate(keys):
        lo,hi=np.quantile(boot[:,j],[.025,.975])
        rows.append(dict(target=target,budget=budget,left=left,right=right,gain=mean[j],lower=lo,upper=hi,
            simultaneous_lower=mean[j]-critical*se[j],simultaneous_upper=mean[j]+critical*se[j],
            within_declared_equivalence_margin=bool(lo>-.001 and hi<.001)))
    result=pd.DataFrame(rows);result.to_csv(RUN/'primary_contrasts.csv',index=False)
    put(RUN/'primary_inference.json',dict(method='Centered joint scenario-bootstrap max-standardized 95% simultaneous intervals',
        contrasts=50,bootstrap_samples=len(boot),critical=float(critical),
        amendment='Use joint max-standardized intervals rather than poorly resolved Bonferroni tail quantiles with 2000 samples. All 50 declared contrasts retained.',
        limitation='Approximate, conditional on saved held-out predictions; does not include retraining variance'))
    return result


def validate(p,cfg):
    expected=len(cfg['policies'])*len(specifications())*3000
    assert len(p)==expected,(len(p),expected)
    assert not p.duplicated(['target','model','group','budget']).any()
    old=pd.read_csv(WEIGHTS/'report/predictions.csv');checks=[]
    for target in cfg['policies']:
        for new,oldname in [('all_common','common_positive'),('original_common','original_index'),('constant','intercept_only'),('all_separate','separate')]:
            a=p[(p.target==target)&(p.model==new)]
            q=old[(old.policy==target)&(old.model.str.startswith('deadline_') if oldname=='separate' else old.model==oldname)]
            joined=a.merge(q,on=['group','budget'],validate='one_to_one')
            assert len(joined)==3000
            err=float(np.max(abs(joined.p-joined.predicted)))
            checks.append(dict(target=target,model=new,max_prediction_error=err))
            assert err<2e-5,checks[-1]
    put(RUN/'baseline_refit_validation.json',dict(comparisons=checks,tolerance=2e-5))
    for stage in ['nested','lofo','probability_nested','probability_lofo']:
        for path in (RUN/'checkpoints'/stage).rglob('*.json'):
            obj=read(path);assert not set(obj['train_groups'])&{r['group'] for r in obj['predictions']}


def coefficient_diagnostics():
    convergence=[];weights=[]
    for path in (RUN/'checkpoints/nested').rglob('*_all_rank1.json'):
        o=read(path);m=o['meta']['model'];coefs=np.array(m['coefficients'])
        for d in m['diagnostics']:convergence.append(dict(target=o['target'],fold=o['fold'],**d))
        for b,c in zip(m['levels'],coefs):
            w=c/c.sum() if c.sum()>1e-12 else np.zeros_like(c)
            weights.append(dict(target=o['target'],fold=o['fold'],budget=b,scale=c.sum(),
                weather_group=w[1]+w[3],demand_group=w[2]+w[4],storage_group=w[6]+w[7],B1=w[12],B3=w[14],B4=w[15],
                uncertainty='descriptive variation over outer training folds; penalty reselected'))
    pd.DataFrame(convergence).to_csv(RUN/'rank1_convergence.csv',index=False)
    pd.DataFrame(weights).to_csv(RUN/'rank1_group_weights.csv',index=False)


def calibration_and_reference(p,df):
    tables=[];subgroups=[]
    p=p.merge(df[['group','budget','reference_certified']],on=['group','budget'],suffixes=('','_source'),validate='many_to_one')
    for (target,model,budget),q in p.groupby(['target','model','budget']):
        if target not in ('E','ES'):continue
        q=q.copy();q['bin']=np.minimum((q.p*10).astype(int),9)
        for i,s in q.groupby('bin'):
            tables.append(dict(target=target,model=model,budget=budget,bin=i,n=len(s),predicted=s.p.mean(),observed=s.y.mean()))
        for certified,s in q.groupby('reference_certified'):
            subgroups.append(dict(target=target,model=model,budget=budget,certified=bool(certified),n=len(s),brier=float(np.mean((s.p-s.y)**2))))
    pd.DataFrame(tables).to_csv(RUN/'calibration.csv',index=False)
    pd.DataFrame(subgroups).to_csv(RUN/'binary_reference_subgroups.csv',index=False)


def allocation_summary():
    a=pd.read_csv(RUN/'allocation_results.csv');b=pd.read_parquet(RUN/'allocation_bootstrap.parquet')
    total=a.groupby(['endpoint','mean_budget','model']).agg(successes=('successes','sum'),reserved_seconds=('reserved_seconds','sum'),observed_exit_seconds=('observed_exit_seconds','sum')).reset_index()
    boot=b.groupby(['endpoint','mean_budget','model','repetition']).successes.sum().unstack('model')
    intervals=[]
    for (endpoint,average),q in boot.groupby(level=['endpoint','mean_budget']):
        for model in ['size_direct','original_direct','compact_direct','all_direct','oracle']:
            for ref in ['uniform','size_direct']:
                lo,hi=np.quantile(q[model]-q[ref],[.025,.975]);intervals.append(dict(endpoint=endpoint,mean_budget=average,model=model,reference=ref,lower=lo,upper=hi))
    total.to_csv(RUN/'allocation_totals.csv',index=False);pd.DataFrame(intervals).to_csv(RUN/'allocation_gain_intervals.csv',index=False)
    return total,pd.DataFrame(intervals)


def figures(scores,lofo,prob,allocation,contrasts,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    DEST.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight');fig.savefig(DEST/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
    failures=pd.read_csv(RUN/'failure_decomposition.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.4));bottom=np.zeros(5)
    for col,label,color in [('not_delivered','No timely valid delivery','#404b69'),('delivered_quality_failure','Delivered, economic failure','#d88950'),('economic_service_failure','Economic success, service failure','#d8bd6b'),('success_ES','Economic and service success','#469785')]:
        vals=failures[col].to_numpy()/600;ax.bar(np.arange(5),vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax.set_xticks(np.arange(5),failures.budget);ax.set(xlabel='Independent deadline (s)',ylabel='Fraction of scenarios');ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',frameon=False)
    save(fig,'revision_failure_decomposition')
    fig,axs=plt.subplots(1,5,figsize=(12,2.9),sharey=True)
    for ax,target in zip(axs,cfg['policies']):
        q=scores[(scores.target==target)&(scores.budget!='all')]
        for model,label,color in [('all_common','Common','#8f9db4'),('all_rank1','Common direction + scales','#e8a34b'),('all_separate','Separate vectors','#2e7885')]:
            z=q[q.model==model].sort_values('budget');ax.plot(range(5),z.mse,'o-',label=label,color=color,markersize=3)
        ax.set_title(LABEL[target]);ax.set_xticks(range(5),cfg['deadlines'],rotation=45);ax.set_xlabel('Deadline (s)')
    axs[0].set_ylabel('Held-out MSE');axs[2].legend(loc='upper center',bbox_to_anchor=(.5,-.35),ncol=3,frameon=False)
    save(fig,'revision_scale_direction')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.4))
    q=lofo[(lofo.target=='equal4')&(lofo.model=='all_separate')].copy();q['loss']=(q.y-q.p)**2
    pivot=q.groupby(['family','budget']).loss.mean().unstack().reindex(columns=cfg['deadlines'])
    short=lambda s:s.replace('_criticality_scenarios','').replace('persona_','').replace('_scenarios','').replace('_',' ')
    im=axs[0].imshow(pivot,aspect='auto',cmap='YlOrRd');axs[0].set_yticks(range(6),[short(s) for s in pivot.index]);axs[0].set_xticks(range(5),cfg['deadlines']);axs[0].set(xlabel='Deadline (s)',title='Held-out family: Equal preference')
    fig.colorbar(im,ax=axs[0],label='MSE',fraction=.035)
    for model,label,color in [('uniform','Uniform / fixed mixture','#8d939c'),('size_direct','Size','#cc9b48'),('compact_direct','B1 + B3 + B4','#765592'),('all_direct','22 components','#227c87'),('oracle','Observed oracle','#262d39')]:
        z=allocation[(allocation.endpoint=='ES')&(allocation.model==model)].sort_values('mean_budget');axs[1].plot(z.mean_budget,z.successes,'o-',label=label,color=color)
    axs[1].set(xlabel='Reserved mean budget (s)',ylabel='Joint successes / 600',title='Allocation within held-out portfolios');axs[1].legend(fontsize=7,frameon=False)
    fig.tight_layout();save(fig,'revision_transfer_allocation')
    c=pd.read_csv(RUN/'calibration.csv');fig,axs=plt.subplots(1,3,figsize=(10,3))
    for ax,b in zip(axs,[5,30,120]):
        for model,label in [('all_direct','Direct'),('all_hurdle','Conditional product')]:
            z=c[(c.target=='ES')&(c.model==model)&(c.budget==b)].sort_values('predicted')
            ax.plot(z.predicted,z.observed,'o-',label=label,markersize=4)
            for row in z.itertuples():ax.annotate(str(row.n),(row.predicted,row.observed),fontsize=5,xytext=(2,3),textcoords='offset points')
        ax.plot([0,1],[0,1],'k:',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),title=f'{b} s',xlabel='Predicted ES probability')
    axs[0].set_ylabel('Observed frequency');axs[-1].legend(fontsize=7);fig.tight_layout();save(fig,'revision_calibration')
    # Harmonized legacy curves, regenerated from the original numerical table.
    legacy=pd.read_csv(COMP/'report/tables/prediction_scores.csv')
    for control in [False,True]:
        fig,ax=plt.subplots(figsize=(7.5,3.6))
        for name,label in [('size','Size'),('index','Index'),('axes','Two blocks'),('interpretable','Six components'),('linear','22 linear'),('additive','Additive'),('interactions','Products')]:
            z=legacy[(legacy.endpoint=='economic_5')&(legacy.family_control==control)&(legacy.description==name)].sort_values('budget')
            ax.plot(z.budget,z.brier,'o-',label=label,markersize=3)
        ax.set(xlabel='Independent deadline (s)',ylabel='Held-out Brier loss',title='Economic success - '+('family-conditioned models' if control else 'models without family controls'));ax.legend(fontsize=7,ncol=2,frameon=False)
        save(fig,f'prediction_economic_5_family{int(control)}')


def tex(scores,lofo,prob,allocation,ci,cfg):
    pooled=scores[scores.budget=='all'].pivot(index='target',columns='model',values='mse')
    lp=summary(lofo);lp=lp[lp.budget=='all'].pivot(index='target',columns='model',values='mse')
    stage1=100*(1-pooled.all_rank1/pooled.all_common);stage2=100*(1-pooled.all_separate/pooled.all_rank1)
    positive=int((ci[ci.right=='all_separate'].simultaneous_lower>0).sum())
    main=[r'\subsection{Do changing scales explain deadline-specific weights?}',r'\label{sec:revision_scale}',
        r'The intermediate model preserves one nonnegative normalized direction while allowing a separate nonnegative amplitude and intercept at each deadline. It is fitted with three fixed initializations and a ridge penalty on the effective coefficients; all tuning remains inside the training folds. The separate model selects its penalty separately for each deadline.',
        f'The common-direction model reduces pooled MSE by {stage1.min():.1f}--{stage1.max():.1f}\\% relative to the common-coefficient fit. Allowing the direction also to change reduces MSE by another {stage2.min():.1f}--{stage2.max():.1f}\\% (Table~\\ref{{tab:revision_models}}). Thus amplitude changes explain a substantial, preference-dependent part of the original gain, but do not account for it completely.',
        f'Of the 25 preference--deadline contrasts between the common-direction and separate models, {positive} have positive lower bounds in the joint scenario-bootstrap simultaneous intervals over all 50 primary contrasts. These approximate intervals condition on the held-out predictions and do not include model-refitting uncertainty. A nonsignificant contrast is not evidence of equivalence; the declared exploratory margin is 0.001 absolute MSE.',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrrrrr}\toprule',
        r'Preference & Common & Scaled & Separate & B1 & B1+B3+B4 & Boosting\\\midrule']
    for target in cfg['policies']:
        vals=pooled.loc[target,['all_common','all_rank1','all_separate','size_separate','compact_separate','boosting']]
        main.append(LABEL[target].replace('CO2',r'CO$_2$')+' & '+' & '.join(f'{v:.5f}' for v in vals)+r'\\')
    main += [r'\bottomrule\end{tabular}\caption{Nested held-out MSE over 3,000 observations per preference. Scaled denotes the common-direction model. B1 and the compact model have deadline-specific coefficients, as does boosting. These comparisons use the original five scenario folds of the weight study.}\label{tab:revision_models}\end{table}',
        r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{paper1/results/revision_scale_direction.pdf}\caption{Separating amplitude and direction changes using independent deadline-specific observations.}\end{figure}',
        r'\subsection{How much representation is necessary?}',r'\label{sec:revision_compact}',
        r'The 22-component separate model improves on B1 alone and on B1+B3+B4 for each pooled preference, but the flexible boosting model is more accurate for all five. The linear index is therefore an interpretability--accuracy compromise. Removing training-fold constants and correlations above 0.95 produces nearly unchanged losses; this is evidence against treating all 22 entries as distinct empirically identified dimensions. Fixed grouping of (A2,A4), (A3,A5) and (A7,A8), and a separate nonsaturating transformation of A6 and B8 are reported in the reproducibility tables. These are evaluated alternative representations, not silent changes to the historical index.',
        r'\subsection{What is predicted: delivery, quality or service?}',r'\label{sec:revision_failures}',
        r'At 5 seconds, 468 of 600 runs do not deliver a valid dispatch on time, 28 deliver but fail economic quality, and 3 pass economics but fail service. At 120 seconds, all deliver, 95 fail economics and 30 additional runs fail service. This decomposition separates pipeline delivery difficulty from the quality of a delivered dispatch (Figure~\ref{fig:revision_failures}).',
        r'\begin{figure}[htbp]\centering\includegraphics[width=.94\linewidth]{paper1/results/revision_failure_decomposition.pdf}\caption{Mutually exclusive observed outcomes. Short deadlines predominantly assess delivery of the complete computational pipeline.}\label{fig:revision_failures}\end{figure}',
        r'Conditional fractional-logistic fits estimate delivery, economic success given delivery, and service given economic success. Shared tuning of the conditional factors guarantees $\widehat P(ES)\leq\widehat P(E)$. A continuous two-part model combines non-delivery probability with the conditional bounded loss. Projection of linear predictions onto $[0,1]$ is also reported; it cannot increase squared error for this target. Calibration bins and their counts are included in the supplement.',
        r'\subsection{How far do the findings transfer?}',r'\label{sec:revision_transfer}',
        r'Leaving one complete family out, with leave-one-family-out tuning among the remaining five, reveals substantial deterioration. Family labels are excluded from all predictors. This assesses transfer among the six constructed regimes, not an independent external system.',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule Preference & Within-family folds & Held-out family & Ratio\\\midrule']
    for target in cfg['policies']:
        a=pooled.loc[target,'all_separate'];b=lp.loc[target,'all_separate']
        main.append(LABEL[target].replace('CO2',r'CO$_2$')+f' & {a:.5f} & {b:.5f} & {b/a:.2f}'+r'\\')
    main += [r'\bottomrule\end{tabular}\caption{Pooled MSE of separate 22-component models under the two validation designs. Ratios describe the change of evaluation distribution; they are not paired causal effects.}\label{tab:revision_transfer}\end{table}',
        r'The independent physical campaign is prepared but has not been executed: four batches of 100 fresh scenarios cover unfiltered draws, temporal structure, topology and decoupled parameters. No performance claim is made for these unobserved outcomes. Horizon coefficients remain unidentified in the historical training population.',
        r'\subsection{Can the descriptors improve a time-allocation decision?}',r'\label{sec:revision_allocation}',
        r'Within each outer test fold, an exact multiple-choice budget allocation maximizes predicted ES success subject to a sum of reserved deadlines. The model has seen none of the scenarios in that portfolio. E is a secondary endpoint. Uniform allocations, a predetermined adjacent-deadline mixture, size-only predictions, the original index, B1+B3+B4 and the observed-outcome oracle use the same budget constraint.',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{rrrrrrr}\toprule Mean seconds & Uniform & Size & Index & Compact & 22 & Oracle\\\midrule']
    for average in cfg['allocation_mean_budgets']:
        row=allocation[(allocation.endpoint=='ES')&(allocation.mean_budget==average)].set_index('model')
        vals=[int(row.loc[m,'successes']) for m in ['uniform','size_direct','original_direct','compact_direct','all_direct','oracle']]
        main.append(str(average)+' & '+' & '.join(map(str,vals))+r'\\')
    main += [r'\bottomrule\end{tabular}\caption{Observed ES successes across five disjoint test portfolios, totaling 600 scenarios. The oracle uses observed outcomes, not true success probabilities.}\label{tab:revision_allocation}\end{table}',
        r'The compact representation matches or exceeds the 22-component allocation at three of the four budgets, despite its higher continuous prediction loss. Predictive accuracy and decision utility therefore yield different representation choices. Portfolio bootstrap intervals resample scenarios and reoptimize the allocation within each fold; models remain fixed. Reserved seconds and observed process-exit seconds are recorded separately. No observed portfolio exceeds its total reserve, but these retrospective sums do not establish a strict real-time guarantee or include the online cost of feature computation and allocation.',
        r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{paper1/results/revision_transfer_allocation.pdf}\caption{Family transfer for the equal preference (left) and retrospective ES budget allocation (right). Predictive generalization is limited while within-regime decision utility can remain useful.}\end{figure}']
    (DEST/'revision_results.tex').write_text('\n\n'.join(main)+'\n',encoding='utf-8')
    put(RUN/'headline_results.json',dict(scale_gain_percent=stage1.to_dict(),direction_gain_percent=stage2.to_dict(),
        simultaneous_positive_direction_contrasts=positive,lofo_mse_ratio=(lp.all_separate/pooled.all_separate).to_dict(),
        physical_campaign_executed=False))


def run():
    df,cfg=load();p=collect('nested');validate(p,cfg)
    lofo=collect('lofo');prob=collect('probability_nested');problofo=collect('probability_lofo')
    for stage,q in [('continuous_nested',p),('continuous_lofo',lofo),('probability_nested',prob),('probability_lofo',problofo)]:
        q.to_parquet(RUN/(stage+'_predictions.parquet'),index=False);summary(q).to_csv(RUN/(stage+'_scores.csv'),index=False)
    scores=summary(p);coefficient_diagnostics();calibration_and_reference(prob,df);ci=contrasts(p,cfg)
    joined=prob[prob.target.isin(['E','ES'])].pivot(index=['group','budget','model'],columns='target',values='p')
    ordered=joined[joined.index.get_level_values('model').str.endswith('hurdle')]
    assert (ordered.ES<=ordered.E+1e-12).all()
    bounded=[]
    for (target,budget),q in p[p.model=='all_separate'].groupby(['target','budget']):
        h=prob[(prob.target==target)&(prob.budget==budget)&(prob.model=='all_hurdle')]
        bounded.append(dict(target=target,budget=budget,linear_mse=float(np.mean((q.p-q.y)**2)),
            clipped_mse=float(np.mean((np.clip(q.p,0,1)-q.y)**2)),two_part_mse=float(np.mean((h.p-h.y)**2)),outside_fraction=float(((q.p<0)|(q.p>1)).mean())))
    pd.DataFrame(bounded).to_csv(RUN/'bounded_model_comparison.csv',index=False)
    for certified in [True,False]:
        summary(p[p.reference_certified==certified]).to_csv(RUN/f'continuous_certified_{certified}.csv',index=False)
    total,intervals=allocation_summary();figures(scores,lofo,prob,total,ci,cfg);tex(scores,lofo,prob,total,ci,cfg)
    put(RUN/'report_receipt.json',dict(source_manifest=sha(RUN/'manifest.json'),tables={p.name:sha(p) for p in RUN.glob('*.csv')},
        figures={p.name:sha(p) for p in DEST.glob('revision_*')},status='historical_analyses_complete; independent physical campaign not executed'))
