import pickle, pandas as pd
from scipy.stats import wilcoxon

k1 = pd.DataFrame(pickle.load(open(
    'outputs/uc_jepa/latent_time_budget_120_caps_k1/pipeline_eval_all_families.pkl','rb')))
k5 = pd.DataFrame(pickle.load(open(
    'outputs/uc_jepa/latent_time_budget_120_caps_k5/pipeline_eval_all_families.pkl','rb')))
m = k1[['family','scenario_id','lp_objective']].merge(
    k5[['family','scenario_id','lp_objective']],
    on=['family','scenario_id'], suffixes=('_k1','_k5'))
for fam, sub in m.groupby('family'):
    diff = sub['lp_objective_k5'] - sub['lp_objective_k1']
    print(fam, 'median delta=', diff.median(),
          'wilcoxon p=', wilcoxon(sub['lp_objective_k1'], sub['lp_objective_k5']).pvalue)