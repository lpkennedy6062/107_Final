"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data for SDT or delta plot analysis"""
    data = pd.read_csv(file_path)

    # Map categorical to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if display:
        print("\nRaw data sample:")
        print(data.head())

    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg(
            nTrials=('accuracy','count'),
            correct=('accuracy','sum')
        ).reset_index()

        sdt_rows = []
        for p in grouped['pnum'].unique():
            sub = grouped[grouped['pnum']==p]
            for cond in sub['condition'].unique():
                cd = sub[sub['condition']==cond]
                sig = cd[cd['signal']==0]
                noise = cd[cd['signal']==1]
                if not sig.empty and not noise.empty:
                    sdt_rows.append({
                        'pnum':p,
                        'condition':cond,
                        'hits':sig['correct'].iloc[0],
                        'nSignal':sig['nTrials'].iloc[0],
                        'false_alarms':noise['nTrials'].iloc[0]-noise['correct'].iloc[0],
                        'nNoise':noise['nTrials'].iloc[0]
                    })
        return pd.DataFrame(sdt_rows)

    elif prepare_for == 'delta plots':
        dp_list=[]
        for p in data['pnum'].unique():
            for cond in data['condition'].unique():
                cd = data[(data['pnum']==p)&(data['condition']==cond)]
                if cd.empty: continue
                for mode, mask in [('overall', slice(None)),
                                   ('accurate', cd['accuracy']==1),
                                   ('error', cd['accuracy']==0)]:
                    subset = cd if mode=='overall' else cd[mask]
                    if subset.empty: continue
                    row={'pnum':p,'condition':cond,'mode':mode}
                    for pctl in PERCENTILES:
                        row[f'p{pctl}'] = np.percentile(subset['rt'],pctl)
                    dp_list.append(row)
        return pd.DataFrame(dp_list)


def apply_hierarchical_sdt_model(data):
    """Builds and returns a non-centered hierarchical SDT PyMC model"""
    P = data['pnum'].nunique()

    with pm.Model() as model:
        # Population-level parameters
        mu_d       = pm.Normal('mu_d', 0, 1)
        beta_stim  = pm.Normal('beta_stim', 0, 1)
        beta_diff  = pm.Normal('beta_diff', 0, 1)
        mu_c       = pm.Normal('mu_c', 0, 1)
        gamma_stim = pm.Normal('gamma_stim',0,1)
        gamma_diff = pm.Normal('gamma_diff',0,1)

        # Non-centered random effects
        sigma_d = pm.Exponential('sigma_d', 1)
        z_d     = pm.Normal('z_d', 0, 1, shape=P)
        d_sub   = pm.Deterministic('d_sub', z_d * sigma_d)

        sigma_c = pm.Exponential('sigma_c', 1)
        z_c     = pm.Normal('z_c', 0, 1, shape=P)
        c_sub   = pm.Deterministic('c_sub', z_c * sigma_c)

        # Indices and cell-level
        subj = data['pnum'].astype('category').cat.codes.values
        stim = data['condition'] % 2
        diff = data['condition'] // 2

        d_prime = mu_d + beta_stim*stim + beta_diff*diff + d_sub[subj]
        c_crit  = mu_c + gamma_stim*stim + gamma_diff*diff + c_sub[subj]

        # SDT link
        hit_p = pm.math.invprobit((d_prime/2)-c_crit)
        fa_p  = pm.math.invprobit((-d_prime/2)-c_crit)

        # Likelihood
        pm.Binomial('hit_obs', n=data['nSignal'], p=hit_p, observed=data['hits'])
        pm.Binomial('fa_obs',  n=data['nNoise'],  p=fa_p,  observed=data['false_alarms'])

    return model


def draw_delta_plots(data, pnum):
    conds = sorted(data['condition'].unique())
    n = len(conds)
    fig, axes = plt.subplots(n,n,figsize=(4*n,4*n))
    for i,c1 in enumerate(conds):
        for j,c2 in enumerate(conds):
            if i>=j:
                axes[i,j].axis('off'); continue
            for mode in [('overall','black'),('error','red'),('accurate','green')]:
                d1 = data[(data['pnum']==pnum)&(data['condition']==c1)&(data['mode']==mode)]
                d2 = data[(data['pnum']==pnum)&(data['condition']==c2)&(data['mode']==mode)]
                if d1.empty or d2.empty: continue
                delta = d2[[f'p{p}' for p in PERCENTILES]].values.flatten() - d1[[f'p{p}' for p in PERCENTILES]].values.flatten()
                axes[i,j].plot(PERCENTILES, delta, marker='o', linewidth=2)
            axes[i,j].axhline(0, linestyle='--', alpha=0.5)
            axes[i,j].set_title(f"{CONDITION_NAMES[c2]} - {CONDITION_NAMES[c1]}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # SDT analysis
    sdt_df    = read_data('data.csv', prepare_for='sdt', display=True)
    sdt_model = apply_hierarchical_sdt_model(sdt_df)
    with sdt_model:
        sdt_trace = pm.sample(draws=3000, tune=1000, target_accept=0.99, cores=4, return_inferencedata=True)
    az.summary(sdt_trace, var_names=['mu_d','beta_stim','beta_diff','mu_c','gamma_stim','gamma_diff'])
    az.plot_posterior(sdt_trace, var_names=['beta_stim','beta_diff','gamma_stim','gamma_diff'])
    plt.show()

    # Delta plot for participant 1
    dp_df = read_data('data.csv', prepare_for='delta plots', display=True)
    draw_delta_plots(dp_df, pnum=1)
