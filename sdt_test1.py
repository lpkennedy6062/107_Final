import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    data = pd.read_csv(file_path)
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if display:
        print("\nRaw data sample:")
        print(data.head())

    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        sdt_data = []
        for p in grouped['pnum'].unique():
            sub = grouped[grouped['pnum'] == p]
            for cond in sub['condition'].unique():
                cd = sub[sub['condition'] == cond]
                sig = cd[cd['signal'] == 0]
                noise = cd[cd['signal'] == 1]
                if not sig.empty and not noise.empty:
                    sdt_data.append({
                        'pnum': p,
                        'condition': cond,
                        'hits': sig['correct'].iloc[0],
                        'misses': sig['nTrials'].iloc[0] - sig['correct'].iloc[0],
                        'false_alarms': noise['nTrials'].iloc[0] - noise['correct'].iloc[0],
                        'correct_rejections': noise['correct'].iloc[0],
                        'nSignal': sig['nTrials'].iloc[0],
                        'nNoise': noise['nTrials'].iloc[0]
                    })
        df = pd.DataFrame(sdt_data)
        return df

    if prepare_for == 'delta plots':
        dp_list = []
        for p in data['pnum'].unique():
            for cond in data['condition'].unique():
                cd = data[(data['pnum']==p)&(data['condition']==cond)]
                if cd.empty: continue
                for mode in ['overall','accurate','error']:
                    subset = cd if mode=='overall' else (cd[cd['accuracy']==(1 if mode=='accurate' else 0)])
                    if subset.empty: continue
                    row = {'pnum':p,'condition':cond,'mode':mode}
                    for pctl in PERCENTILES:
                        row[f'p{pctl}'] = np.percentile(subset['rt'],pctl)
                    dp_list.append(row)
        return pd.DataFrame(dp_list)


def apply_hierarchical_sdt_model(data):
    P = data['pnum'].nunique()
    C = data['condition'].nunique()
    with pm.Model() as model:
        mu_d = pm.Normal('mu_d', 0, 1)
        beta_stim = pm.Normal('beta_stim', 0, 1)
        beta_diff = pm.Normal('beta_diff', 0, 1)
        mu_c = pm.Normal('mu_c', 0, 1)
        gamma_stim = pm.Normal('gamma_stim', 0, 1)
        gamma_diff = pm.Normal('gamma_diff', 0, 1)
        sigma_d = pm.Exponential('sigma_d',1)
        sigma_c = pm.Exponential('sigma_c',1)
        d_sub = pm.Normal('d_sub', 0, sigma_d, shape=P)
        c_sub = pm.Normal('c_sub', 0, sigma_c, shape=P)
        stim = data['condition'] % 2
        diff = data['condition'] // 2
        subj = data['pnum'].astype('category').cat.codes.values
        d_prime = mu_d + beta_stim*stim + beta_diff*diff + d_sub[subj]
        c_crit = mu_c + gamma_stim*stim + gamma_diff*diff + c_sub[subj]
        hit_p = pm.math.invprobit((d_prime/2)-c_crit)
        fa_p  = pm.math.invprobit((-d_prime/2)-c_crit)
        pm.Binomial('hit_obs', n=data['nSignal'], p=hit_p, observed=data['hits'])
        pm.Binomial('fa_obs', n=data['nNoise'], p=fa_p, observed=data['false_alarms'])
    return model


def draw_delta_plots(dp_data, pnum):
    conds = sorted(dp_data['condition'].unique())
    n = len(conds)
    fig, axes = plt.subplots(n,n,figsize=(4*n,4*n))
    for i, c1 in enumerate(conds):
        for j, c2 in enumerate(conds):
            if i>=j: axes[i,j].axis('off'); continue
            for mode, col in [('overall','black'),('error','red'),('accurate','green')]:
                d1 = dp_data[(dp_data['pnum']==pnum)&(dp_data['condition']==c1)&(dp_data['mode']==mode)]
                d2 = dp_data[(dp_data['pnum']==pnum)&(dp_data['condition']==c2)&(dp_data['mode']==mode)]
                if d1.empty or d2.empty: continue
                delta = d2[[f'p{p}' for p in PERCENTILES]].values.flatten() - d1[[f'p{p}' for p in PERCENTILES]].values.flatten()
                axes[i,j].plot(PERCENTILES, delta, color=col, marker='o', linewidth=2)
            axes[i,j].axhline(0, linestyle='--', alpha=0.5)
            axes[i,j].set_title(f"{CONDITION_NAMES[c2]} - {CONDITION_NAMES[c1]}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1) SDT analysis
    sdt_df = read_data('data.csv', prepare_for='sdt', display=True)
    sdt_model = apply_hierarchical_sdt_model(sdt_df)
    with sdt_model:
        sdt_trace = pm.sample(draws=3000, tune=2000, target_accept=0.95, return_inferencedata=True)
    az.summary(sdt_trace, var_names=['mu_d','beta_stim','beta_diff','mu_c','gamma_stim','gamma_diff'])
    az.plot_posterior(sdt_trace, var_names=['beta_stim','beta_diff','gamma_stim','gamma_diff'])
    plt.show()


    dp_df = read_data('data.csv', prepare_for='delta plots', display=True)
    draw_delta_plots(dp_df, pnum=1)
