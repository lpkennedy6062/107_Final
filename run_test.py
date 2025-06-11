from sdtretest import read_data, apply_hierarchical_sdt_model, draw_delta_plots, plot_sdt_posteriors
import pymc as pm
import arviz as az

sdt_df    = read_data('data.csv', prepare_for='sdt', display=True)
sdt_model = apply_hierarchical_sdt_model(sdt_df)

    # Sample and save trace
with sdt_model:
    trace = pm.sample(draws=2000, tune=1000, target_accept=0.99, cores=4, return_inferencedata=True)
    az.to_netcdf(trace, 'sdt_trace.nc')

    # Population-level summary
pop_sum = az.summary(trace, var_names=['mu_d','beta_stim','beta_diff','mu_c','gamma_stim','gamma_diff'], round_to=2)
print("\nPopulation-level Posteriors:\n", pop_sum)

    # Subject-level summary
subj_sum = az.summary(trace, var_names=['d_sub','c_sub'], hdi_prob=0.94, round_to=2)
print("\nSubject-level Posteriors:\n", subj_sum)

plot_sdt_posteriors(trace)

    # Convergence (concise)
rhat_ds  = az.rhat(trace)
rhat_vals = {p: round(float(rhat_ds[p].values),2) for p in ['mu_d','beta_stim','beta_diff','mu_c','gamma_stim','gamma_diff']}
ess_ds   = az.ess(trace)
ess_vals  = {p: int(ess_ds[p].values) for p in ['mu_d','beta_stim','beta_diff','mu_c','gamma_stim','gamma_diff']}
print("\nR-hat:", rhat_vals)
print("ESS:", ess_vals)

    # Delta plot for participant 1
dp_df = read_data('data.csv', prepare_for='delta plots', display=False)
draw_delta_plots(dp_df, pnum=1)
