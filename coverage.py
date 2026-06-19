import os
import pickle

import ROOT
import numpy as np

import matplotlib.pyplot as plt

import emm

from tools import storage
from tools import scale_out as so

coverage_cache = storage.ensure_cache("coverage")

def evaluate_model(x, model, params, x_vals):
    model = model(x)
    model.set_params(params)
    return model.name, emm.evaluate_pdf(x, model, x_vals)

def bootstrap_filename(toy_model, seed, n_bootstraps, n_events):
    return f"{coverage_cache}/{toy_model.name}_seed{seed}_{n_bootstraps}bootstraps_{n_events}events.pkl"

def fit_models_bootstrap(
    x, toy_model, model_primitives,
    n_bootstraps, n_events, seed,
    test_points
    ):

    # Set the random seed for reproducibility
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))
    np.random.seed(int(seed))

    data = toy_model.pdf.generate(ROOT.RooArgSet(x), n_events)
    data_arr = np.array([data.get(i).getRealValue("x") for i in range(n_events)])

    # bootstrapped_param_list = []
    test_val_dict = {model.name: {} for model in model_primitives}
    for i_boot in range(n_bootstraps):
        bootstrapped_data = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrapped_dataset = ROOT.RooDataSet("bootstrapped_data", "bootstrapped_data", ROOT.RooArgSet(x))
        for val in bootstrapped_data:
            x.setVal(val)
            bootstrapped_dataset.add(ROOT.RooArgSet(x))

        for model_primitive in model_primitives:
            model = model_primitive(x)
            fit_results = emm.fit_random_restarts(
                x, bootstrapped_dataset, model_primitive,
                seed, n_samples=10, n_retries=42,
                save=False,
            )
            fit_result = fit_results[-1]

            # Get predictions at test points
            model.set_params(fit_result["final_pars"])
            fit_result["predictions"] = emm.evaluate_pdf(x, model, test_points)

            test_val_dict[model.name][i_boot] = fit_result

    # Cache results
    output_file = bootstrap_filename(toy_model, seed, n_bootstraps, n_events)
    with open(output_file, "wb") as f:
        pickle.dump(test_val_dict, f)

def run_coverage_tasks(
    x, toy_models, model_primitives,
    seeds, n_bootstraps,
    n_events, test_points,
    use_condor=False,
    remake = False,
    ):
    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            output_file = bootstrap_filename(toy_model, seed, n_bootstraps, n_events)
            if os.path.exists(output_file) and remake:
                os.remove(output_file)
            elif os.path.exists(output_file) and not remake:
                print(f"Coverage results for {toy_model.name} seed {seed} already exist. Skipping.")
                continue

            task = so.Task(
                fit_models_bootstrap,
                x,
                toy_model,
                model_primitives,
                n_bootstraps,
                n_events,
                seed,
                test_points,
            )
            tasks.append(task)

    return so.run_tasks(
        tasks,
        use_condor=use_condor,
        condor_job_name=f"coverage_{n_bootstraps}b_{n_events}e",
        env_wrapper=so.run_in_mamba
    )

def load_result(
        filename: str,
        toy_model_name: str,
        seed: int,
        ) -> dict:
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result, toy_model_name, seed

def load_results(
        toy_models,
        model_primitives,
        seeds,
        n_bootstraps,
        n_events,
        ) -> dict:
    missing_results = 0
    found_results = 0

    results = {
        toy_model.name:{
            model.name: {
                seed: {} for seed in seeds
            } for model in model_primitives
        } for toy_model in toy_models
    }

    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            cache_file = bootstrap_filename(toy_model, seed, n_bootstraps, n_events)
            if not os.path.exists(cache_file):
                missing_results += 1
                continue
            found_results += 1
            task = so.Task(load_result, cache_file, toy_model.name, seed)
            tasks.append(task)

    loaded_results = so.run_tasks(tasks)
    for result, toy_model_name, seed in loaded_results:
        for model_name, fit_results in result.items():
            results[toy_model_name][model_name][seed] = fit_results

    print(f"Loaded results from {found_results} jobs.")
    print(f"Missing results for {missing_results} jobs.")
    return results

def plot_results(
        results: dict,
        true_model_name: str,
        true_pdf_vals: np.ndarray,
        test_points: np.ndarray,
        test_point_idxs: list[int],
        toy_idxs: list[int],
        models_to_plot: list[str] = None,
        n_bins: int = 8,
        model_CI: str = None,
        alpha=0.32,
    ):
 
    # Plot the distribution of the boostrap for several toys and points
    fig, axs = plt.subplots(len(toy_idxs), len(test_point_idxs), figsize=(6 * len(test_point_idxs), 3 * len(toy_idxs)),
    sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(f"Bootstrap PDF value distributions for truth: {true_model_name}", fontsize=16, y=0.91)

    for i_row, toy_idx in enumerate(toy_idxs):
        for j_col, test_point_idx in enumerate(test_point_idxs):
            ax = axs[i_row, j_col] if len(toy_idxs) > 1 else axs[j_col]
            
            # test_val_arr has shape (n_toys, n_bootstraps, n_test_points)
            test_val_arr = results[true_model_name]

            # Get the binning
            min_val = np.inf
            max_val = -np.inf
            for model_name, model_test_val_arr in test_val_arr.items():
                vals = model_test_val_arr[toy_idx, :, test_point_idx]
                min_val = min(min_val, np.nanmin(vals))
                max_val = max(max_val, np.nanmax(vals))
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)

            colors = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
            for i, (model_name, model_test_val_arr) in enumerate(test_val_arr.items()):
                if models_to_plot is not None and model_name not in models_to_plot:
                    continue
                vals = model_test_val_arr[toy_idx, :, test_point_idx]
                ax.hist(vals, bins=bin_edges, histtype='step', label=model_name, color=colors[i])

                if model_CI is not None and model_name == model_CI:
                    lo, hi = np.nanpercentile(vals, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
                    ax.axvline(lo, color=colors[i], linestyle='--', label=f"{model_name} {100*(1 - alpha):.1f}% CI")
                    ax.axvline(hi, color=colors[i], linestyle='--')

            true_val = true_pdf_vals[test_point_idx]
            ax.axvline(true_val, color='k', linestyle='--', label="True value")
            
            # X-label at top and bottom
            xlabel = f"f( x={test_points[test_point_idx]:.2f})"
            if i_row == 0:
                ax.xaxis.set_label_position('top') 
                ax.set_xlabel(xlabel)
            if i_row==len(toy_idxs)-1:
                ax.set_xlabel(xlabel)
        
            # Y-label and legend at left
            if j_col==0:
                ax.set_ylabel(f"Bootstraps for toy {toy_idx}")
                ax.legend(fontsize='small', frameon=False)

def get_bootstrap_coverage(x, x_vals, true_vals, test_models, bootstrapped_param_list, alpha=0.05, print_level=0):

    n_bootstraps = len(bootstrapped_param_list)
    failed_bootstraps = 0
    model_vals = {model.name: [] for model in test_models}

    for bootstrapped_params in bootstrapped_param_list:
        if any(params == {} for params in bootstrapped_params.values()):
            failed_bootstraps += 1
            continue
        for model in test_models:
            params = bootstrapped_params[model.name]
            model_vals[model.name].append(evaluate_model(x, model, params, x_vals)[1])
    
    if print_level > 0:
        print(f"Failed bootstraps: {failed_bootstraps}/{n_bootstraps}\n")

    coverage = {}
    for model in test_models:
        model_vals_array = np.array(model_vals[model.name])
        lower, upper = np.percentile(model_vals_array, [100 * (alpha / 2), 100 * (1 - alpha / 2)], axis=0)
        coverage[model.name] = (true_vals >= lower) & (true_vals <= upper)
    
    return coverage


def get_in_CI(toy_model_name, true_vals, cache_file, alpha=0.05, model_selection=True, n_events=None):
    
    if model_selection and n_events is None:
        raise ValueError("n_events must be provided when model_selection is True.")
    
    with open(cache_file, "rb") as f:
        result = pickle.load(f)
    
    in_CI_dict = {}
    for model_name, bootstrapped_fit_results in result.items():
        if model_selection and "ExponentialMixture" in model_name:
            continue
        bootstrapped_predictions = []
        for i_boot, fit_result in bootstrapped_fit_results.items():
            bootstrapped_predictions.append(fit_result["predictions"])
        bootstrapped_predictions = np.array(bootstrapped_predictions)
        lower, upper = np.percentile(
            bootstrapped_predictions,
            [100 * (alpha / 2), 100 * (1 - alpha / 2)],
            axis=0
            )
        in_ci = (true_vals >= lower) & (true_vals <= upper)
        in_CI_dict[model_name] = in_ci
    
    if model_selection:
        # Also compute for the ExponentialMixtureModel with model selection
        bootstrapped_AICs = {}
        bootstrapped_BICs = {}
        bootstrapped_predictions_AIC = {}
        bootstrapped_predictions_BIC = {}
        for model_name, bootstrapped_fit_results in result.items():
            if "ExponentialMixture" not in model_name:
                continue
            for i_boot, fit_result in bootstrapped_fit_results.items():
                if i_boot not in bootstrapped_AICs:
                    bootstrapped_AICs[i_boot] = np.inf
                if i_boot not in bootstrapped_BICs:
                    bootstrapped_BICs[i_boot] = np.inf

                # Compute AIC and BIC
                k = int(model_name.split("-")[-1])
                n_params = 2*k-1
                aic = 2*n_params + 2*fit_result["nll"]
                bic = n_params * np.log(n_events) + 2*fit_result["nll"]
                if aic < bootstrapped_AICs[i_boot]:
                    bootstrapped_AICs[i_boot] = aic
                    bootstrapped_predictions_AIC[i_boot] = fit_result["predictions"]
                if bic < bootstrapped_BICs[i_boot]:
                    bootstrapped_BICs[i_boot] = bic
                    bootstrapped_predictions_BIC[i_boot] = fit_result["predictions"]
        
        # Compute in_CI for AIC
        names = ["Exponential Mixture (AIC)", "Exponential Mixture (BIC)"]
        predictions = [
            np.array(list(bootstrapped_predictions_AIC.values())),
            np.array(list(bootstrapped_predictions_BIC.values()))
        ]
        for name, preds in zip(names, predictions):
            lower, upper = np.percentile(
                preds,
                [100 * (alpha / 2), 100 * (1 - alpha / 2)],
                axis=0
                )
            in_ci = (true_vals >= lower) & (true_vals <= upper)
            in_CI_dict[name] = in_ci

    return in_CI_dict, toy_model_name

def get_coverages(
    toy_models,
    model_primitives,
    seeds,
    n_bootstraps,
    n_events,
    true_pdf_vals,
    model_selection=True,
    alpha=0.05,
    ):
    
    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            cache_file = bootstrap_filename(toy_model, seed, n_bootstraps, n_events)
            if not os.path.exists(cache_file):
                continue
            task = so.Task(
                get_in_CI,
                toy_model.name,
                true_pdf_vals[toy_model.name],
                cache_file,
                alpha,
                model_selection=model_selection,
                n_events=n_events,)
            tasks.append(task)
    
    results = so.run_tasks(tasks)
    in_CI_results = {tm.name: {} for tm in toy_models}
    for in_CI_dict, toy_model_name in results:
        for model_name, in_CI in in_CI_dict.items():
            if model_name not in in_CI_results[toy_model_name]:
                in_CI_results[toy_model_name][model_name] = []
            in_CI_results[toy_model_name][model_name].append(in_CI)
    
    coverages = {tm.name: {} for tm in toy_models}
    for toy_model_name, test_model_dict in in_CI_results.items():
        for test_model_name, in_CI_list in test_model_dict.items():
            in_CI_array = np.array(in_CI_list)
            coverage = np.mean(in_CI_array, axis=0)
            coverages[toy_model_name][test_model_name] = coverage

    return coverages
            

def plot_coverages(
        coverages, test_points,
        alpha=0.05,
        y_min=0.4,
        models_to_skip=None,
        range: tuple = None,
        skip_every: int = 1,
        linewidth=3.5,
        labelsize=16,
        fontsize=18,
    ):

    if range is not None:
        mask = (test_points >= range[0]) & (test_points <= range[1])
    else:
        mask = np.ones_like(test_points, dtype=bool)

    if skip_every > 1:
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[np.arange(0, len(mask), skip_every)] = True
        mask = mask & new_mask

    fig, axs = plt.subplots(
        len(coverages), 1,
        figsize=(15, 4 * len(coverages)),
        sharex=True, gridspec_kw={'hspace': 0})

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#984ea3', '#a65628',
                '#999999', '#e41a1c', '#dede00']
    
    line_styles = ["solid", "dashed", "dotted", "dashdot"]

    for i, (toy_model_name, test_model_dict) in enumerate(coverages.items()):
        ax = axs[i] if len(coverages) > 1 else axs

        # Print the true model name in the top left corner of the plot
        ax.text(
            0.01, 0.97,
            f"Truth Model: ${toy_model_name}$",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            color=colors[i % len(colors)],
            alpha=1,
        )

        for j, (test_model_name, coverage) in enumerate(test_model_dict.items()):
            if models_to_skip is not None and test_model_name in models_to_skip:
                continue
            alpha_line = 0.3
            line_style=line_styles[0]
            label = test_model_name
            if "_" in label:
                label = f"${label}$"
            if test_model_name == toy_model_name:
                # label += " (true model)"
                alpha_line = 1.0
            if "Exponential Mixture" in test_model_name:
                alpha_line = 1.0
                if "AIC" in test_model_name:
                    line_style = line_styles[1]
                else:
                    line_style = line_styles[2]
            ax.plot(
                test_points[mask],
                100*coverage[mask],
                label=label,
                color=colors[j % len(colors)],
                linestyle=line_style,
                alpha=alpha_line,
                linewidth=linewidth,
                )
        ax.axhline(
            100*(1 - alpha),
            color='k',
            linestyle='--',
            alpha=0.75,
            linewidth=linewidth
        )
        # ax.set_ylabel(f"Coverage for {toy_model_name} truth")
        ax.set_ylabel("Coverage (C) [%]", fontsize=fontsize)
        # Make the y-tick labels larger
        ax.tick_params(axis='y', labelsize=labelsize)

        ax.set_ylim(100*y_min, 100)
        if i == 0:
            # Customize legend: organize f_1-f_4 in 2x2 grid, then exponential mixtures
            handles, labels = ax.get_legend_handles_labels()
            f_models = [h for h, l in zip(handles, labels) if l.startswith('$f_')]
            exp_models = [h for h, l in zip(handles, labels) if 'Exponential' in l]
            
            # Create custom legend with f_1-f_4 in 2x2 grid, then exponential mixtures
            all_handles = f_models + exp_models
            all_labels = [l for h, l in zip(handles, labels) if l.startswith('$f_') or 'Exponential' in l]
            
            legend = ax.legend(
                all_handles, all_labels,
                framealpha=0, 
                loc=(0.2, 0),
                fontsize=fontsize,
                ncol=2  # Use 2 columns for compact layout
            )
            # Set legend line alpha to 1 (full opacity) for legend display only
            for line in legend.get_lines():
                line.set_alpha(1.0)

        # Remove 1.0 from yticks so it doesn't overlap with the other plot
        yticks = ax.get_yticks().tolist()
        yticks = yticks[1:-1]
        ax.set_yticks(yticks)
    
    axs[-1].set_xlabel("$m_{\gamma\gamma}$ [GeV]", fontsize=fontsize)
    axs[-1].tick_params(axis='x', labelsize=labelsize)