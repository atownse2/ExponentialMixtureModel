import gc
import os
import pickle

import ROOT
import numpy as np

import matplotlib.pyplot as plt

from .models import (
    evaluate_pdf,
    GaussianSignalModel,
    SignalPlusBackgroundModel,
)
from .fitting import fit_random_restarts, fit_n_retries, train_test_split

from tools import storage
from tools import scale_out as so

# Bias studies
bias_cache = storage.ensure_cache("bias")
def get_bias_fits_cache_path(toy_model, seed, n_toys):
    return f"{bias_cache}/{toy_model.name}_seed{seed}_{n_toys}toys.pkl"

def run_bias_fits(
        x, toy_model, model_primitives,
        seed, n_toys, n, grid,
        n_restarts=50, n_retries=20
    ) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    model_names = [mp.name for mp in model_primitives]
    all_fit_results = {model_name: {} for model_name in model_names}
    for itoy in range(n_toys):
        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        # Fit models to toy data
        for model_primitive in model_primitives:
            model = model_primitive(x)
            fit_result = fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_restarts=20, n_retries=42,
                save=False,
            )

            # Extract the best fit result and predictions
            model = model_primitive(x) # Re-instantiate the model to avoid any side effects from previous fits
            model.set_params(fit_result["final_pars"])
            fit_result["predictions"] = evaluate_pdf(x, model, grid)
            #

            all_fit_results[model.name][itoy] = fit_result

    # Save result to cache
    cache_file = get_bias_fits_cache_path(toy_model, seed, n_toys)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)
    return all_fit_results

def load_and_format_bias_results(
    toy_model_name,
    cache_file,
    n_events=None,
    model_selection=True,
):

    if model_selection and n_events is None:
        raise ValueError("n_events must be provided when model_selection is True.")

    seed = os.path.basename(cache_file).split("_seed", maxsplit=1)[1].split("_", maxsplit=1)[0]

    with open(cache_file, "rb") as f:
        result = pickle.load(f)

    fit_results = {}
    for model_name, model_fit_results in result.items():
        if model_selection and "ExponentialMixture" in model_name:
            continue
        fit_results[model_name] = {
            f"{seed}_{i_dataset}": dataset_results
            for i_dataset, dataset_results in model_fit_results.items()
        }

    if model_selection:
        exponential_mixture_results = {
            model_name: model_fit_results
            for model_name, model_fit_results in result.items()
            if "ExponentialMixture" in model_name
        }
        if len(exponential_mixture_results) > 0:
            fit_results["ExponentialMixture (AIC)"] = {}
            fit_results["ExponentialMixture (BIC)"] = {}
            dataset_ids = sorted(
                {
                    i_dataset
                    for model_fit_results in exponential_mixture_results.values()
                    for i_dataset in model_fit_results
                }
            )
            for i_dataset in dataset_ids:
                best_AIC = np.inf
                best_BIC = np.inf
                for model_name, model_fit_results in exponential_mixture_results.items():
                    if i_dataset not in model_fit_results:
                        continue

                    fit_result = model_fit_results[i_dataset]
                    k = int(model_name.split("-")[-1])
                    n_params = 2 * k - 1
                    aic = 2 * n_params + 2 * fit_result["nll"]
                    bic = n_params * np.log(n_events) + 2 * fit_result["nll"]
                    uid = f"{seed}_{i_dataset}"
                    if aic < best_AIC:
                        best_AIC = aic
                        fit_results["ExponentialMixture (AIC)"][uid] = fit_result
                    if bic < best_BIC:
                        best_BIC = bic
                        fit_results["ExponentialMixture (BIC)"][uid] = fit_result

    return fit_results, toy_model_name


def get_bias_results(
    toy_models,
    seeds,
    n_toys_per_seed,
    n_events,
    model_selection=True,
):

    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            cache_file = get_bias_fits_cache_path(toy_model, seed, n_toys_per_seed)
            if not os.path.exists(cache_file):
                continue
            task = so.Task(
                load_and_format_bias_results,
                toy_model.name,
                cache_file,
                n_events=n_events,
                model_selection=model_selection,
            )
            tasks.append(task)

    results = so.run_tasks(tasks)
    bias_results = {toy_model.name: {} for toy_model in toy_models}
    for fit_results, toy_model_name in results:
        for model_name, model_fit_results in fit_results.items():
            if model_name not in bias_results[toy_model_name]:
                bias_results[toy_model_name][model_name] = {}
            bias_results[toy_model_name][model_name].update(model_fit_results)

    return bias_results

# Cross-validation tools
def get_bias_fits_CV_cache_path(toy_model, seed, n_toys, n_folds):
    return f"{bias_cache}/{toy_model.name}_seed{seed}_{n_toys}toys_CV{n_folds}.pkl"

def run_bias_fits_CV(
        x, toy_model, model_primitives,
        seed, n_toys, n, grid, n_folds, CV_seed=42) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    model_names = [mp.name for mp in model_primitives]
    all_fit_results = {model_name: {} for model_name in model_names}
    for itoy in range(n_toys):

        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        # Do n_folds cross validation
        train_datasets, test_datasets = train_test_split(x, toy_data, n_folds, seed=CV_seed)

        # Fit models to toy data
        for model_primitive in model_primitives:
            model = model_primitive(x)
            fit_result = fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_restarts=8, n_retries=42,
                save=False,
            )

            # Extract the best fit result and predictions
            model = model_primitive(x)
            model.set_params(fit_result["final_pars"])
            fit_result["predictions"] = evaluate_pdf(x, model, grid)
            #

            # Get the CV log-likelihoods
            cv_nlls = []
            for i_fold, train_dataset, test_dataset in zip(range(n_folds), train_datasets, test_datasets):
                cv_fit_result = fit_random_restarts(
                    x, train_dataset, model_primitive,
                    seed, n_restarts=20, n_retries=42,
                    save=False,
                )
                model.set_params(cv_fit_result["final_pars"])
                nll = model.pdf.createNLL(test_dataset)
                cv_nlls.append(nll.getVal())

            fit_result["cv_ll"] = -np.sum(cv_nlls)
            all_fit_results[model.name][itoy] = fit_result

    # Save result to cache
    cache_file = get_bias_fits_CV_cache_path(toy_model, seed, n_toys, n_folds)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)
    return all_fit_results

def _compute_bias_summary(x, toy_model, fit_results, x_vals, x_range=None):
    true_vals = evaluate_pdf(x, toy_model, x_vals)
    model_vals = {model_name: [] for model_name in list(fit_results.keys())}
    for model_name, model_fit_results in fit_results.items():
        for _, fit_result in model_fit_results.items():
            model_vals[model_name].append(fit_result["predictions"])

    for model_name in model_vals:
        model_vals[model_name] = np.array(model_vals[model_name])

    if x_range is not None:
        mask = (x_vals >= x_range[0]) & (x_vals <= x_range[1])
        x_vals = x_vals[mask]
        true_vals = true_vals[mask]
        for model_name in model_vals:
            model_vals[model_name] = model_vals[model_name][:, mask]

    model_mean = {}
    model_std = {}
    for model_name, vals in model_vals.items():
        model_mean[model_name] = np.mean(vals, axis=0)
        model_std[model_name] = np.std(vals, axis=0)

    return x_vals, true_vals, model_mean, model_std

# Plotting utilities
def _plot_bias_single_axis(
    x,
    ax,
    toy_model,
    fit_results,
    x_vals,
    abs_bias=False,
    x_range=None,
    models_to_skip=None,
    linewidth=3.5,
    fontsize=18,
    labelsize=16,
):
    x_vals, true_vals, model_mean, model_std = _compute_bias_summary(
        x, toy_model, fit_results, x_vals, x_range=x_range
    )

    denom = model_std.get(toy_model.name)
    if denom is None:
        raise ValueError(f"Toy model {toy_model.name} missing from fit_results keys")
    denom = np.where(denom == 0, np.nan, denom)
    fn = lambda val: (val - true_vals) / denom

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#984ea3', '#a65628',
              '#999999', '#e41a1c', '#dede00']
    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']

    for i, model_name in enumerate(model_mean.keys()):
        if models_to_skip is not None and model_name in models_to_skip:
            continue

        label = model_name if "ExponentialMixture" in model_name else f"${model_name}$"
        line_style = line_styles[0]
        alpha = 0.3

        if model_name == toy_model.name:
            alpha = 1.0
        if "ExponentialMixture" in model_name:
            alpha = 1.0
            line_style = line_styles[1] if "AIC" in model_name else line_styles[2]

        ax.plot(
            x_vals,
            fn(model_mean[model_name]),
            label=label,
            color=colors[i % len(colors)],
            linestyle=line_style,
            linewidth=linewidth,
            alpha=alpha,
        )

        if model_name == toy_model.name:
            ax.fill_between(
                x_vals,
                fn(model_mean[model_name] - model_std[model_name]),
                fn(model_mean[model_name] + model_std[model_name]),
                alpha=0.4,
                color=colors[i % len(colors)],
            )

    if abs_bias:
        ax.set_yscale("log")
    else:
        ax.axhline(0, color='black', linestyle='--', linewidth=linewidth, alpha=0.7)
    ax.set_ylabel("Relative Bias (B)", fontsize=fontsize)
    # Make the y-tick labels larger
    ax.tick_params(axis='y', labelsize=labelsize)

def plot_bias(
    x,
    toy_models,
    fit_results,
    x_vals,
    ax=None,
    abs_bias=False,
    n_cores=16,
    range=None,
    models_to_skip=None,
    linewidth=3.5,
    legend_loc=(0.25, 0),
    labelsize=16,
    fontsize=18,
):
    # Backward compatible path: a single toy model + fit results for just that model.
    if isinstance(toy_models, (list, tuple)):
        toy_model_list = list(toy_models)
        fit_results_by_toy = fit_results
    else:
        toy_model_list = [toy_models]
        if toy_models.name in fit_results and isinstance(fit_results[toy_models.name], dict):
            fit_results_by_toy = {toy_models.name: fit_results[toy_models.name]}
        else:
            fit_results_by_toy = {toy_models.name: fit_results}

    if ax is None:
        fig, axs = plt.subplots(
            len(toy_model_list),
            1,
            figsize=(15, 4 * len(toy_model_list)),
            sharex=True,
            gridspec_kw={'hspace': 0},
        )
    else:
        axs = ax

    if len(toy_model_list) == 1:
        axs = [axs]

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#984ea3', '#a65628',
              '#999999', '#e41a1c', '#dede00']

    for i, toy_model in enumerate(toy_model_list):
        axis = axs[i]
        axis.text(
            0.8,
            0.9,
            f"Truth Model: ${toy_model.name}$",
            transform=axis.transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            horizontalalignment='right',
            color=colors[i % len(colors)],
        )

        if toy_model.name not in fit_results_by_toy:
            print(f"Toy model {toy_model.name} not found in fit results. Skipping.")
            continue

        _plot_bias_single_axis(
            x,
            axis,
            toy_model,
            fit_results_by_toy[toy_model.name],
            x_vals,
            abs_bias=abs_bias,
            x_range=range,
            models_to_skip=models_to_skip,
            linewidth=linewidth,
            fontsize=fontsize,
            labelsize=labelsize,
        )

    handles, labels = axs[0].get_legend_handles_labels()
    f_models = [h for h, l in zip(handles, labels) if l.startswith('$f_')]
    exp_models = [h for h, l in zip(handles, labels) if 'Exponential' in l]
    all_handles = f_models + exp_models
    all_labels = [l for _, l in zip(handles, labels) if l.startswith('$f_') or 'Exponential' in l]

    legend = axs[0].legend(
        all_handles,
        all_labels,
        framealpha=0,
        loc=legend_loc,
        fontsize=fontsize,
        ncol=2,
    )
    for line in legend.get_lines():
        line.set_alpha(1.0)

    if len(toy_model_list) > 1:
        axs[-1].set_xlabel("$m_{\\gamma\\gamma}$ [GeV]", fontsize=fontsize)
        axs[-1].tick_params(axis='x', labelsize=labelsize)
    else:
        axs[0].set_xlabel("$m_{\\gamma\\gamma}$ [GeV]", fontsize=fontsize)
        axs[0].tick_params(axis='x', labelsize=labelsize)

    return axs

# Spurious signal tests
spurious_signal_cache = storage.ensure_cache("spurious_signal")
def get_spurious_signal_fits_cache_path(toy_model, seed, n_toys):
    return f"{spurious_signal_cache}/{toy_model.name}_seed{seed}_{n_toys}toys_signal_fits.pkl"

def run_spurious_signal_fits(x_orig, toy_model, model_primitives, seed, n_toys, n, grid) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    all_fit_results = {itoy: {sp: {} for sp in grid} for itoy in range(n_toys)}
    for itoy in range(n_toys):
        x = x_orig.clone("x")
        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        for model_primitive in model_primitives:
            # Fit the background model first to stabilize the fit
            bkg_fit_result = fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_restarts=5, n_retries=5,
                save=False,
            )
            if bkg_fit_result is None:
                print(f"Background fit failed for toy {itoy}, model {model_primitive.name}. Skipping.")
                continue

            for sp in grid:
                sig_mean, sig_width = sp
                bkg_model = model_primitive(x)
                bkg_model.set_params(bkg_fit_result["final_pars"])

                # Get the number of background events within 1 sigma of the signal mean
                x.setRange("sig_range", sig_mean - sig_width, sig_mean + sig_width)
                subset = toy_data.reduce(CutRange="sig_range")
                n_evt_in_sig_region = subset.sumEntries()
                if n_evt_in_sig_region == 0:
                    max_sig = 10
                else:
                    max_sig = 10*np.sqrt(n_evt_in_sig_region)

                sig_model = GaussianSignalModel(x, sig_mean, sig_width)
                model = SignalPlusBackgroundModel(sig_model, bkg_model, max_sig=max_sig)
                result = fit_n_retries(
                    model, toy_data, n_attempts=5,
                    fit_options=[ROOT.RooFit.RecoverFromUndefinedRegions(1.0)]
                )
                if result is None:
                    print(f"Fit failed for toy {itoy}, signal point {sp}, model {model_primitive.name}. Skipping.")
                    continue

                # Save relevant info
                fit_result = {
                    "n_sig": model.get_param("n_sig").getVal(),
                    "n_sig_err": model.get_param("n_sig").getError(),
                    # "n_bkg": model.n_bkg.getVal(),
                    "bkg_model_nll": bkg_fit_result["nll"],
                    "sig_plus_bkg_nll": result.minNll(),
                }
                all_fit_results[itoy][sp][bkg_model.name] = fit_result

                # --- EXPLICIT CLEANUP (Inner Loop) ---
                del subset
                del result
                del sig_model
                del bkg_model
                del model

        # --- EXPLICIT CLEANUP (Outer Loop) ---
        del toy_data
        del x
        
        # Periodically force Python to collect garbage to ensure C++ destructors fire
        if itoy % 10 == 0:
            gc.collect()

    # Save result to cache
    cache_file = get_spurious_signal_fits_cache_path(toy_model, seed, n_toys)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)

    return all_fit_results

def load_spurious_signal_fit_results(toy_model, seeds, n_toys_per_seed):
    results = {}
    for seed in seeds:
        cache_file = get_spurious_signal_fits_cache_path(toy_model, seed, n_toys_per_seed)
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist. Skipping.")
            continue
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        for itoy, sp_results in result.items():
            uid = f"{seed}_{itoy}"
            results[uid] = sp_results
    return results