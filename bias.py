import gc
import os
import pickle

import ROOT
import numpy as np

import matplotlib.pyplot as plt

import emm
import model_selection as ms

from tools import storage
from tools import scale_out as so

bias_cache = storage.ensure_cache("bias")
def bias_fits_filename(toy_model, seed, n_toys):
    return f"{bias_cache}/{toy_model.name}_seed{seed}_{n_toys}toys.pkl"

def fit_models(x, toy_model, model_primitives, seed, n_toys, n, grid) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    all_fit_results = {model.name: {} for model in model_primitives}
    for itoy in range(n_toys):
        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        # Fit models to toy data
        for model_primitive in model_primitives:
            model = model_primitive(x)
            # fit_result = model.pdf.fitTo(toy_data, PrintLevel=-1, Save=True)
            fit_results = emm.fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_samples=20, n_retries=42,
                save=False,
            )

            # Extract the best fit result and predictions
            best_fit_result = fit_results[-1]
            model = model_primitive(x)
            model.set_params(best_fit_result["final_pars"])
            best_fit_result["predictions"] = emm.evaluate_pdf(x, model, grid)
            #

            all_fit_results[model.name][itoy] = best_fit_result

    # Save result to cache
    cache_file = bias_fits_filename(toy_model, seed, n_toys)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)
    return all_fit_results

def load_results(toy_model, seeds, n_toys_per_seed):
    results = {}
    for seed in seeds:
        cache_file = bias_fits_filename(toy_model, seed, n_toys_per_seed)
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist. Skipping.")
            continue
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        for model_name, fit_results in result.items():
            if model_name not in results:
                results[model_name] = {}
            for i_dataset, dataset_results in fit_results.items():
                uid = f"{seed}_{i_dataset}"
                results[model_name][uid] = dataset_results
    return results

def evaluate_model(x, model, params, x_vals):
    model = model(x)
    model.set_params(params)
    return model.name, emm.evaluate_pdf(x, model, x_vals)

def bias_fits_CV_filename(toy_model, seed, n_toys, n_folds):
    return f"{bias_cache}/{toy_model.name}_seed{seed}_{n_toys}toys_CV{n_folds}.pkl"

def fit_models_CV(
        x, toy_model, model_primitives,
        seed, n_toys, n, grid, n_folds, CV_seed=42) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    all_fit_results = {model.name: {} for model in model_primitives}
    for itoy in range(n_toys):

        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        # Do n_folds cross validation
        train_datasets, test_datasets = ms.train_test_split(x, toy_data, n_folds, seed=CV_seed)

        # Fit models to toy data
        for model_primitive in model_primitives:
            model = model_primitive(x)
            fit_results = emm.fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_samples=8, n_retries=42,
                save=False,
            )

            # Extract the best fit result and predictions
            best_fit_result = fit_results[-1]
            model = model_primitive(x)
            model.set_params(best_fit_result["final_pars"])
            best_fit_result["predictions"] = emm.evaluate_pdf(x, model, grid)
            #

            # Get the CV log-likelihoods
            cv_nlls = []
            for i_fold, train_dataset, test_dataset in zip(range(n_folds), train_datasets, test_datasets):
                cv_fit_results = emm.fit_random_restarts(
                    x, train_dataset, model_primitive,
                    seed, n_samples=20, n_retries=42,
                    save=False,
                )
                model.set_params(cv_fit_results[-1]["final_pars"])
                nll = model.pdf.createNLL(test_dataset)
                cv_nlls.append(nll.getVal())

            best_fit_result["cv_ll"] = -np.sum(cv_nlls)
            all_fit_results[model.name][itoy] = best_fit_result

    # Save result to cache
    cache_file = bias_fits_CV_filename(toy_model, seed, n_toys, n_folds)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)
    return all_fit_results

def load_results_CV(toy_model, seeds, n_toys_per_seed, n_folds):
    results = {}
    for seed in seeds:
        cache_file = bias_fits_CV_filename(toy_model, seed, n_toys_per_seed, n_folds)
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist. Skipping.")
            continue
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        for model_name, fit_results in result.items():
            if model_name not in results:
                results[model_name] = {}
            for i_dataset, dataset_results in fit_results.items():
                uid = f"{seed}_{i_dataset}"
                results[model_name][uid] = dataset_results
    return results

# line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
line_styles = ['solid']

def plot_bias(
    x, toy_model, fit_results, x_vals,
    ax=None,
    abs_bias=False,
    n_cores=16,
    range=None,
    models_to_skip=None
    ):
    
    true_vals = emm.evaluate_pdf(x, toy_model, x_vals)
    model_vals = {model_name: [] for model_name in list(fit_results.keys())}
    for model_name, model_fit_results in fit_results.items():
        for i_dataset, fit_result in model_fit_results.items():
            model_vals[model_name].append(fit_result["predictions"])

    # Convert lists to arrays
    for model_name in model_vals:
        model_vals[model_name] = np.array(model_vals[model_name])
    
    if range is not None:
        mask = (x_vals >= range[0]) & (x_vals <= range[1])
        x_vals = x_vals[mask]
        true_vals = true_vals[mask]
        for model_name in model_vals:
            model_vals[model_name] = model_vals[model_name][:, mask]

    # fn = lambda val: 100*abs(val - true_vals)/true_vals
    fn = lambda val: (val- true_vals)

    # if abs_bias:
    #     bias_fn = lambda val: 100*abs(val - true_vals)/true_vals
    #     ylabel = f"% Absolute Deviation"
    # else:
    #     bias_fn = lambda val: 100*(val - true_vals)/true_vals
    #     ylabel = f"% Deviation"

    model_mean = {}
    model_std = {}

    for model_name, model_vals in model_vals.items():
        # low, med, hi = np.percentile(model_vals, [2.5, 50, 97.5], axis=0)
        model_mean[model_name] = np.mean(model_vals, axis=0)
        model_std[model_name] = np.std(model_vals, axis=0)

    # Use the standard deviation of the true model for normalization
    fn = lambda val: (val-true_vals)/model_std[toy_model.name]
    fn_label = "Bias (D)"

    # Relative deviation
    # fn = lambda val: 100*(val - true_vals)/true_vals
    # fn_label = "Relative Deviation [%]"

    # model_avg_vals = {name: np.mean(vals, axis=0) for name, vals in model_vals.items()}
    
    # model_bias = {name: 100*(model_avg_vals[name] - true_vals)/true_vals for name in model_avg_vals}
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#984ea3', '#a65628',
                '#999999', '#e41a1c', '#dede00']

    line_styles = ['solid', 'dashed', 'dotted', 'dashdot']

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_mean.keys()):
        if models_to_skip is not None and model_name in models_to_skip:
            continue
        label = model_name
        line_style = line_styles[0]
        alpha = 0.3
        if "ExponentialMixture" not in model_name:
            label = f"${model_name}$"

        if model_name == toy_model.name:
            label += " (true model)"
            line_style = line_styles[0]
            alpha = 1.0
            
        if "ExponentialMixture" in model_name:
            alpha = 1.0
            if "AIC" in model_name:
                line_style = line_styles[1]
            else:
                line_style = line_styles[2]
        ax.plot(
            x_vals, fn(model_mean[model_name]),
            label=label,
            color=colors[i],
            # linestyle=line_styles[i % len(line_styles)],
            linestyle=line_style,
            linewidth=2.5,
                alpha=alpha,
                
        )
        if model_name == toy_model.name:# or model_name.startswith("ExponentialMixture"):
            ax.fill_between(
                x_vals,
                fn(model_mean[model_name] - model_std[model_name]),
                fn(model_mean[model_name] + model_std[model_name]),
                alpha=0.3, color=colors[i], )

    if abs_bias:
        ax.set_yscale("log")
    else:
        ax.axhline(0, color='black', linestyle='--')
    ax.set_xlabel("x")
    # ax.set_ylabel(f"Relative Deviation")
    ax.set_ylabel(fn_label)
    ax.legend()



# Spurious signal tests
spurious_signal_cache = storage.ensure_cache("spurious_signal")
def signal_fits_filename(toy_model, seed, n_toys):
    return f"{spurious_signal_cache}/{toy_model.name}_seed{seed}_{n_toys}toys_signal_fits.pkl"

def fit_signal_models(x_orig, toy_model, model_primitives, seed, n_toys, n, grid) -> list[dict]:

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    all_fit_results = {itoy: {sp: {} for sp in grid} for itoy in range(n_toys)}
    for itoy in range(n_toys):
        x = x_orig.clone("x")
        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        for model_primitive in model_primitives:
            # Fit the background model first to stabilize the fit
            bkg_fit_results = emm.fit_random_restarts(
                x, toy_data, model_primitive,
                seed, n_samples=5, n_retries=5,
                save=False,
            )
            if bkg_fit_results is None:
                print(f"Background fit failed for toy {itoy}, model {model_primitive.name}. Skipping.")
                continue

            for sp in grid:
                sig_mean, sig_width = sp
                bkg_model = model_primitive(x)
                bkg_model.set_params(bkg_fit_results[-1]["final_pars"])

                # Get the number of background events within 1 sigma of the signal mean
                x.setRange("sig_range", sig_mean - sig_width, sig_mean + sig_width)
                subset = toy_data.reduce(CutRange="sig_range")
                n_evt_in_sig_region = subset.sumEntries()
                if n_evt_in_sig_region == 0:
                    max_sig = 10
                else:
                    max_sig = 10*np.sqrt(n_evt_in_sig_region)

                sig_model = emm.GaussianSignalModel(x, sig_mean, sig_width)
                model = emm.SignalPlusBackgroundModel(sig_model, bkg_model, max_sig=max_sig)
                result = emm.fit_n_times(
                    model, toy_data, n_attempts=5,
                    fit_options=[ROOT.RooFit.RecoverFromUndefinedRegions(1.0)]
                )
                if result is None:
                    print(f"Fit failed for toy {itoy}, signal point {sp}, model {model_primitive.name}. Skipping.")
                    continue

                # Save relevant info
                fit_result = {
                    "n_sig": model.n_sig.getVal(),
                    # "n_bkg": model.n_bkg.getVal(),
                    "bkg_model_nll": bkg_fit_results[-1]["nll"],
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
    cache_file = signal_fits_filename(toy_model, seed, n_toys)
    with open(cache_file, "wb") as f:
        pickle.dump(all_fit_results, f)

    return all_fit_results

def load_signal_fit_results(toy_model, seeds, n_toys_per_seed):
    results = {}
    for seed in seeds:
        cache_file = signal_fits_filename(toy_model, seed, n_toys_per_seed)
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist. Skipping.")
            continue
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        for itoy, sp_results in result.items():
            uid = f"{seed}_{itoy}"
            results[uid] = sp_results
    return results