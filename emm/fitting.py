import time
import random
import pickle

import numpy as np

import ROOT
from array import array

from tools import storage

random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

fit_options = [
    # ROOT.RooFit.IntegrateBins(0.0001),
    # ROOT.RooFit.PrintLevel(-1),
    # ROOT.RooFit.Offset(True),
    # # ROOT.RooFit.Strategy(2),
    # ROOT.RooFit.Save(),
    # # ROOT.RooFit.Range("fit_range")
]

def fit(model, data, fit_args=[], print_level=0):
    if print_level > 0:
        print(f"Fitting model {model.name} to data {data.GetName()} with {len(data)} entries")
        t1 = time.time()

    fit_result = model.pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        *fit_args
    )

    if print_level > 0:
        t2 = time.time()
        print(f"Fitted in {t2 - t1:.2f} seconds")
        print(f"Fit status: {fit_result.status()}, covQual: {fit_result.covQual()}")
        print(f"NLL: {fit_result.minNll()}")

    return fit_result

def fit_n_times(model, data, n_attempts=5, fit_options=fit_options, print_level=0):
    """
    Fit a model to data, retrying up to n_attempts times if the fit fails
    (fit status > 2). Returns the fit result if successful, or None if all attempts fail.
    """
    import ROOT
    if ROOT.RooFit.Save(True) not in fit_options:
        fit_options.append(ROOT.RooFit.Save(True))
    for attempt in range(n_attempts):
        fit_result = model.pdf.fitTo(
            data,
            *fit_options
        )
        if fit_result.status() <= 2:
            if attempt > 0 and print_level > 0:
                print(f"Fit succeeded after {attempt} retries")
            return fit_result
    if print_level > 0:
        print(f"Fit failed after {n_attempts} attempts, returning None" )
    return None

# Random restarts:
fit_cache = storage.ensure_cache("fits")
def random_restarts_filename(
        model_name,
        data_name,
        n_samples,
        seed,
        ):
    tags = f"{model_name}_{data_name}_nrestarts{n_samples}_seed{seed}.pkl"
    f = f"{fit_cache}/{tags}"
    return f

def fit_random_restarts(
        x, data,
        model_primitive,
        seed, n_samples,
        n_retries=5,
        save=True,
        fit_options=fit_options,
        print_level=0,
        return_all_results=False
    ):

    best_nll = np.inf
    fit_results = []

    rng = np.random.default_rng(seed=seed)
    for i in range(n_samples):
        model = model_primitive(x)
        model.randomize_params(rng=rng)
        initial_pars = {p.GetName(): p.getVal() for p in model.params()}

        fit_result = fit_n_times(model, data, n_attempts=n_retries, fit_options=fit_options, print_level=print_level)
        if fit_result is None:
            if print_level > 0:
                print(f"Random Restart {i+1}/{n_samples}: Fit failed after {n_retries} attempts.")
            continue

        if fit_result.status() <= 2 and fit_result.minNll() < best_nll:
            best_nll = fit_result.minNll()
            fit_result = {
                "nll": best_nll,
                "initial_pars": initial_pars,
                "final_pars": {p.GetName(): p.getVal() for p in model.params()}
                }
            fit_results.append(fit_result)
        
        del model  # Free memory

    if len(fit_results) == 0:
        print("No successful fits were found.")
        return None
    
    if print_level > 0:
        print(f"Random Restarts: {len(fit_results)} successful fits out of {n_samples} attempts.")
        print(f"Best NLL: {best_nll:.3f}")

    if save:
        model = model_primitive(x)
        fout = random_restarts_filename(model.name, data.GetName(), n_samples, seed)
        with open(fout, "wb") as f:
            pickle.dump(fit_results, f)

    if return_all_results:
        return fit_results
    else:
        # Return the best fit result
        best_fit_result = min(fit_results, key=lambda r: r["nll"])
        return best_fit_result

# Goodness-of-fit metrics
def compute_information_criteria(nll, n_params, n_observations):
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")

    return {
        "AIC": 2 * n_params + 2 * nll,
        "BIC": n_params * np.log(n_observations) + 2 * nll,
    }

def rebin_for_low_stats(hist, min_events=30):
    """
    Dynamically merges adjacent bins of a TH1 histogram until every bin 
    contains at least `min_events`. Returns a new variable-bin-width TH1.
    """
    edges = [hist.GetBinLowEdge(1)]
    current_events = 0
    
    for b in range(1, hist.GetNbinsX() + 1):
        current_events += hist.GetBinContent(b)
        # If the accumulated events hit the threshold, seal the bin edge
        if current_events >= min_events:
            edges.append(hist.GetBinLowEdge(b) + hist.GetBinWidth(b))
            current_events = 0
            
    # Handle leftovers: merge remaining events into the final bin
    if current_events > 0 and len(edges) > 1:
        edges[-1] = hist.GetBinLowEdge(hist.GetNbinsX()) + hist.GetBinWidth(hist.GetNbinsX())
    elif len(edges) == 1:
        # Fallback if the entire histogram has fewer than min_events
        edges.append(hist.GetBinLowEdge(hist.GetNbinsX()) + hist.GetBinWidth(hist.GetNbinsX()))
        
    # Convert list to a C-style double array for ROOT
    edges_arr = array('d', edges)
    
    # TH1::Rebin handles the recalculation of contents and SumW2 errors automatically
    rebinned_hist = hist.Rebin(len(edges_arr) - 1, f"{hist.GetName()}_rebinned", edges_arr)
    return rebinned_hist

def chi2(x, hist, model, min_events=5, print_chi2=False, integrate_bins_precision=1e-3):
    """
    Compute chi2 for a 1D RooDataHist after merging adjacent bins
    until each combined bin contains at least min_events observed entries.
    """
    if min_events < 0:
        raise ValueError("min_events must be non-negative")

    if integrate_bins_precision is not None and integrate_bins_precision <= 0:
        raise ValueError("integrate_bins_precision must be positive or None")

    if not hasattr(model, "pdf") or not hasattr(model, "params"):
        raise TypeError("model must provide pdf and params()")

    pdf = model.pdf

    # Rebin the histogram to ensure at least min_events per bin
    rebinned_hist = rebin_for_low_stats(hist, min_events=min_events)
    n_combined_bins = rebinned_hist.GetNbinsX()
    ndf = n_combined_bins - len(model.params())

    rebinned_data = ROOT.RooDataHist(
        f"{rebinned_hist.GetName()}_rebinned",
        f"{rebinned_hist.GetTitle()} (rebinned)",
        x,
        rebinned_hist
    )

    chi2_args = []
    if integrate_bins_precision is not None:
        # Integrate the PDF across each rebinned bin. Point sampling at the bin
        # center can strongly overestimate chi2 for steeply falling models.
        chi2_args.append(ROOT.RooFit.IntegrateBins(integrate_bins_precision))

    chi2_var = ROOT.RooChi2Var(
        f"chi2_var_{random_string()}",
        "chi2",
        pdf,
        rebinned_data,
        *chi2_args,
    )
    raw_chi2 = float(chi2_var.getVal())
    result = {
        "chi2": raw_chi2,
        "ndf": ndf,
        "chi2_ndf": raw_chi2 / ndf,
        "n_combined_bins": int(n_combined_bins),
        "min_events": min_events,
        "integrate_bins_precision": integrate_bins_precision,
        "binned_data": rebinned_data,
        "hist": rebinned_hist,
        "bin_edges": rebinned_hist.GetXaxis().GetXbins(),
    }

    if print_chi2:
        print(
            f"chi2={result['chi2']:.2f}, ndf={result['ndf']}, "
            f"chi2/ndf={result['chi2_ndf']:.3f}, "
        )

    return result

# Cross-validation utilities
def train_test_split(x, data, n_folds, seed=1234):
    """Split a RooDataSet into train/test subsets for cross-validation.
    Returns lists of RooDataSets for training and testing, one per fold.
    If n_folds < 1, returns the leave-one-out split where each test set is a single entry.
    """

    import numpy as np

    n = data.numEntries()
    data_arr = np.array([data.get(i).getRealValue(x.GetName()) for i in range(n)])

    if n_folds < 1:
        n_folds = n
    else:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(data_arr)
    data_arr_folds = np.array_split(data_arr, n_folds)

    train_datasets = []
    test_datasets = []
    for i in range(n_folds):
        name = f"{data.GetName()}_{n_folds}folds_{i}"
        test_data_arr = data_arr_folds[i]
        train_data_arr = np.concatenate([data_arr_folds[j] for j in range(n_folds) if j != i])

        test_dataset = ROOT.RooDataSet(name+"_test", name+"_test", ROOT.RooArgSet(x))
        for val in test_data_arr:
            x.setVal(val)
            test_dataset.add(ROOT.RooArgSet(x))
        test_datasets.append(test_dataset)

        train_dataset = ROOT.RooDataSet(name+"_train", name+"_train", ROOT.RooArgSet(x))
        for val in train_data_arr:
            x.setVal(val)
            train_dataset.add(ROOT.RooArgSet(x))
        train_datasets.append(train_dataset)

    return train_datasets, test_datasets