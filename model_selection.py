import os
import pickle

import numpy as np
import pandas as pd

import ROOT
import emm

import multiprocessing as mp
import matplotlib.pyplot as plt

from tools import storage


class SignalPlusBackgroundModel(emm.RooFitModel):

    def __init__(self, signal_model, background_model, **kwargs):
        self.signal_model = signal_model
        self.background_model = background_model
        self.name = f"{signal_model.name}_plus_{background_model.name}"
        
        self.prefix = kwargs.get("prefix", "")

        # Initialize signal strength parameter
        if "signal_strength" in kwargs:
            if isinstance(kwargs["signal_strength"], ROOT.RooRealVar):
                self.signal_strength = kwargs["signal_strength"]
            elif isinstance(kwargs["signal_strength"], (float, int)):
                self.signal_strength = ROOT.RooRealVar(
                    f"{self.prefix}signal_strength",
                    "Signal Strength",
                    float(kwargs["signal_strength"]),
                    0,
                    1
                )
            else:
                raise ValueError("signal_strength must be a RooRealVar or a float/int")
        else:
            self.signal_strength = ROOT.RooRealVar(f"{self.prefix}signal_strength", "Signal Strength", 0.1, 0, 1)
        self.pdf = ROOT.RooGenericPdf(
            f"{self.prefix}{self.name}",
            f"{self.prefix}{self.name}",
            f"{self.signal_strength.GetName()}*{signal_model.pdf.GetName()} + (1 - {self.signal_strength.GetName()})*{background_model.pdf.GetName()}",
            ROOT.RooArgSet(self.signal_strength, signal_model.pdf, background_model.pdf)
        )

    def params(self):
        return [self.signal_strength] + self.signal_model.params() + self.background_model.params()

model_selection_cache = storage.ensure_cache("model_selection")

def AIC(fit_result):
    k = len(fit_result.floatParsFinal())
    nll = fit_result.minNll()
    aic = 2 * k + 2 * nll
    return aic

def BIC(fit_result, n):
    k = len(fit_result.floatParsFinal())
    nll = fit_result.minNll()
    bic = k * np.log(n) + 2 * nll
    return bic

def AIC_BIC_loo(n_components, t, k_folds=-1):

    x = ROOT.RooRealVar("x", "Diphoton Mass [GeV]", 500, 4000)
    index=ROOT.RooRealVar("index", "index", 0, 0, 1e6)

    data = ROOT.RooDataSet("mgg", "mgg", ROOT.RooArgSet(x, index), ROOT.RooFit.Import(t))
    
    model_inst = ExponentialMixtureModel(x, n_components)
    model = model_inst.pdf

    AICs = []
    BICs = []

    if k_folds == -1:
        k_folds = data.numEntries()

    step_size = data.numEntries() // k_folds

    for i in range(k_folds):
        # print(f"Fitting dataset {i+1}/{k_folds}")

        data_loo = data.reduce(f"index<({i*step_size}) || index>=({(i+1)*step_size})")

        # Fit
        fit_result = model.fitTo(data_loo)#, ROOT.RooFit.PrintLevel(-1))

        nll = model.createNLL(data_loo).getVal()
        n_pars = model.getParameters(data_loo).getSize()

        AICs.append(md.AIC(nll, n_pars))
        BICs.append(md.BIC(nll, n_pars, data_loo.numEntries()))

    return {"AIC": AICs, "BIC": BICs}

def get_AIC_BIC_loo(k_max=4, t=None, remake=False, tag="data"):

    aic_bic_file = os.path.join(emm_cache, f"aic_bic_results_{tag}.csv")
    if os.path.exists(aic_bic_file) and not remake:
        df = pd.read_csv(aic_bic_file)
    else:
        if t is None:
            t = get_data()

        n_components = [i for i in range(1, k_max + 1)]
        results = []
        with mp.Pool(len(n_components)) as pool:
            results = pool.starmap(
                AIC_BIC_loo, 
                [(i, t) for i in n_components]
            )

        # Collect results
        df = []
        for i, result in enumerate(results):
            AIC_low, AIC_med, AIC_high = np.percentile(result["AIC"], [16, 50, 84])
            BIC_low, BIC_med, BIC_high = np.percentile(result["BIC"], [16, 50, 84])
            df.append({
                "n_components": n_components[i],
                "AIC_low": AIC_low,
                "AIC_med": AIC_med,
                "AIC_high": AIC_high,
                "BIC_low": BIC_low,
                "BIC_med": BIC_med,
                "BIC_high": BIC_high
            })
        # Convert to DataFrame
        df = pd.DataFrame(df)

        df.to_csv(aic_bic_file, index=False)
    
    return df

def plot_AIC_BIC_loo(df=None):
    if df is None:
        df = get_AIC_BIC_loo(remake=False)
    
    fig, ax = plt.subplots(1,2, figsize=(12, 5))

    ax[0].errorbar(
        df['n_components'],
        df['AIC_med'],
        yerr=[df['AIC_med']-df['AIC_low'], df['AIC_high']-df['AIC_med']],
        capsize=5,
        markersize=10
    )
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("AIC")
    ax[0].set_xticks(df['n_components'])

    ax[1].errorbar(
        df['n_components'],
        df['BIC_med'],
        yerr=[df['BIC_med']-df['BIC_low'], df['BIC_high']-df['BIC_med']],
        capsize=5,
        markersize=10
    )
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("BIC")
    ax[1].set_xticks(df['n_components'])

    # fig.suptitle("Leave-One-Out Cross Validation")
    plt.tight_layout()
    plt.show()

def compare_models(x, data, k_min):
    bkg_model_k = emm.ExponentialMixtureModel(x, k_min, prefix=f"comp_k{k_min}_")
    bkg_model_kplus1 = emm.ExponentialMixtureModel(x, k_min + 1, prefix=f"comp_k{k_min+1}_")

    fit_result_k = bkg_model_k.pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1)
    )
    if fit_result_k.status() > 2:
        return None

    fit_result_kplus1 = bkg_model_kplus1.pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1)
    )
    if fit_result_kplus1.status() > 2:
        return None

    logL_k = fit_result_k.minNll()
    logL_kplus1 = fit_result_kplus1.minNll()

    t = 2 * (logL_k - logL_kplus1)

    return t

def calculate_AD_statistic(x, data, model):
    
    n = data.numEntries()

    # Put data into a numpy array and sort
    data_values = np.array([data.get(i).getRealValue("x") for i in range(n)])
    data_values.sort()

    # Calculate CDF values at data points
    cdf = model.pdf.createCdf(ROOT.RooArgSet(x))
    cdf_values = []
    for val in data_values:
        x.setVal(val)
        cdf_val = cdf.getVal(ROOT.RooArgSet(x))
        cdf_values.append(cdf_val)
    cdf_values = np.array(cdf_values)

    cdf_values_rev = cdf_values[::-1]
    idxs = np.arange(1, n+1)

    S = np.sum((2*idxs - 1) * (np.log(cdf_values) + np.log(1 - cdf_values_rev)))/n

    A2 = -n - S
    return A2


def calculate_KS_statistic(x, data, model):
    pass

def result_filename(k, seed, n_toys):
    tags = f"model_comp_k{k}_seed{seed}_ntoys{n_toys}_AD.pkl"
    f = f"{model_selection_cache}/{tags}"
    return f


def get_toy_distribution(x, data, k, seed, n_toys):
    import numpy as np
    import ROOT

    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    toy_model = emm.ExponentialMixtureModel(x, k, prefix=f"toy_{seed}_")
    toy_model.pdf.fitTo(
        data,
        ROOT.RooFit.PrintLevel(-1)
    )

    t_values = []
    for i in range(n_toys):
        # Generate toy data
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), data.numEntries())
        model_i = emm.ExponentialMixtureModel(x, k, prefix=f"fit_{seed}_{i}_")
        fit_result = model_i.pdf.fitTo(
            toy_data,
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Save(True)
        )
        if fit_result.status() > 2:
            continue

        t = calculate_AD_statistic(x, toy_data, model_i)
        t_values.append(t)

    # return np.array(t_values)
    filename = result_filename(k, seed, n_toys)
    with open(filename, "wb") as f:
        pickle.dump(np.array(t_values), f)

def run_jobs(x, data, ks, seeds, n_toys, remake=False, **run_task_kwargs):
    from tools import scale_out as so

    if remake:
        storage.clear_cache(model_selection_cache)

    tasks = []
    for k in ks:
        for seed in seeds:
            # Check for existing results
            filename = result_filename(k, seed, n_toys)
            if os.path.exists(filename):
                continue

            task = so.Task(
                get_toy_distribution,
                x, data, k, seed, n_toys
            )
            tasks.append(task)
            
    _ = so.run_tasks(tasks, **run_task_kwargs)

def load_results(ks, seeds, n_toys):
    import numpy as np
    results = {}
    for k in ks:
        t_values = []
        for seed in seeds:
            filename = result_filename(k, seed, n_toys)
            if not os.path.exists(filename):
                print(f"Warning: missing file {filename}")
                continue
            with open(filename, "rb") as f:
                t_vals = pickle.load(f)
                t_values.extend(t_vals)
        results[k] = np.array(t_values)
    return results

def bootstrap_bias_filename(k, n_bootstraps, seed):
    tags = f"bootstrap_bias_k{k}_nboot{n_bootstraps}_seed{seed}.pkl"
    f = f"{model_selection_cache}/{tags}"
    return f

def bootstrap_bias(x, data, k, n_bootstraps, seed):
    import numpy as np
    import ROOT

    # Load data into numpy array
    n = data.numEntries()
    data_arr = np.array([data.get(i).getRealValue("x") for i in range(n)])

    Ds = []
    np.random.seed(seed)
    for _ in range(n_bootstraps):
        bootstrapped_data = np.random.choice(data_arr, size=n, replace=True)
        bootstrapped_dataset = ROOT.RooDataSet("bootstrapped_data", "bootstrapped_data", ROOT.RooArgSet(x))
        for val in bootstrapped_data:
            x.setVal(val)
            bootstrapped_dataset.add(ROOT.RooArgSet(x))

        model = emm.ExponentialMixtureModel(x, k, prefix=f"boot_k{k}_")
        fit_result = model.pdf.fitTo(
            bootstrapped_dataset,
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Save(True)
        )
        if fit_result.status() > 2:
            continue

        # D = l(data_b|theta_b) - l(data|theta_b)
        nll_boot = model.pdf.createNLL(bootstrapped_dataset)
        nll_data = model.pdf.createNLL(data)

        ll_boot = -1 * nll_boot.getVal()
        ll_data = -1 * nll_data.getVal()

        D = ll_boot - ll_data
        Ds.append(D)
    
    # Save results
    filename = bootstrap_bias_filename(k, n_bootstraps, seed)
    with open(filename, "wb") as f:
        pickle.dump(np.array(Ds), f)

def run_bootstrap_bias_jobs(
        x, data,
        ks, n_bootstraps_per_seed, seeds,
        remake=False, **run_task_kwargs
    ):
    from tools import scale_out as so

    # if remake:
    #     storage.clear_cache(model_selection_cache)
    
    tasks = []
    for k in ks:
        for seed in seeds:
            filename = bootstrap_bias_filename(k, n_bootstraps_per_seed, seed)
            if os.path.exists(filename) and not remake:
                continue

            task = so.Task(
                bootstrap_bias,
                x, data, k, n_bootstraps_per_seed, seed
            )
            tasks.append(task)
    
    return so.run_tasks(tasks, **run_task_kwargs)

def load_bootstrap_bias_results(ks, n_bootstraps_per_seed, seeds):
    import numpy as np
    results = {}
    for k in ks:
        D_values = []
        for seed in seeds:
            filename = bootstrap_bias_filename(k, n_bootstraps_per_seed, seed)
            if not os.path.exists(filename):
                print(f"Warning: missing file {filename}")
                continue
            with open(filename, "rb") as f:
                Ds = pickle.load(f)
                D_values.extend(Ds)
        results[k] = np.array(D_values)
    return results


def train_test_split(x, data, n_folds, seed=1234):
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

def CV_log_likelihood_name(k, n_folds, data_name, n_random_restarts, ordered):
    if n_folds < 1:
        n_folds = "LOO"
    ordered_tag = "_ordered" if ordered else ""
    tags = f"CV_log_likelihood_{data_name}_k{k}_nfolds{n_folds}_nrestarts{n_random_restarts}{ordered_tag}.pkl"
    f = f"{model_selection_cache}/{tags}"
    return f

def CV_log_likelihood(x, data, k, n_folds, n_random_restarts=10, ordered=False, i_fold=None):
    import numpy as np
    import ROOT

    n = data.numEntries()
    data_arr = np.array([data.get(i).getRealValue("x") for i in range(n)])

    n_folds_default = n_folds
    if n_folds < 1:
        n_folds = n
    else:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(data_arr)
    data_arr_folds = np.array_split(data_arr, n_folds)

    ll_values = np.zeros(n_folds)
    best_fit_par_list = []
    i_folds = range(n_folds) if i_fold is None else [i_fold]
    for i in i_folds:
        test_data_arr = data_arr_folds[i]
        train_data_arr = np.concatenate([data_arr_folds[j] for j in range(n_folds) if j != i])

        train_dataset = ROOT.RooDataSet("train_data", "train_data", ROOT.RooArgSet(x))
        for val in train_data_arr:
            x.setVal(val)
            train_dataset.add(ROOT.RooArgSet(x))
        
        best_nll = np.inf
        best_fit_pars = None

        rng = np.random.default_rng(seed=42)
        for attempt in range(n_random_restarts):
            if ordered:
                model = emm.ExponentialMixtureModel_Ordered(x, k, prefix=f"cv_k{k}_fold{i}_", data_mean=data.mean(x))
            else:
                model = emm.ExponentialMixtureModel(x, k, prefix=f"cv_k{k}_fold{i}_", data_mean=data.mean(x))
            model.randomize_params(rng=rng)

            fit_result = fit_n_times(model, train_dataset, n_attempts=42)
            if fit_result.status() <= 2 and fit_result.minNll() < best_nll:
                best_nll = fit_result.minNll()
                best_fit_pars = {p.GetName(): p.getVal() for p in model.params()}

        if best_fit_pars is None:
            ll_values[i] = np.nan
            print(f"Warning: fit failed for k={k}, fold={i}")
            continue

        # Set best fit parameters
        if ordered:
            model = emm.ExponentialMixtureModel_Ordered(x, k, prefix=f"cv_k{k}_fold{i}_", data_mean=data.mean(x))
        else:
            model = emm.ExponentialMixtureModel(x, k, prefix=f"cv_k{k}_fold{i}_", data_mean=data.mean(x))
        model.set_params(best_fit_pars)
        best_fit_par_list.append(best_fit_pars)

        test_dataset = ROOT.RooDataSet("test_data", "test_data", ROOT.RooArgSet(x))
        for val in test_data_arr:
            x.setVal(val)
            test_dataset.add(ROOT.RooArgSet(x))
        
        nll_test = model.pdf.createNLL(test_dataset)
        ll_test = -1 * nll_test.getVal()
        ll_values[i] = ll_test
    
    # Save results
    filename = CV_log_likelihood_name(k, n_folds_default, data.GetName(), n_random_restarts, ordered)
    output = {
        "ll_values": ll_values,
        "best_fit_pars": best_fit_par_list
    }
    with open(filename, "wb") as f:
        pickle.dump(output, f)

def run_CV_log_likelihood_jobs(
        x, data,
        ks, n_folds,
        n_random_restarts=10,
        ordered=False,
        remake=False,
        **run_task_kwargs
    ):
    from tools import scale_out as so

    # if remake:
    #     storage.clear_cache(model_selection_cache)
    
    tasks = []
    for k in ks:
        filename = CV_log_likelihood_name(k, n_folds, data.GetName(), n_random_restarts, ordered)
        if os.path.exists(filename) and not remake:
            continue

        task = so.Task(
            CV_log_likelihood,
            x, data, k, n_folds,
            n_random_restarts=n_random_restarts,
            ordered=ordered
        )
        tasks.append(task)
    if not tasks:
        return []
    
    return so.run_tasks(tasks, **run_task_kwargs)

def load_CV_log_likelihood_results(ks, data_name, n_folds, n_random_restarts, ordered, drop_failed_fits=True):
    import numpy as np
    results = {}
    for k in ks:
        filename = CV_log_likelihood_name(k, n_folds, data_name, n_random_restarts, ordered)
        if not os.path.exists(filename):
            print(f"Warning: missing file {filename}")
            continue
        with open(filename, "rb") as f:
            ll_values = pickle.load(f)
            results[k] = np.array(ll_values)

    return results

def run_random_restarts_for_CV(
        x, full_dataset, n_folds,
        model_primitives, seeds,
        n_random_restarts,
        n_retries=42,
        ordered=False,
        remake=False,
        **run_task_kwargs
    ):
    from tools import scale_out as so
    tasks = []

    for seed in seeds:
        train_datasets, test_datasets = train_test_split(x, full_dataset, n_folds, seed=seed)
        for i_fold in range(n_folds):
            train_dataset = train_datasets[i_fold]
            for model_primitive in model_primitives:
                filename = emm.random_restarts_filename(
                    model_primitive.name,
                    train_dataset.GetName(),
                    n_random_restarts,
                    seed,
                )
                if os.path.exists(filename) and not remake:
                    continue

                task = so.Task(
                    emm.fit_random_restarts,
                    x, train_dataset,
                    model_primitive,
                    seed, n_random_restarts,
                    n_retries=n_retries,
                )
                tasks.append(task)
        
    return so.run_tasks(tasks, **run_task_kwargs)


def load_CV_random_restart_results(
        x, data, n_folds,
        model_primitives, seeds,
        n_random_restarts,
        ):
    import pickle

    results = {
        model.name: {
            seed: {
                i_fold: {} for i_fold in range(n_folds)
                } for seed in seeds
            } for model in model_primitives
        }

    for seed in seeds:
        train_datasets, test_datasets = train_test_split(x, data, n_folds, seed=seed)
        for model in model_primitives:
            for i_fold in range(n_folds):
                data_name = train_datasets[i_fold].GetName()
                filename = emm.random_restarts_filename(
                    model.name,
                    data_name,
                    n_random_restarts,
                    seed,
                )
                if not os.path.exists(filename):
                    print(f"Warning: missing file {filename}")
                    continue

                with open(filename, "rb") as f:
                    fit_results = pickle.load(f)
                results[model.name][seed][i_fold] = fit_results[-1]

    return results


def evaluate_CV_log_likelihood(
        x, data, n_folds, seeds,
        model_primitives,
        results,
        ordered=False,
        ):
    
    import numpy as np
    import ROOT

    CV_results = {}
    for model_primitive in model_primitives:
        model_name = model_primitive.name
        CV_results[model_name] = {}
        for seed in seeds:
            train_datasets, test_datasets = train_test_split(x, data, n_folds, seed=seed)
            ll_values = np.zeros(n_folds)
            for i_fold in range(n_folds):
                train_dataset = train_datasets[i_fold]
                test_dataset = test_datasets[i_fold]

                fit_result = results[model_name][seed][i_fold]
                model = model_primitive(x)
                model.set_params(fit_result['final_pars'])
                nll_test = model.pdf.createNLL(test_dataset)
                ll_test = -1 * nll_test.getVal()
                ll_values[i_fold] = ll_test
            CV_results[model_name][seed] = np.sum(ll_values)
    return CV_results
                