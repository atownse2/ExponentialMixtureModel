import os

import ROOT
from array import array
import os

import pandas as pd
import numpy as np
import multiprocessing as mp

import json

import matplotlib.pyplot as plt

import random
random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

from tools import root_tools
from tools import storage_config as stor

top_dir = os.path.dirname(os.path.abspath(__file__))
ensure_cache = lambda name, cache_dir=None: stor.ensure_cache(name, cache_dir=cache_dir)
emm_cache = ensure_cache("emm")

# Data
data_dir = f"{top_dir}/data/high_mass_diphoton/"

def get_data():
    triggers = {
        "2016": "HLT_DoublePhoton60",
        "2017": "HLT_DoublePhoton70",
        "2018": "HLT_DoublePhoton70",
    }

    
    # Get list of diphoton invariant masses
    mgg = []
    for year in triggers.keys():
        d = ROOT.RDataFrame(
            "diphoton/fTree",  # Name of the tree in the file
            f"{data_dir}/Data{year}/Run{year}*.root"
        )

        # Apply selections
        isGood = d.Filter("isGood == 1")
        pass_trig = isGood.Filter(f"TriggerBit.{triggers[year]} == 1 | TriggerBit.HLT_ECALHT800 == 1")
        pass_kin = pass_trig.Filter("Diphoton.Minv > 500 && Diphoton.deltaR > 0.45 && Photon1.pt > 125 && Photon2.pt > 125 && Diphoton.isEBEB")

        # Get the invariant mass of the diphoton system and append to the list
        mgg.extend(list(pass_kin.AsNumpy(["Diphoton.Minv"])['Diphoton.Minv']))

    print(f"Loaded {len(mgg)} diphoton invariant masses from data.")
    t_mgg = root_tools.to_root_tree([mgg], "mgg", ["x"], index=True)
    return t_mgg

def get_fine_binning():
    # Get fine histogram
    txt_file = f"{data_dir}/high_mass_diphoton_EBEB.txt"

    # Read txt file and use to fill histogram
    bins_low = []
    bins_high = []
    values = []

    with open(txt_file, 'r') as f:
        for line in f:
            bin_info, vals = line.split(":")
            bin_low, bin_high = bin_info.replace("GeV", "").split("-")
            bin_low = float(bin_low)
            bin_high = float(bin_high)

            integral_info, divide_bin_width = vals.split(",")
            integral = float(integral_info.replace("Integral=", ""))
            bins_low.append(bin_low)
            bins_high.append(bin_high)
            values.append(round(integral*(bin_high-bin_low)))

    bins = bins_low + [bins_high[-1]]

    # print( f"Loaded fine binning histogram with {sum(values)} total events." )
    return bins

def get_coarse_binning():
    # Get coarse histogram
    import re

    txt_file = "/project01/ndcms/atownse2/AN-23-135/data/hep_data/high_mass_diphoton/high_mass_diphoton_EBEB_coarse.txt"
    bins = np.arange(500, 3500, 100)
    h = ROOT.TH1F("h", "h", len(bins)-1, array('d', bins))

    with open(txt_file, "r") as f:
        for line in f:
            m = re.search(r"(\d+)\s*GeV\s*\[Post-fit Pred=([\d\.]+)\]\s*\[Data=(\d+)\]", line)
            if m:
                bin_center = float(m.group(1))
                post_fit = float(m.group(2))
                data = int(m.group(3))

            for i in range(data):
                h.Fill(bin_center)

    print(h.Integral())

# Background Models:
class ExponentialMixtureModel:
    name = "ExponentialMixtureModel"
    def __init__(self, x, n_exp, **par_specs):

        fitted_par_specs = {
            "weight": (1/n_exp, 0, 1),
            "raw_rate": (-0.0045, -1e-1, -1e-5),
            # "raw_rate": (-60, -400, -1e-3),
        }

        for key, spec in par_specs.items():
            if isinstance(spec, tuple):
                fitted_par_specs[key] = spec
            elif isinstance(spec, float):
                _spec = fitted_par_specs[key]
                fitted_par_specs[key] = (spec, _spec[1], _spec[2])

        fitted = {}
        other = {}

        ## Fitted Parameters
        for i in range(n_exp-1):
            fitted[f"weight_{i}"] = ROOT.RooRealVar(f"weight_{i}", f"Mixture weight {i}", *fitted_par_specs["weight"])
        for i in range(n_exp):
            fitted[f"raw_rate_{i}"] = ROOT.RooRealVar(f"raw_rate_{i}", f"Unordered rate {i}", *fitted_par_specs["raw_rate"])
        
        ## Ordered Rates for identifiability
        ## a1, a2, a3 => a1, a1+a2, a1+a2+a3
        for i in range(n_exp):
            other[f"rate_{i}"] = ROOT.RooAddition(
                f"rate_{i}",
                f"Ordered Rate for exponential {i}",
                ROOT.RooArgList(*[fitted[f"raw_rate_{j}"] for j in range(i+1)]))
        
        
        ## Exponentials
        for i in range(n_exp):
            other[f"exp_{i}"] = ROOT.RooExponential(
                f"exp_{i}",
                f"exp_{i}", x,
                other[f"rate_{i}"]
                )
        
        ## PDFs
        other["pdf"] = ROOT.RooAddPdf(
            "emm_pdf",
            "Exponential Mixture",
            ROOT.RooArgList(*[other[f"exp_{i}"] for i in range(n_exp)]),
            ROOT.RooArgList(*[fitted[f"weight_{i}"] for i in range(n_exp-1)]),
            True,
            )
        
        self.fitted = fitted
        self.other = other
        self.pdf = other["pdf"]

class Dijet:
    name="Dijet"
    def __init__(self, x):
        
        # self.p0 = ROOT.RooRealVar("p0", "p0", 0.13, 0.05, 0.3)  
        self.p1 = ROOT.RooRealVar("p1", "p1", 5.7, 5.5, 5.9)
        self.p2 = ROOT.RooRealVar("p2", "p2", -0.78, -1.0, -0.5)

        self.pdf = ROOT.RooGenericPdf(
            "dijet_pdf", 
            "pow(x,p1+p2*TMath::Log(x))",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x),  # Arguments for the formula
        )

class ExpPow:
    name= "ExpPow"
    def __init__(self, x):
        self.p1 = ROOT.RooRealVar("p1", "p1", -0.0016, -0.003, -0.001)
        self.p2 = ROOT.RooRealVar("p2", "p2", 1.8, 1.5, 2.0)

        self.pdf = ROOT.RooGenericPdf(
            "exppow_pdf",
            "exp(p1*x)*pow(x,-1*p2*p2)",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x)  # Arguments for the formula
        )

def AIC_BIC_loo(n_exp, t, k_folds=-1):

    x = ROOT.RooRealVar("x", "Diphoton Mass [GeV]", 500, 4000)
    index=ROOT.RooRealVar("index", "index", 0, 0, 1e6)

    data = ROOT.RooDataSet("mgg", "mgg", ROOT.RooArgSet(x, index), ROOT.RooFit.Import(t))
    
    model_inst = ExponentialMixtureModel(x, n_exp)
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

        n_exps = [i for i in range(1, k_max + 1)]
        results = []
        with mp.Pool(len(n_exps)) as pool:
            results = pool.starmap(
                AIC_BIC_loo, 
                [(i, t) for i in n_exps]
            )

        # Collect results
        df = []
        for i, result in enumerate(results):
            AIC_low, AIC_med, AIC_high = np.percentile(result["AIC"], [16, 50, 84])
            BIC_low, BIC_med, BIC_high = np.percentile(result["BIC"], [16, 50, 84])
            df.append({
                "n_exp": n_exps[i],
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
        df['n_exp'],
        df['AIC_med'],
        yerr=[df['AIC_med']-df['AIC_low'], df['AIC_high']-df['AIC_med']],
        capsize=5,
        markersize=10
    )
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("AIC")
    ax[0].set_xticks(df['n_exp'])

    ax[1].errorbar(
        df['n_exp'],
        df['BIC_med'],
        yerr=[df['BIC_med']-df['BIC_low'], df['BIC_high']-df['BIC_med']],
        capsize=5,
        markersize=10
    )
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("BIC")
    ax[1].set_xticks(df['n_exp'])

    # fig.suptitle("Leave-One-Out Cross Validation")
    plt.tight_layout()
    plt.show()

def fit_random_subset(tree):

    bounds = (500, 4000)

    lower = random.uniform(bounds[0], bounds[1])
    upper = random.uniform(lower, bounds[1])

    x = ROOT.RooRealVar("x", "x", lower, upper)
    data = ROOT.RooDataSet("data", "data", tree, ROOT.RooArgSet(x))

    rate = ROOT.RooRealVar("rate", "rate", -1e-3, -1e-1, -1e-9)
    pdf = ROOT.RooExponential("pdf", "pdf", x, rate)
    
    result = pdf.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))

    # Check if the fit converged
    fit_status = result.status()  # 0 means OK
    cov_quality = result.covQual()  # 3 is the best

    if fit_status != 0 or cov_quality < 2:
        # print(f"Fit did not converge properly: status={fit_status}, covQual={cov_quality}")
        return None  # or np.nan or some error value

    rate_val = rate.getVal()
    return rate_val

# Fit many random subsets
import multiprocessing as mp

n_subsets = 9000
with mp.Pool(12) as pool:
    rates = pool.map(fit_random_subset, [t_mgg] * n_subsets)

rates = [rate for rate in rates if rate is not None]  # Filter out failed fits

def plot_fits(
    data, x, bins,
    models,
    model_labels,
    fit_results,
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kMagenta, ROOT.kCyan],
    ):

    c = ROOT.TCanvas(random_string(), "canvas", 800, 800)
    c.cd()

    # Frames
    main_frame = x.frame(ROOT.RooFit.Title("Diphoton Mass Fit"))
    # main_frame.GetYaxis().SetRangeUser(1e-3, main_frame.GetMaximum()*5)
    # main_frame.GetXaxis().Set(len(bins)-1, array('d', bins))

    pull_frame = x.frame(ROOT.RooFit.Title("Pull"))
    pull_frame.SetTitle("")  # Remove title
    pull_frame.GetYaxis().SetTitle("Pull")
    pull_frame.GetYaxis().SetTitleSize(0.1)
    pull_frame.GetYaxis().SetTitleOffset(0.5)
    pull_frame.GetYaxis().SetLabelSize(0.08)
    pull_frame.GetXaxis().SetTitleSize(0.1)
    pull_frame.GetXaxis().SetTitle("m_{#gamma#gamma} [GeV]")
    pull_frame.GetXaxis().SetLabelSize(0.08)
    pull_frame.GetYaxis().SetRangeUser(-5, 5)

    # Create a legend
    legend = ROOT.TLegend(0.55, 0.65, 0.89, 0.89)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # Transparent legend background


    data.plotOn(main_frame)
    for i, model in enumerate(models):
        pdf = model.pdf
        fit_result = fit_results[i]
        model_label = model_labels[i]

        pdf.plotOn(
            main_frame,
            ROOT.RooFit.LineColor(colors[i]),
            ROOT.RooFit.Name(model_label),  # Name for the PDF in the legend
            ROOT.RooFit.DrawOption("L"),  # Use "L" for line only
        )
        # Visualize the error band if fit_result is available
        if fit_result is not None:
            pdf.plotOn(
                main_frame,
                ROOT.RooFit.VisualizeError(fit_result, 1),
                ROOT.RooFit.FillColor(colors[i]-10),
                ROOT.RooFit.MoveToBack()
            )
        
        # Pull
        pull_hist = main_frame.pullHist()
        pull_hist.SetLineColor(colors[i])
        pull_hist.SetMarkerColor(colors[i])

        pull_frame.addPlotable(pull_hist, "P")

        # Legend
        legend.AddEntry(model_label, model_label, "l")


    # Plot the histograms
    main_pad = ROOT.TPad("main_pad", "Main Pad", 0, 0.3, 1, 1)
    pull_pad = ROOT.TPad("pull_pad", "Pull Pad", 0, 0, 1, 0.3)

    main_pad.SetLogy()
    main_pad.SetLogx()
    main_pad.SetBottomMargin(0)
    main_pad.Draw()

    pull_pad.SetLogx()
    pull_pad.SetTopMargin(0)
    pull_pad.SetBottomMargin(0.35)
    pull_pad.Draw()

    main_pad.cd()
    main_frame.SetMinimum(1e-3)
    main_frame.Draw()
    legend.Draw()

    pull_pad.cd()
    pull_frame.Draw()


    # pull_frame.Draw()

    # # Draw a dotted line at y=0
    # x_min = pull_frame.GetXaxis().GetXmin()
    # x_max = pull_frame.GetXaxis().GetXmax()
    # zero_line = ROOT.TLine(x_min, 0, x_max, 0)
    # zero_line.SetLineStyle(2)  # Dotted line
    # zero_line.SetLineColor(ROOT.kBlack)
    # zero_line.Draw()

    c.Update()
    c.Draw()


def get_bias_inputs(toy_model, n_toys, n_events_per_toy, n_exp):

    cache_dir = ensure_cache("emm/bias")
    tags = [toy_model, f"{n_toys}toys", f"{n_events_per_toy}events", f"{n_exp}exp"]
    tag = "_".join(tags)
    bias_file = os.path.join(cache_dir, f"bias_inputs_{tag}.root")
    if os.path.exists(bias_file):
        with open(bias_file, 'r') as f:
            bias_inputs = json.load(f)
        return bias_inputs
    
    if not run:
        return None
    
    print(f"Generating bias inputs for {toy_model} with {n_toys} toys and {n_events_per_toy} events each...")

    x = ROOT.RooRealVar("x", "Diphoton Mass [GeV]", 500, 4000)

    if toy_model == "Dijet":
        toy_model = Dijet(x)
    elif toy_model == "ExpPow":
        toy_model = ExpPow(x)
    else:
        raise ValueError("Unsupported toy model. Use 'Dijet' or 'ExpPow'.")

    model_to_fit = ExponentialMixtureModel(x, n_exp)

    bias_inputs = []
    for i in range(n_toys):
        toy_data = ROOT.RooDataSet(
            f"toy_data_{i}", "toy_data",
            ROOT.RooArgSet(x),
            ROOT.RooFit.Import(
                toy_model.pdf.generate(
                    ROOT.RooArgSet(x),
                    n_events_per_toy,
                    )
            )
        )

        fit_result = model_to_fit.pdf.fitTo(
            toy_data, ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

        # Calculate the bias for each toy
        h_pdf_fit = model_to_fit.pdf.createHistogram(f"h_pdf_fit_{i}", x)
        h_pdf_true = toy_model.pdf.createHistogram(f"h_pdf_true_{i}", x)

        true_values = []
        fit_values = []
        fit_error = []
        for i in range(h_pdf_fit.GetNbinsX()):
            true_values.append(h_pdf_true.GetBinContent(i+1))
            fit_values.append(h_pdf_fit.GetBinContent(i+1))
            fit_error.append(h_pdf_fit.GetBinError(i+1))
        
        bias_inputs.append({
            "true": true_values,
            "fit": fit_values,
            "fit_error": fit_error,
        })
    
    # Save bias inputs to cache
    with open(bias_file, 'w') as f:
        json.dump(bias_inputs, f, indent=4)
    print(f"Saved bias inputs to cache: {bias_file}")
    return bias_inputs

def get_bias_info(toy_model, n_toys, n_events_per_toy_list, n_exps):
    df = []
    one = False
    for n_exp in n_exps:
        for n_events_per_toy in n_events_per_toy_list:
            l = get_bias_inputs(
                toy_model=toy_model,
                n_toys=n_toys,
                n_events_per_toy=n_events_per_toy,
                n_exp=n_exp,
            )
            if l is None:
                print(f"No bias inputs found for n_exp={n_exp}, n_events_per_toy={n_events_per_toy}. Skipping.")
                continue
        
            fit_values = []
            fit_errors = []
            true_values = []
            for d in l:
                fit_values.append(d['fit'])
                fit_errors.append(d['fit_error'])
                true_values.append(d['true'])
            fit_values = np.array(fit_values)
            fit_errors = np.array(fit_errors)
            true_values = np.array(true_values)
            pull_values = (fit_values- true_values) / fit_errors
            pull_mean = np.mean(pull_values, axis=0)
            pull_err = np.std(pull_values, axis=0) / np.sqrt(n_toys)  # Standard error of the mean

            covered = (true_values > (fit_values - fit_errors)) & (true_values < (fit_values + fit_errors))

            covered_percentage = np.mean(covered, axis=0) * 100  # Percentage of toys where the true value is within the error range

            df.append({
                'n_exp': n_exp,
                'n_events_per_toy': n_events_per_toy,
                'pull_mean': pull_mean,
                'pull_err': pull_err,
                'covered_percentage': covered_percentage,
            })

    # Convert to DataFrame
    df = pd.DataFrame(df)
    return df