import os
import time

import json

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
from tools import condor
from tools import cache

emm_cache = cache.ensure_cache("emm")

data_dir = f"{cache.top_dir}/data/high_mass_diphoton/"

def get_data(normalize=False, sort_and_index=False, tree=False):
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

    if normalize:
        mgg = (np.array(mgg) - np.min(mgg)) / np.mean(mgg)
    
    if not tree:
        mgg = np.array(mgg)
        return mgg
    print(f"Loaded {len(mgg)} diphoton invariant masses from data.")
    if sort_and_index:
        mgg = np.sort(mgg)
        t_mgg = root_tools.to_root_tree([mgg], "mgg", ["x"], index=True)
    else:
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
def stick_breaking_weights(n_components: int, **custom_ranges):
    """
    Generate stick-breaking weights for a mixture model.
    """
    stick_proportions = [
        ROOT.RooRealVar(
            f"raw_weight_{i}",
            f"Stick proportion {i}",
            1/(n_components - i),
            *custom_ranges.get(f"raw_weight_{i}", (0, 1))  
        ) for i in range(n_components - 1)
    ]
    stick_proportions.append(ROOT.RooRealVar(
        f"raw_weight_{n_components-1}",
        f"Stick proportion {n_components-1}",
        1 # Fixed to 1
    ))

    weights = []
    for i in range(n_components):
        prod_terms = [f"(1-raw_weight_{j})" for j in range(i)]
        prod_str = "*".join(prod_terms) if prod_terms else "1"
        # print(f"Formula for weight_{i}: raw_weight_{i} * {prod_str}")
        weight = ROOT.RooFormulaVar(
                    f"weight_{i}",
                    f"Weight for component {i}",
                    f"raw_weight_{i} * {prod_str}",
                    ROOT.RooArgList(*stick_proportions[:i+1])
        )
        weights.append(weight)
    
    return weights, stick_proportions[:-1]

def normalization_weights(n_components, **custom_ranges):
    """
    Generate normalization construction weights for a mixture model.
    """
    unnormalized_weights = []
    for i in range(n_components):
        w = ROOT.RooRealVar(
                f"raw_weight_{i}",
                f"Unnormalized weight for component {i}",
                custom_ranges.get(f"raw_weight_{i}", 0),
                # 0.5, 0, 1
            )
        if i == 0:
            w.setConstant(True)
            # w.setConstant(False) # In general this should be fixed but for now let it float
        else:
            w.setConstant(False)
        unnormalized_weights.append(w)

    sum_weights_str = "+".join([f"exp({w.GetName()})" for w in unnormalized_weights])
    # sum_weights_str = "+".join([w.GetName() for w in unnormalized_weights])
    weights = [
        ROOT.RooFormulaVar(
            f"weight_{i}",
            f"Weight for component {i}",
            f"exp({unnormalized_weights[i].GetName()})/({sum_weights_str})",
            # f"{unnormalized_weights[i].GetName()}/({sum_weights_str})",
            ROOT.RooArgList(*unnormalized_weights)
        ) for i in range(n_components)
    ]
    return weights, unnormalized_weights

def ordered_weights(n_components: int, **custom_ranges):
    """
    Transform free parameters to positive parameters with exponential
    Transform positive parameters to ordered parameters with sequential sums
    Transform ordered parameters to normalized parameters
    """

    raw_weights = [
        ROOT.RooRealVar(
            f"raw_weight_{i}",
            f"Raw weight for component {i}",
            *custom_ranges.get(f"raw_weight_{i}", (0.5, 0, 1)),
        ) for i in range(n_components)
    ]

    ordered_weights = [
        ROOT.RooFormulaVar(
            f"ordered_weight_{i}",
            f"Ordered Raw Weight for component {i}",
            "+".join([p.GetName() for p in raw_weights[:i+1]]),
            ROOT.RooArgList(*raw_weights[:i+1])
        ) for i in range(n_components)
    ]

    sum_ordered_weights = "+".join([p.GetName() for p in ordered_weights])
    weights = [
        ROOT.RooFormulaVar(
            f"weight_{i}",
            f"Weight for component {i}",
            f"({ordered_weights[i].GetName()})/({sum_ordered_weights})",
            ROOT.RooArgList(*ordered_weights)
        ) for i in range(n_components)
    ]

    return weights, raw_weights, ordered_weights

    # unordered_unnormalized_weights = []
    # for i in range(n_components):
    #     w = ROOT.RooRealVar(
    #         f"raw_weight_{i}",
    #         f"Unnormalized weight for component {i}",
    #         custom_ranges.get(f"raw_weight_{i}", 0),
    #     )
    #     if i == 0:
    #         w.setConstant(True)
    #     else:
    #         w.setConstant(False)
    #     unordered_unnormalized_weights.append(w)
    
    # ordered_weights = []
    # for i in range(n_components):
    #     sum_str = "+".join([f"exp({p.GetName()})" for p in unordered_unnormalized_weights[:i+1]])
    #     w = ROOT.RooFormulaVar(
    #         f"ordered_weight_{i}",
    #         f"Ordered weight for component {i}",
    #         f"{sum_str}",
    #         ROOT.RooArgList(*unordered_unnormalized_weights[:i+1])
    #     )
    #     ordered_weights.append(w)

    # weights = []

    # denominator_str = "+".join([p.GetName() for p in ordered_weights])
    # for i in range(n_components):
    #     w = ROOT.RooFormulaVar(
    #         f"weight_{i}",
    #         f"Ordered weight for component {i}",
    #         f"({ordered_weights[i].GetName()})/({denominator_str})",
    #         ROOT.RooArgList(*ordered_weights)
    #     )
    #     weights.append(w)

    # return weights, unordered_unnormalized_weights, ordered_weights

def mixture_pdf(weights, pdfs, name="pdf"):
    assert len(weights) == len(pdfs), "Weights and PDFs must have the same length"
    n = len(weights)

    pdf_terms = [f"{w.GetName()}*{p.GetName()}" for w, p in zip(weights, pdfs)]
    pdf_str = "+".join(pdf_terms)
    pdf = ROOT.RooGenericPdf(
        name,
        "Mixture PDF",
        pdf_str,
        ROOT.RooArgList(*(weights + pdfs))
    )
    return pdf

def print_par(par):
    """
    Print the parameter name, value, and error.
    """
    print_str = f"{par.GetName()}: {par.getVal()}"
    if hasattr(par, 'isConstant') and not par.isConstant():
        if hasattr(par, 'getErrorHi'):
            print_str += f" + {par.getErrorHi()} {par.getErrorLo()}"
        elif hasattr(par, 'getError'):
            print_str += f" ± {par.getError()}"
        else:
            pass
    
    print(print_str)

class RooFitModel:
    name = "GenericRooFitModel"
    def params(self):
        """
        Return the parameters of the model.
        """
        raise NotImplementedError("Subclasses must implement params method")
    
    def get_param(self, name: str):
        """
        Get a parameter by name.
        """
        for par in self.params():
            if par.GetName() == name:
                return par
        raise ValueError(f"Parameter {name} not found in model parameters")
    
    def set_param(self, name: str, value: float, constant=False):
        p = self.get_param(name)
        p.setVal(value)
        if constant:
            p.setConstant(True)
        else:
            p.setConstant(False)
    
    def set_params(self, name_value_dict: dict, constant=False):
        """
        Set multiple parameters from a dictionary.
        """
        for name, value in name_value_dict.items():
            self.set_param(name, value, constant=constant)
    
    def print(self):
        """
        Print the model parameters.
        """
        print(f"Model: {self.name}")
        for par in self.params():
            print_par(par)

class MixtureModel(RooFitModel):

    def __init__(self, x, n_components: int, **kwargs):
        self.n_components = n_components
        self.name = f"MixtureModel_{n_components}"
        self.kwargs = kwargs
        
        self.init_weights()
        self.init_pdfs(x)
        self.pdf = mixture_pdf(self.weights, self.pdfs, name=kwargs.get("name", self.name))
    
    def init_weights(self, **custom_ranges):
        if self.kwargs.get("stick_breaking", False):
            self.weights, self.raw_weights = stick_breaking_weights(self.n_components, **custom_ranges)
        elif self.kwargs.get("ordered_weights", False):
            self.weights, self.raw_weights, self.ordered_weights = ordered_weights(self.n_components, **custom_ranges)
        else:
            self.weights, self.raw_weights = normalization_weights(self.n_components)
    
    def init_pdfs(self, x):
        """
        Initialize the PDFs for the mixture model.
        The PDF names should be in the format "pdf_{i}" where i is the index of the component.
        """
        raise NotImplementedError("Subclasses must implement init_pdfs")

class ExponentialMixtureModel(MixtureModel):

    def integral(self, x, lo, hi):
        integral = 0
        for i in range(self.n_components):
            rate = self.rates[i].getVal()
            weight = self.weights[i].getVal()
            integral += weight*(np.exp(rate*lo) - np.exp(rate*hi))
        return integral

    def init_rates(self, x):
        self.name = f"Exponential{self.name}"
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        # Initialize raw rates
        if "initial_raw_rates" in self.kwargs:
            initial_raw_rates = self.kwargs["initial_raw_rates"]
            assert len(initial_raw_rates) == self.n_components, "initial_raw_rates must have the same length as n_components"
        else:
            initial_raw_rates = [(i+1) for i in range(self.n_components)]
        
        self.raw_rates = [
            ROOT.RooRealVar(
                f"raw_rate_{i}",
                f"Raw rate for exponential {i}",
                initial_raw_rates[i], 0, 100
            ) for i in range(self.n_components)
        ]

        self.rates = [
            ROOT.RooFormulaVar(
                f"rate_{i}",
                f"Rate scaled by data for exponential {i}",
                f"{rate_scaling}*raw_rate_{i}",
                ROOT.RooArgList(self.raw_rates[i])
            ) for i in range(self.n_components)
        ]
    
    def init_pdfs(self, x):
        self.init_rates(x)

        if "max_tail_prob" in self.kwargs:    
            self.weights[0].setRange(0, self.kwargs["max_tail_prob"])

        self.pdfs = [
            ROOT.RooExponential(
                f"pdf_{i}",
                f"Exponential PDF {i}",
                x, self.rates[i]
            ) for i in range(self.n_components)    
        ]
    
    def params(self):
        """
        Return the parameters of the model.
        """
        return self.raw_rates + self.raw_weights

class ExponentialMixtureModel_Ordered(ExponentialMixtureModel):
    def init_rates(self, x):
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        self.raw_rate_diffs = [
            ROOT.RooRealVar(
                f"raw_rate_diff_{i}",
                f"Inverse rate difference {i}",
                0.5, 0, 100
            ) for i in range(self.n_components)
        ]

        self.rates = [
            ROOT.RooFormulaVar(
                f"rate_{i}",
                f"Ordered Rate for exponential {i}",
                f"{rate_scaling}/({'+'.join([rd.GetName() for rd in self.raw_rate_diffs[:i+1]])})",
                ROOT.RooArgList(*self.raw_rate_diffs[:i+1])
            ) for i in range(self.n_components)
        ]

    def params(self):
        """
        Return the parameters of the model.
        """
        return self.raw_rate_diffs + self.raw_weights

class ExponentialMixtureModel_Ordered_SmallTail(ExponentialMixtureModel_Ordered):
    def init_weights(self):
        assert self.n_components >= 2, "n_components must be at least 2 for SmallTail model"
        custom_ranges = {
            f"raw_weight_{self.n_components-2}": (0.95, 1)
            }
        super().init_weights(**custom_ranges)
    

class LomaxMixtureModel(MixtureModel):
    name = "LomaxMixtureModel"
    def init_alphas(self, x):
        self.alphas = [
            ROOT.RooRealVar(
                f"alpha_{i}", f"Shape parameter {i}",
                *self.kwargs.get(f"alpha_{i}", (2+i, 0.01, 10000))
            ) for i in range(self.n_components)
        ]
    
    def init_betas(self, x):
        self.betas = [
            ROOT.RooRealVar(
                f"beta_{i}", f"Scale parameter {i}",
                *self.kwargs.get(f"beta_{i}", (0.005, 0.000001, 10))
            ) for i in range(self.n_components)
        ]

    def init_pdfs(self, x):
        self.init_alphas(x)
        self.init_betas(x)
        
        self.pdfs = [
            ROOT.RooGenericPdf(
                f"pdf_{i}", f"Lomax {i}",
                f"(alpha_{i}*beta_{i}) * (1 + x*beta_{i})^(-alpha_{i}-1)",
                ROOT.RooArgList(
                    x,
                    self.betas[i],
                    self.alphas[i],
                )
            ) for i in range(self.n_components)
        ]

    def params(self):
        return self.alphas + self.betas + self.raw_weights

class GaussianSignalModel(RooFitModel):
    name = "GaussianSignalModel"
    def __init__(self, x, mean, **kwargs):
        self.mean = ROOT.RooRealVar(
            "signal_mean",
            "Signal mean",
            mean,
        )
        self.sigma = ROOT.RooRealVar(
            "signal_sigma",
            "Signal sigma",
            kwargs.get("initial_sigma", 10.0),
            kwargs.get("min_sigma", 0.1),
            kwargs.get("max_sigma", 50.0)
        )
        self.pdf = ROOT.RooGaussian(
            "signal_pdf",
            "Gaussian Signal PDF",
            x,
            self.mean,
            self.sigma
        )
    
    def params(self):
        return [self.mean, self.sigma]

class SignalPlusBackgroundModel(RooFitModel):
    name = "SignalPlusBackgroundModel"
    def __init__(self, signal_model: RooFitModel, background_model: RooFitModel, **kwargs):
        self.signal_model = signal_model
        self.background_model = background_model

        self.signal_strength = ROOT.RooRealVar(
            "signal_strength",
            "Signal Strength",
            *kwargs.get("signal_strength", (1, 0, 10))
        )

        self.signal_fraction = ROOT.RooFormulaVar(
            "signal_fraction",
            "Signal Fraction",
            f"signal_strength * (1/1000)",
            ROOT.RooArgList(self.signal_strength)
        )

        self.pdf = ROOT.RooAddPdf(
            "signal_plus_background_pdf",
            "Signal plus Background PDF",
            ROOT.RooArgList(self.signal_model.pdf, self.background_model.pdf),
            ROOT.RooArgList(self.signal_fraction)
        )
    
    def params(self):
        return [self.signal_strength] + self.signal_model.params() + self.background_model.params()

def SCAD_penalty(weights, penalty_strength=0.1):
    """
    Create a SCAD penalty term for the weights.
    """
    assert len(weights) > 0, "At least one weight is required for the penalty"

    t = penalty_strength
    a = 3.7 # SCAD parameter Fan and Li (2001)

    SCAD = lambda w: f"({w}<{t})*{t}*{w} + ({w}>{t} && {w}<={a}*{t})*TMath::Sq({a}*{t}-{w})/(2*({a}-1)) + ({w}>{a}*{t})*TMath::Sq({t})*({a}+1)/2"

    penalty_terms = []
    for w in weights:
        penalty_terms.append(SCAD(w.GetName()))
    
    penalty_str = "+".join(penalty_terms)
    penalty = ROOT.RooFormulaVar(
        "SCAD_penalty",
        "SCAD penalty for weights",
        penalty_str,
        ROOT.RooArgList(*weights)
    )
    return penalty

def weight_penalty(weights, penalty_strength=0.1):
    """
    Create a penalty term for the weights.
    """
    assert len(weights) > 0, "At least one weight is required for the penalty"

    penalty_terms = [f"{penalty_strength}*{w.GetName()}" for w in weights]
    penalty_str = "+".join(penalty_terms)
    penalty = ROOT.RooFormulaVar(
        "weight_penalty",
        "Penalty for weights",
        penalty_str,
        ROOT.RooArgList(*weights)
    )
    return penalty

def unordered_penalty(rates, weights, penalty_strength=0.1):
    """
    Create a penalty term for unordered exponential rates.
    penalty = sum_i sum_j b/(weight_i*weight_j)*1/(rates_i - rates_j)^2)
    """
    assert len(rates) > 1, "At least two rates are required for the penalty"

    n = len(rates)
    penalty_terms = []
    prefactor = lambda i, j: f"{penalty_strength}/({weights[i].GetName()}*{weights[j].GetName()})"
    square_diff = lambda i, j: f"({rates[i].GetName()}-{rates[j].GetName()})^2"
    for i in range(n):
        for j in range(i+1, n):
            penalty_terms.append(f"{prefactor(i, j)}/{square_diff(i, j)}")
    penalty_str = "+".join(penalty_terms)
    penalty = ROOT.RooFormulaVar(
        "unordered_penalty",
        "Penalty for unordered rates",
        penalty_str,
        ROOT.RooArgList(*(rates + weights))
    )
    return penalty
            
def ordered_penalty(rate_diffs, weights, penalty_strength=0.1):
    """
    Create a penalty term for ordered exponential rates.
    penalty = sum_i b/(weight_i*weight_{i+1})*1/(rate_diff_i)^2
    """
    assert len(rate_diffs) > 1, "At least two rate differences are required for the penalty"

    n = len(rate_diffs)
    penalty_terms = []
    prefactor = lambda i: f"{penalty_strength}/({weights[i].GetName()}"
    square_diff = lambda i: f"({rate_diffs[i].GetName()})^2"
    for i in range(n):
        penalty_terms.append(f"{prefactor(i)}*{square_diff(i)})")
    penalty_str = "+".join(penalty_terms)
    penalty = ROOT.RooFormulaVar(
        "ordered_penalty",
        "Penalty for ordered rates",
        penalty_str,
        ROOT.RooArgList(*(rate_diffs + weights))
    )
    return penalty

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

def fit_random_subsets(t_mgg, n_subsets=1000):
    pass
    #     """
    #     Fit many random subsets of the data to estimate the rate parameter.
    #     """
    #     rates = []
    #     for _ in range(n_subsets):
    #         rate = fit_random_subset(t_mgg)
    #         if rate is not None:
    #             rates.append(rate)
        
    #     return rates
    # # Fit many random subsets
    # import multiprocessing as mp

    # with mp.Pool(12) as pool:
    #     rates = pool.map(fit_random_subset, [t_mgg] * n_subsets)

    # rates = [rate for rate in rates if rate is not None]  # Filter out failed fits

    
def fit(model, data, penalty=None, minos=False, hesse=False, quiet=False, return_nll=False, return_fit_result=False):
    t1 = time.time()
    if penalty is not None:
        nll_base = model.pdf.createNLL(data)
        # nll = ROOT.RooFormulaVar(
        #     "nll_penalty", "nll + penalty",
        #     "@0 + @1", ROOT.RooArgList(nll_base, penalty)
        # )
        nll = ROOT.RooAddition("nll_penalty", "nll + penalty", ROOT.RooArgList(nll_base, penalty))
    else:
        nll = model.pdf.createNLL(data)
    
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.migrad()
    # minimizer.simplex()
    # minimizer.setMinimizerType("scan")
    if quiet:
        minimizer.setPrintLevel(-1) # Set print level: -1 (quiet), 0 (minimal), 1 (normal), 2 (verbose)

    # minimizer.setStrategy(2)  # Set strategy: 0 (speed), 1 (balance), 2 (robust)
    minimizer.minimize("Minuit2", "migrad")
    # minimizer.minimize("Minuit2", "simplex")

    if minos:
        minimizer.minos()
    elif hesse:
        minimizer.hesse()

    if return_nll:
        return nll.getVal()

    if return_fit_result:
        fit_result = minimizer.save()
        t2 = time.time()
        print(f"Fitted in {t2 - t1:.2f} seconds")
        print(f"Fit status: {fit_result.status()}, covQual: {fit_result.covQual()}")
        # Fit status codes:
        #    status = 0    : OK
        #    status = 1    : Covariance was mad  epos defined
        #    status = 2    : Hesse is invalid
        #    status = 3    : Edm is above max
        #    status = 4    : Reached call limit
        #    status = 5    : Any other failure
        # Covariance quality codes:
        #    covQual = 0   : No covariance matrix
        #    covQual = 1   : Diagonal approximation, not accurate
        #    covQual = 2   : Full matrix, forced positive-definite
        #    covQual = 3   : Full matrix, accurate

        # if minos:
        #     print(f"Minos status: {minimizer.MinosStatus()}")
        print(f"NLL: {fit_result.minNll()}")

        if penalty:
            print(f"Penalty: {penalty.getVal()}")

        return fit_result

def fit_and_plot(
    model, data, x,
    fit_result=None,
    minos=False,
    penalty=None,
    print_pars=True,
    **kwargs):

    if fit_result is None:
        fit_result = fit(model, data, minos=minos, penalty=penalty, quiet=True, return_fit_result=True)

    plot_fits(
        data, x,
        [model],
        [model.name],
        [fit_result, None, None],
        **kwargs
    )
    plot_correlation_matrix(fit_result)
    if hasattr(model, 'print') and print_pars:
        model.print()

def plot_fits(
    data, x, #bins,
    models,
    model_labels,
    fit_results,
    logx=False,
    nbins=128,
    # colors = [ROOT.kP6Blue, ROOT.kP6Yellow, ROOT.kP6Red, ROOT.kP6Grape],
    colors = [ROOT.kBlue, ROOT.kYellow, ROOT.kRed]
    ):
    # colors = [hex_to_tcolor(c) if isinstance(c, str) else c for c in colors]

    c = ROOT.TCanvas(random_string(), "canvas", 1600, 800)
    c.cd()

    # Set binning
    binning = ROOT.RooBinning(nbins, x.getMin(), x.getMax())
    x.setBinning(binning)

    # Frames
    main_frame = x.frame(ROOT.RooFit.Title("Diphoton Mass Fit"))
    main_frame.GetYaxis().SetTitleSize(0.05)
    main_frame.GetYaxis().SetTitleOffset(0.6)

    pull_frame = x.frame(ROOT.RooFit.Title("Pull"))
    pull_frame.SetTitle("")  # Remove title
    pull_frame.GetYaxis().SetTitleSize(0.12)
    pull_frame.GetYaxis().SetTitleOffset(0.23)
    pull_frame.GetYaxis().SetTitle("Pull")
    pull_frame.GetYaxis().SetLabelSize(0.08)
    pull_frame.GetYaxis().SetNdivisions(1*4 + 100*0 + 10000*0)  # 3 primary, 6 secondary, 0 tertiary
    pull_frame.GetYaxis().SetRangeUser(-4.5, 4.5)

    pull_frame.GetXaxis().SetTitleSize(0.125)
    pull_frame.GetXaxis().SetLabelSize(0.08)
    pull_frame.GetXaxis().SetTitle("m_{#gamma#gamma} [GeV]")

    # Create a legend
    legend = ROOT.TLegend(0.55, 0.65, 0.89, 0.89)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # Transparent legend background

    # Plot data and fits
    data.plotOn(main_frame)
    legend.AddEntry(data, "Data", "p")
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
        
        curve_obj = main_frame.findObject(model_label)
        if curve_obj:
            legend.AddEntry(curve_obj, model_label, "l")

        # Pull
        pull_hist = main_frame.pullHist()
        pull_hist.SetLineColor(colors[i])
        pull_hist.SetMarkerColor(colors[i])

        pull_frame.addPlotable(pull_hist, "P")

    # Plot the histograms
    main_pad = ROOT.TPad("main_pad", "Main Pad", 0, 0.3, 1, 1)
    pull_pad = ROOT.TPad("pull_pad", "Pull Pad", 0, 0, 1, 0.3)

    main_pad.SetLogy()
    if logx:
        main_pad.SetLogx()
    main_pad.SetBottomMargin(0)
    main_pad.Draw()

    if logx:
        pull_pad.SetLogx()
    pull_pad.SetTopMargin(0)
    pull_pad.SetBottomMargin(0.35)
    pull_pad.Draw()

    main_pad.cd()
    main_frame.SetMinimum(1.1e-3)
    main_frame.Draw()

    legend.Draw()
    ROOT.SetOwnership(legend, False)

    pull_pad.cd()
    pull_frame.Draw()

    c.Update()
    # c.Modified()
    c.Draw()


def plot_correlation_matrix(fit_result, title="Correlation Matrix", save_path=None):
    """
    Plot the correlation matrix from a RooFit result using matplotlib.
    """
    corr_matrix = fit_result.correlationMatrix()
    
    # Get parameter names from the fit result
    param_names = []
    for i in range(fit_result.floatParsFinal().getSize()):
        param = fit_result.floatParsFinal().at(i)
        param_names.append(param.GetName())
    
    # Convert ROOT matrix to numpy array
    n_params = corr_matrix.GetNrows()
    corr_array = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            corr_array[i, j] = corr_matrix[i][j]
    
    # Create matplotlib figure
    fac = n_params/4
    fig, ax = plt.subplots(figsize=(4*fac, 3*fac))
    
    # Create heatmap
    im = ax.imshow(corr_array, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add text annotations
    for i in range(n_params):
        for j in range(n_params):
            value = corr_array[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=text_color, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    # Set title and layout
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig

from typing import List, Dict

def fit_points(ws_cache, points: List[Dict], profile: bool):
    import ROOT
    import pandas as pd

    # Load workspace from cache
    fin = ROOT.TFile(ws_cache, "READ")
    ws = fin.Get("ws")
    data = ws.data("data")
    pdf = ws.pdf("pdf")
    fin.Close()

    defaults = {par.GetName(): par.getVal() for par in pdf.getParameters(data)}
    
    nll = pdf.createNLL(data)

    new_points = []
    for i, point in enumerate(points):
        for par_name, par_value in point.items():
            par = ws.var(par_name)
            par.setVal(par_value)
            if profile:
                par.setConstant(True)

        pdf.fitTo(data, PrintLevel=-1)
        new_point = point.copy()
        for par_name, par_value in point.items():
            par = ws.var(par_name)
            new_point[f"{par_name}_final"] = par.getVal()
        # new_point["nll"] = fit(model, data, penalty=penalty, quiet=True, return_nll=True)
        new_point["nll"] = nll.getVal()
        new_points.append(new_point)
    
        # Reset parameters to defaults
        for par_name, par_value in defaults.items():
            par = ws.var(par_name)
            par.setVal(par_value)
            par.setConstant(False)

    return pd.DataFrame(new_points)

def concatenate_dfs(dfs: List[pd.DataFrame]):
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    # Remove duplicates
    df = df.drop_duplicates()
    return df

profile_cache = cache.ensure_cache("profiles")
def scan_parameters(
    model, data, pset,
    constant=True, n_batches=12, use_condor=False,
    cache_name=None, remake_cache=False
    ):

    # Cache the ROOT objects in a workspace
    ws_cache_name = "tmp_workspace"
    if cache_name is not None:
        ws_cache_name = cache_name
    ws_cache = os.path.join(profile_cache, f"{ws_cache_name}.root")

    import ROOT
    ws = ROOT.RooWorkspace("ws", "ws")
    model.pdf.SetName("pdf")
    data.SetName("data")
    getattr(ws, 'import')(data)
    getattr(ws, 'import')(model.pdf)

    fout = ROOT.TFile(ws_cache, "RECREATE")
    ws.Write()
    fout.Close()

    if use_condor:
        assert cache_name is not None, "cache_file must be provided when use_condor is True"
    
    if cache_name is not None:
        cache_file = os.path.join(profile_cache, f"{cache_name}.csv")
        
        if os.path.exists(cache_file) and not remake_cache:
            import pandas as pd
            df = pd.read_csv(cache_file)
            return df


    # Profile each pair of parameters
    pset_linspaces = {
        k: np.round(np.linspace(*v),3) for k, v in pset.items()
    }
    from itertools import product
    points = [dict(zip(pset_linspaces.keys(), vals)) for vals in product(*pset_linspaces.values())]
    # Randomize order
    random.shuffle(points)
    # Split into batches
    point_batches = np.array_split(points, n_batches)

    tasks = []
    for batch in point_batches:
        tasks.append(condor.Task(fit_points, (ws_cache, batch, constant), {}))

    df = condor.run_tasks(
        tasks,
        n_cores=n_batches,
        merge_results_fn=concatenate_dfs,
        use_condor=use_condor,
        cache_results=True if use_condor else False,
        condor_job_name=cache_name,
        env_wrapper=condor.run_in_mamba,
        clear_logs=True,
    )

    # Remove nan
    df = df.dropna()

    if cache_name is not None:
        df.to_csv(cache_file, index=False)

    return df

def get_initial(n_components, worst_case=False):
    scan_file = os.path.join(profile_cache, f"{n_components}D_grid_restarts.csv")
    if os.path.exists(scan_file):
        df = pd.read_csv(scan_file)
        if worst_case:
            row = df.loc[df['nll'].idxmax()]
        else:
            row = df.loc[df['nll'].idxmin()]
        rates = { f"raw_rate_{i}": row[f"raw_rate_{i}"] for i in range(n_components) }
        return rates
    else:
        print(f"Warning: No scan file found for {n_components} components. Using default initial values.")
        return {}

def plot_2D_profile(df: pd.DataFrame, p1_name, p2_name, ax=None, fig=None, plot_contours=True, worst_case=False):

    X_vals = df[p1_name].unique()
    Y_vals = df[p2_name].unique()

    # Sort
    X_vals.sort()
    Y_vals.sort()

    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

    # Compute the minimum NLL over the other dimensions
    Z_grid = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            mask = (df[p1_name] == X_grid[i, j]) & (df[p2_name] == Y_grid[i, j])
            if not mask.any():
                print(f"Warning: No data point found for ({X_grid[i, j]}, {Y_grid[i, j]})")
            
            if worst_case:
                Z_val = df[mask]['nll'].max()
            else:
                Z_val = df[mask]['nll'].min()

            if np.isnan(Z_val):
                print(f"Warning: No NLL value found for ({X_grid[i, j]}, {Y_grid[i, j]})")
            Z_grid[i, j] = Z_val

    # Compute delta NLL
    delta_nll = Z_grid - np.nanmin(Z_grid)

    # Convert to log scale, add small offset to avoid log(0)
    # log_delta_nll = np.log(delta_nll + 1e-6)

    # Make plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Contour plot (log scale)
    cf = ax.contourf(X_grid, Y_grid, delta_nll, levels=50, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('ΔNLL')

    if plot_contours:
        # Add confidence interval contours (on original delta_nll scale)
        sigma_levels = [2.30, 6.18, 11.83]  # 1σ, 2σ, 3σ for 2D
        cl = ax.contour(X_grid, Y_grid, delta_nll, levels=sigma_levels, colors='white', linestyles='dashed')
        ax.clabel(cl, inline=True, fontsize=10)

    # Add star at the maximum likelihood point (minimum NLL)
    # Find the location of the minimum NLL (maximum likelihood)
    min_idx = np.unravel_index(np.nanargmin(Z_grid), Z_grid.shape)
    max_x = X_grid[min_idx]
    max_y = Y_grid[min_idx]
    ax.plot(max_x, max_y, '.', color='red', markersize=1)


def plot_pair_profiles(df: pd.DataFrame, pset: Dict[str, tuple], plot_contours=True, worst_case=False):
    params = list(pset.keys())
    n = len(params)
    fig, axes = plt.subplots(n-1, n-1, figsize=(6*(n-1), 6*(n-1)))
    for i, param_x in enumerate(params[:-1]):
        for j, param_y in enumerate(params[1:]):
            if n == 2:
                # Special case for 2 parameters
                ax = axes
            else:
                ax = axes[j, i]
            if i==j+1:
                ax.axis('off')
                continue
            plot_2D_profile(df, param_x, param_y, ax=ax, fig=fig, plot_contours=plot_contours, worst_case=worst_case)
            print(f"Plotting {param_x} vs {param_y} on axes ({j}, {i})")
    
            if j == n-2:
                ax.set_xlabel(param_x)
            if i == 0:
                ax.set_ylabel(param_y)
    
    return fig, axes

def get_bias_inputs(toy_model, n_toys, n_events_per_toy, n_components):

    cache_dir = ensure_cache("emm/bias")
    tags = [toy_model, f"{n_toys}toys", f"{n_events_per_toy}events", f"{n_components}exp"]
    tag = "_".join(tags)
    bias_file = os.path.join(cache_dir, f"bias_inputs_{tag}.root")
    if os.path.exists(bias_file):
        with open(bias_file, 'r') as f:
            bias_inputs = json.load(f)
        return bias_inputs
    
    print(f"Generating bias inputs for {toy_model} with {n_toys} toys and {n_events_per_toy} events each...")

    x = ROOT.RooRealVar("x", "Diphoton Mass [GeV]", 500, 4000)

    if toy_model == "Dijet":
        toy_model = Dijet(x)
    elif toy_model == "ExpPow":
        toy_model = ExpPow(x)
    else:
        raise ValueError("Unsupported toy model. Use 'Dijet' or 'ExpPow'.")

    model_to_fit = ExponentialMixtureModel(x, n_components)

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

def get_bias_info(toy_model, n_toys, n_events_per_toy_list, n_components):
    df = []
    one = False
    for n_components in n_components:
        for n_events_per_toy in n_events_per_toy_list:
            l = get_bias_inputs(
                toy_model=toy_model,
                n_toys=n_toys,
                n_events_per_toy=n_events_per_toy,
                n_components=n_components,
            )
            if l is None:
                print(f"No bias inputs found for n_components={n_components}, n_events_per_toy={n_events_per_toy}. Skipping.")
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
                'n_components': n_components,
                'n_events_per_toy': n_events_per_toy,
                'pull_mean': pull_mean,
                'pull_err': pull_err,
                'covered_percentage': covered_percentage,
            })

    # Convert to DataFrame
    df = pd.DataFrame(df)
    return df