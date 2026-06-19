import os
import time

from typing import List, Dict

import json
import textwrap

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
from tools import scale_out as so
from tools import storage
from tools import combine
emm_cache = storage.ensure_cache("emm")
plot_cache = storage.ensure_cache("plots")
profile_cache = storage.ensure_cache("profiles")
workspace_cache = storage.ensure_cache("workspaces")

# top_dir = "/project01/ndcms/atownse2/ExponentialMixtureModel"
top_dir = storage.top_dir
cache_dir = f"{top_dir}/cache"
data_dir = f"{top_dir}/data/high_mass_diphoton"

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
                custom_ranges.get(f"raw_weight_{i}", 0+i),
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

def zero_one_weights(n_components: int):
    weights = [
        ROOT.RooRealVar(
            f"weight_{i}",
            f"Weight for component {i}",
            1/2,
            0,
            1
        ) for i in range(n_components-1)
    ]
    # Fix the last weight to be 1 - sum of others
    # weights[-1].setConstant(True)
    return weights, []

def mixture_pdf(weights, pdfs, name="pdf"):
    assert len(weights) == len(pdfs), "Weights and PDFs must have the same length"
    # n = len(weights)

    # pdf_terms = [f"{w.GetName()}*{p.GetName()}" for w, p in zip(weights, pdfs)]
    # pdf_str = "+".join(pdf_terms)
    # pdf = ROOT.RooGenericPdf(
    #     name,
    #     "Mixture PDF",
    #     pdf_str,
    #     ROOT.RooArgList(*(weights + pdfs))
    # )
    # print(f"Npdf: {len(pdfs)}, Nweights: {len(weights)}")

    pdf = ROOT.RooAddPdf(
        name,
        "Mixture PDF",
        ROOT.RooArgList(*pdfs),
        ROOT.RooArgList(*weights[:-1]),
        True
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

def evaluate_pdf(x, model, x_vals):
    values = np.zeros_like(x_vals)
    x.setVal(x_vals[0])
    first=model.pdf.getVal(ROOT.RooArgSet(x))
    for i, xv in enumerate(x_vals):
        x.setVal(xv)
        values[i] = model.pdf.getVal() # Without normalization
    norm = values[0] / first
    return values / norm

import re

def latex_to_root_formula(latex_str):
    """
    Converts a simple LaTeX mathematical string into a ROOT TFormula string.
    """
    root_str = latex_str
    
    # 1. Handle fractions: \frac{numerator}{denominator} -> ((numerator)/(denominator))
    # Note: This handles single-level fractions. Nested fractions would require a more complex parser.
    root_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', root_str)
    
    # 2. Handle exponents with braces: x^{2y} -> x^(2y)
    root_str = re.sub(r'\^\{([^}]+)\}', r'^(\1)', root_str)
    
    # 3. Handle square roots: \sqrt{x} -> sqrt(x)
    root_str = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', root_str)
    
    # 4. Remove backslashes for standard functions and Greek letters
    # Example: \sin(x) -> sin(x), \alpha -> alpha, \exp(-x) -> exp(-x)
    root_str = re.sub(r'\\([a-zA-Z]+)', r'\1', root_str)
    
    # 5. Convert any remaining structural LaTeX braces to standard parentheses
    # (ROOT requires parentheses for grouping)
    root_str = root_str.replace('{', '(').replace('}', ')').replace("$", "")
    
    # 6. Clean up unnecessary spacing
    root_str = root_str.strip()
    
    return root_str

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
        if name_value_dict is None:
            return
        for name, value in name_value_dict.items():
            self.set_param(name, value, constant=constant)
    
    def randomize_params(self, rng=None, custom_ranges: dict = {}):
        """
        Randomize the parameters of the model within their ranges.
        """

        if rng is None:
            rng = np.random.default_rng()

        for par in self.params():
            if not par.isConstant():
                if par.GetName() in custom_ranges:
                    min_val, max_val = custom_ranges[par.GetName()]
                else:
                    min_val = par.getMin()
                    max_val = par.getMax()
                    if min_val < -1e6:
                        min_val = -1e6
                    if max_val > 1e6:
                        max_val = 1e6
                random_val = rng.uniform(min_val, max_val)
                par.setVal(random_val)

    def print(self):
        """
        Print the model parameters.
        """
        print(f"Model: {self.name}")
        for par in self.params():
            print_par(par)

class GeneralizedPareto(RooFitModel):
    name = "GPD"
    formula = "$(N/p_1)*(1 + p_2*(x-x_min)/p_1)^{-1-1/p_2}$"

    def __init__(self, x, **kwargs):
        p1_name = "p1"
        p2_name = "p2"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        self.p1 = ROOT.RooRealVar(p1_name, p1_name, 0.5, 0, 10)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, 1000, 0.01, 10000)

        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"(1/{p2_name})*pow(1 + {p1_name}*({x.GetName()}-{x.getMin()})/{p2_name}, -1 - 1/{p1_name})",  # Formula for the PDF
            ROOT.RooArgList(self.p2, self.p1, x)  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2]


class f1(RooFitModel):
    name="f_1"
    formula = "$N x^{p_1 + p_2 \log(x)}$"
    def __init__(self, x, **kwargs):
        
        p1_name = "p1"
        p2_name = "p2"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        # self.p0 = ROOT.RooRealVar("p0", "p0", 0.13, 0.05, 0.3)  
        self.p1 = ROOT.RooRealVar(p1_name, p1_name, 5.7, 1, 11)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, -0.78, -1.0, -0.5)
        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"pow(x,{p1_name}+{p2_name}*TMath::Log(x))",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x),  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2]

class Dijet(RooFitModel):
    name="Dijet"

    def __init__(self, x, **kwargs):
        p1_name = "p1"
        p2_name = "p2"
        p3_name = "p3"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            p3_name = f"{kwargs['prefix']}_{p3_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        p1 = ROOT.RooRealVar(p1_name, p1_name, -20, -100, 100)
        p2 = ROOT.RooRealVar(p2_name, p2_name, -50, -100, 100)
        p3 = ROOT.RooRealVar(p3_name, p3_name, -10, -100, 100)
        fn = lambda x, p1, p2, p3: f"pow(1 - {x}, {p1}) * pow({x}, {p2} + {p3} * TMath::Log({x}))"
        pdf = ROOT.RooGenericPdf(
            pdf_name,
            fn(f"{x.GetName()}/13000", p1.GetName(), p2.GetName(), p3.GetName()),  # Formula for the PDF
            ROOT.RooArgList(x, p1, p2, p3)  # Arguments for the formula
        )
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        # self.p4 = p4
        self.pdf = pdf

    def params(self):
        return [self.p1, self.p2, self.p3]

class f2(RooFitModel):
    name= "f_2"
    formula = "$N\exp(p_1*x)*x^{-p_2^2}$"
    def __init__(self, x, **kwargs):
        p1_name = "p1"
        p2_name = "p2"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        self.p1 = ROOT.RooRealVar(p1_name, p1_name, -0.0016, -0.1, 0)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, 1.8, 1, 3)

        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"7e+10*exp({p1_name}*x)*pow(x,-1*{p2_name}*{p2_name})",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x)  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2]

class ExpPow3(RooFitModel):
    name= "ExpPow3"
    def __init__(self, x, **kwargs):
        p1_name = "p1"
        p2_name = "p2"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        self.p1 = ROOT.RooRealVar(p1_name, p1_name, -0.0016, -0.1, 0)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, 1.8, 1, 3)

        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"7e+10*exp({p1_name}*x)*pow(x,-1*{p2_name}*{p2_name})",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x)  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2]

class f3(RooFitModel):
    name= "f_3"
    formula = "$N(1 + p_1 x)^{-p_2^2}$"
    def __init__(self, x, **kwargs):

        p1_name = "p1"
        p2_name = "p2"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        self.p1 = ROOT.RooRealVar(p1_name, p1_name, 0.00228, 0, 0.05)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, 2.7013689, 2, 4)

        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"8760*pow(1+{p1_name}*x, -1*{p2_name}*{p2_name})",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, x)  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2]

class f4(RooFitModel):
    name = "f_4"
    formula = "$N(1 + p_1 x)^{-p_2 - p_3 x}$"
    def __init__(self, x, **kwargs):

        p1_name = "p1"
        p2_name = "p2"
        p3_name = "p3"
        pdf_name = kwargs.get("pdf_name", self.name)
        if "prefix" in kwargs:
            p1_name = f"{kwargs['prefix']}_{p1_name}"
            p2_name = f"{kwargs['prefix']}_{p2_name}"
            p3_name = f"{kwargs['prefix']}_{p3_name}"
            pdf_name = f"{kwargs['prefix']}_{pdf_name}"

        self.p1 = ROOT.RooRealVar(p1_name, p1_name, 0.029456453, 0, 0.1)
        self.p2 = ROOT.RooRealVar(p2_name, p2_name, 3.8645171, 1, 9)
        self.p3 = ROOT.RooRealVar(p3_name, p3_name, 0.00027, 0, 0.01)

        self.pdf = ROOT.RooGenericPdf(
            pdf_name,
            f"2124447*pow(1+{p1_name}*x, -{p2_name} - {p3_name}*x)",  # Formula for the PDF
            ROOT.RooArgList(self.p1, self.p2, self.p3, x)  # Arguments for the formula
        )
    
    def params(self):
        return [self.p1, self.p2, self.p3]

class MixtureModel(RooFitModel):

    def __init__(self, x, n_components: int, **kwargs):
        self.n_components = n_components
        self.name = f"Mixture-{n_components}"
        self.kwargs = kwargs

        self.init_weights()
        self.init_pdfs(x)
        self.pdf = mixture_pdf(self.weights, self.pdfs, name=kwargs.get("pdf_name", self.name))
    
    def init_weights(self, **custom_ranges):
        if self.kwargs.get("stick_breaking", False):
            self.weights, self.raw_weights = stick_breaking_weights(self.n_components, **custom_ranges)
        elif self.kwargs.get("ordered_weights", False):
            self.weights, self.raw_weights, self.ordered_weights = ordered_weights(self.n_components, **custom_ranges)
        else:
            self.weights, self.raw_weights = normalization_weights(self.n_components)
            # self.weights, self.raw_weights = zero_one_weights(self.n_components)
    
    def init_pdfs(self, x):
        """
        Initialize the PDFs for the mixture model.
        The PDF names should be in the format "pdf_{i}" where i is the index of the component.
        """
        raise NotImplementedError("Subclasses must implement init_pdfs")

class ExponentialMixtureModel(MixtureModel):
    rate_max = 50

    def randomize_params(self, custom_ranges = {}, rng=np.random.default_rng()):
        if custom_ranges == {}:

            rate = 0
            for i, raw_rate in enumerate(self.raw_rates):
                rate += rng.uniform(0.01, 3)
                if i == len(self.raw_rates) - 1:
                    rate += rng.uniform(0, 20)
                raw_rate.setVal(rate)
            for raw_weight in self.raw_weights:
                if not raw_weight.isConstant():
                    raw_weight.setVal(0)
        else:
            super().randomize_params(custom_ranges=custom_ranges)

    def integral(self, x, lo, hi):
        integral = 0
        for i in range(self.n_components):
            rate = self.rates[i].getVal()
            weight = self.weights[i].getVal()
            integral += weight*(np.exp(rate*lo) - np.exp(rate*hi))
        return integral

    def init_rates(self, x, random=False):
        self.name = f"Exponential{self.name}"
        # assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        # data_mean = self.kwargs["data_mean"]
        if "data_mean" in self.kwargs:
            data_mean = self.kwargs["data_mean"]
        else:
            raise ValueError("data_mean must be provided to initialize rates")
        rate_scaling = -1/(data_mean - x.getMin())
        # data_mean = 200
        # rate_scaling = -1/data_mean
        random_initialization = False
        if "random" in self.kwargs:
            random_initialization = self.kwargs["random"]

        # Initialize raw rates
        if "initial_raw_rates" in self.kwargs:
            initial_raw_rates = self.kwargs["initial_raw_rates"]
            assert len(initial_raw_rates) == self.n_components, "initial_raw_rates must have the same length as n_components"
        elif self.kwargs.get("random_rates", False):
            initial_raw_rates = [random.uniform(0, 1e-3 if _ == 0 else 10.0) for _ in range(self.n_components)]
            print(f"Using random initial raw rates: {initial_raw_rates}")
        else:
            # initial_raw_rates = [(i+1) for i in range(self.n_components)]
            if random_initialization:
                initial_raw_rates = [np.random.uniform(0, 5.0) for _ in range(self.n_components)]
            else:
                initial_raw_rates = [(0.1+i) for i in range(self.n_components)]

        
        raw_rate_specs = []
        for i in range(self.n_components):
            if f"raw_rate_{i}" in self.kwargs:
                spec = self.kwargs[f"raw_rate_{i}"]
                print(f"Using custom spec for raw_rate_{i}: {spec}")
                if isinstance(spec, (list, tuple)):
                    assert len(spec) == 3, "raw_rate_{i} must be a tuple of (initial, min, max)"
                    raw_rate_specs.append(spec)
                elif isinstance(spec, (int, float)):
                    raw_rate_specs.append((spec, 0, self.rate_max))
                else:
                    raise ValueError("raw_rate_{i} must be a float or a tuple of (initial, min, max)")
            else:
                raw_rate_specs.append((0.1+i, 0, self.rate_max))

        self.raw_rates = [
            ROOT.RooRealVar(
                f"raw_rate_{i}",
                f"Raw rate for exponential {i}",
                *raw_rate_specs[i]
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

        # print("Using MY Exponential PDFs for the mixture model.")
        self.pdfs = [
            ROOT.RooExponential(
            # ROOT.MyExponential(
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

class GPDMixtureModel(MixtureModel):
    def init_pdfs(self, x):
        self.init_powers()
        if "max_x" in self.kwargs:
            self.x_max = self.kwargs["max_x"]
        else:
            print(f"Warning: max_x not provided for GPDMixtureModel, using x.getMax() = {x.getMax()}")
            self.x_max = x.getMax()
        
        self.pdfs = [
            ROOT.RooGenericPdf(
                f"pdf_{i}",
                f"{self.weights[i].GetName()}*pow(1 - ({x.GetName()}-{self.x.getMin()})/({self.x_max}-{self.x.getMin()}), {self.powers[i].GetName()})",
                ROOT.RooArgList(self.weights[i], self.powers[i], x)
            ) for i in range(self.n_components)
        ]

    def init_powers(self):
        self.powers = [
            ROOT.RooRealVar(
                f"power_{i}",
                f"Power for component {i}",
                5 + i*10,
                -1,
                100_000
            ) for i in range(self.n_components)
        ]
    
    def params(self):
        """
        Return the parameters of the model.
        """
        return self.powers + self.raw_weights

class GPDMixtureModel_2(GPDMixtureModel):
    name = "GPD-Mixture-2"
    def __init__(self, x, **kwargs):
        super().__init__(x, 2, **kwargs)

class ExponentialMixtureModel_1(ExponentialMixtureModel):
    name = "ExponentialMixture-1"
    def __init__(self, x, **kwargs):
        super().__init__(x, 1, data_mean=700, **kwargs)

class ExponentialMixtureModel_2(ExponentialMixtureModel):
    name = "ExponentialMixture-2"
    def __init__(self, x, **kwargs):
        super().__init__(x, 2, data_mean=700, **kwargs)

class ExponentialMixtureModel_3(ExponentialMixtureModel):
    name = "ExponentialMixture-3"
    def __init__(self, x, **kwargs):
        super().__init__(x, 3, data_mean=700, **kwargs)

class ExponentialMixtureModel_4(ExponentialMixtureModel):
    name = "ExponentialMixture-4"
    def __init__(self, x, **kwargs):
        super().__init__(x, 4, data_mean=700, **kwargs)

class ExponentialMixtureModel_5(ExponentialMixtureModel):
    name = "ExponentialMixture-5"
    def __init__(self, x, **kwargs):
        super().__init__(x, 5, data_mean=700, **kwargs)

class ExponentialMixtureModel_6(ExponentialMixtureModel):
    name = "ExponentialMixture-6"
    def __init__(self, x, **kwargs):
        super().__init__(x, 6, data_mean=700, **kwargs)

class ExponentialMixtureModel_2_Dijet(ExponentialMixtureModel):
    name = "ExponentialMixture-2"
    def __init__(self, x, **kwargs):
        super().__init__(x, 2, data_mean=1350, **kwargs)

class ExponentialMixtureModel_3_Dijet(ExponentialMixtureModel):
    name = "ExponentialMixture-3"
    def __init__(self, x, **kwargs):
        super().__init__(x, 3, data_mean=1350, **kwargs)

class ExponentialMixtureModel_4_Dijet(ExponentialMixtureModel):
    name = "ExponentialMixture-4"
    def __init__(self, x, **kwargs):
        super().__init__(x, 4, data_mean=1350, **kwargs)

class ExponentialMixtureModel_5_Dijet(ExponentialMixtureModel):
    name = "ExponentialMixture-5"
    def __init__(self, x, **kwargs):
        super().__init__(x, 5, data_mean=1350, **kwargs)

class ExponentialMixtureModel_6_Dijet(ExponentialMixtureModel):
    name = "ExponentialMixture-6"
    def __init__(self, x, **kwargs):
        super().__init__(x, 6, data_mean=1350, **kwargs)

class ExponentialMixtureModel_Ordered(ExponentialMixtureModel):
    rate_min = 0.1
    rate_diff_min = 0.05
    def randomize_params(self, custom_ranges={}, rng=np.random.default_rng()):
        if custom_ranges == {}:
            for i, raw_rate_diff in enumerate(self.raw_rate_diffs):
                if i == 0:
                    rate_diff = rng.uniform(self.rate_min, 1)
                if i == len(self.raw_rate_diffs)-1:
                    rate_diff = rng.uniform(self.rate_diff_min, 100)
                else:
                    rate_diff = rng.uniform(self.rate_diff_min, 1)
                
                raw_rate_diff.setVal(rate_diff)
            for raw_weight in self.raw_weights:
                if not raw_weight.isConstant():
                    raw_weight.setVal(0)
        else:
            return super().randomize_params(custom_ranges, rng)

    def init_rates(self, x, random=False):
        self.name = f"ExponentialMixture-Ordered-{self.n_components}"
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        if "random" in self.kwargs:
            random = self.kwargs["random"]

        mult = 1.0
        if random:
            mult = np.random.uniform(0.5, 2.0)

        self.raw_rate_diffs = [
            ROOT.RooRealVar(
                f"raw_rate_diff_{i}",
                f"Inverse rate difference {i}",
                0.5*mult,
                self.rate_min if i==0 else self.rate_diff_min,
                100
            ) for i in range(self.n_components)
        ]

        self.rates = [
            ROOT.RooFormulaVar(
                f"rate_{i}",
                f"Ordered Rate for exponential {i}",
                f"{rate_scaling}*({'+'.join([rd.GetName() for rd in self.raw_rate_diffs[:i+1]])})",
                ROOT.RooArgList(*self.raw_rate_diffs[:i+1])
            ) for i in range(self.n_components)
        ]

    def params(self):
        """
        Return the parameters of the model.
        """
        return self.raw_rate_diffs + self.raw_weights

class ExponentialMixtureModel_Ordered_2(ExponentialMixtureModel_Ordered):
    name = "ExponentialMixture-Ordered-2"
    def __init__(self, x, **kwargs):
        super().__init__(x, 2, data_mean=700, **kwargs)

class ExponentialMixtureModel_Ordered_3(ExponentialMixtureModel_Ordered):
    name = "ExponentialMixture-Ordered-3"
    def __init__(self, x, **kwargs):
        super().__init__(x, 3, data_mean=700, **kwargs)

class ExponentialMixtureModel_Ordered_4(ExponentialMixtureModel_Ordered):
    name = "ExponentialMixture-Ordered-4"
    def __init__(self, x, **kwargs):
        super().__init__(x, 4, data_mean=700, **kwargs)

class ExponentialMixtureModel_Ordered_5(ExponentialMixtureModel_Ordered):
    name = "ExponentialMixture-Ordered-5"
    def __init__(self, x, **kwargs):
        super().__init__(x, 5, data_mean=700, **kwargs)

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
                *self.kwargs.get(f"alpha_{i}", (2+i, 0.01, 10))
            ) for i in range(self.n_components)
        ]
    
    def init_betas(self, x):
        self.betas = [
            ROOT.RooRealVar(
                f"beta_{i}", f"Scale parameter {i}",
                *self.kwargs.get(f"beta_{i}", (0.005, 0.000001, 1))
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

class LomaxMixtureModel_1(LomaxMixtureModel):
    name = "LomaxMixture-1"
    def __init__(self, x, **kwargs):
        super().__init__(x, 1, **kwargs)

class GaussianSignalModel(RooFitModel):
    name = "Gaussian"
    def __init__(self, x, mean, width, **kwargs):
        self.prefix = kwargs.get("prefix", "")
        self.mean = ROOT.RooRealVar(
            f"{self.prefix}sig_mean",
            "Signal mean",
            mean,
            0.9*mean,
            1.1*mean
        )
        self.sigma = ROOT.RooRealVar(
            f"{self.prefix}sig_sigma",
            "Signal sigma",
            width,
            0.8*width,
            1.2*width
        )
        self.pdf = ROOT.RooGaussian(
            f"{self.prefix}sig_pdf",
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

        max_sig = kwargs.get("max_sig", 100)
        n_bkg = kwargs.get("n_bkg", 5036)

        self.n_sig = ROOT.RooRealVar(
            "n_sig",
            "Number of signal events",
            0, -max_sig, max_sig
        )
        self.n_bkg = ROOT.RooRealVar(
            "n_bkg",
            "Number of background events",
            n_bkg, 0.5*n_bkg, 2*n_bkg
        )
        self.pdf = ROOT.RooAddPdf(
            "signal_plus_background_pdf",
            "Signal plus Background PDF",
            ROOT.RooArgList(self.signal_model.pdf, self.background_model.pdf),
            ROOT.RooArgList(self.n_sig, self.n_bkg)
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


fit_options = [
    # ROOT.RooFit.IntegrateBins(0.0001),
    # ROOT.RooFit.PrintLevel(-1),
    # ROOT.RooFit.Offset(True),
    # # ROOT.RooFit.Strategy(2),
    # ROOT.RooFit.Save(),
    # # ROOT.RooFit.Range("fit_range")
]

def fit_n_times(model, data, n_attempts=5, fit_options=fit_options, verbose=True):
    # Usual issue is EDM above max which I assum is because of flat likelihood surface
    # Keep the fit going at last estimates seems to help
    import ROOT
    if ROOT.RooFit.Save(True) not in fit_options:
        fit_options.append(ROOT.RooFit.Save(True))
    for attempt in range(n_attempts):
        fit_result = model.pdf.fitTo(
            data,
            *fit_options
        )
        if fit_result.status() <= 2:
            if attempt > 0 and verbose:
                print(f"Fit succeeded after {attempt} retries")
            return fit_result
    if verbose:
        print(f"Fit failed after {n_attempts} attempts, returning None" )
    return None
    # return fit_result

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

import pickle
def fit_random_restarts(
        x, data,
        model_primitive,
        seed, n_samples,
        n_retries=5,
        save=True,
        fit_options=fit_options,
        verbose=True,
    ):

    best_nll = np.inf
    fit_results = []

    rng = np.random.default_rng(seed=seed)
    for i in range(n_samples):
        model = model_primitive(x)
        model.randomize_params(rng=rng)
        initial_pars = {p.GetName(): p.getVal() for p in model.params()}

        fit_result = fit_n_times(model, data, n_attempts=n_retries, fit_options=fit_options, verbose=False)
        if fit_result is None:
            if verbose:
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
    
    if save:
        model = model_primitive(x)
        fout = random_restarts_filename(model.name, data.GetName(), n_samples, seed)
        with open(fout, "wb") as f:
            pickle.dump(fit_results, f)

    return fit_results


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

def fit(model, data, fit_args=[]):
    t1 = time.time()
    
    fit_result = model.pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        *fit_args
    )
    t2 = time.time()
    print(f"Fitted in {t2 - t1:.2f} seconds")
    print(f"Fit status: {fit_result.status()}, covQual: {fit_result.covQual()}")
    print(f"NLL: {fit_result.minNll()}")

    return fit_result

def fit_with_penalty(model, data, penalty=None, minos=False, hesse=False, quiet=False, return_nll=False, return_fit_result=False, fit_args=[]):
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
    print_pars=True,
    fit_args=[],
    **kwargs):

    if fit_result is None:
        fit_result = fit(model, data, fit_args=fit_args)

    plot_fits(
        data, x,
        [model],
        [model.name],
        **kwargs
    )
    # plot_correlation_matrix(fit_result)
    if hasattr(model, 'print') and print_pars:
        model.print()

# default_colors = ['#377eb8', '#ff7f00', '#4daf4a',
            # '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00']

default_colors = [
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    # "#000000"   # Black
]

# default_colors = [
#     "#648FFF",  # Ultramarine
#     "#785EF0",  # Indigo
#     "#DC267F",  # Magenta
#     "#FE6100",  # Orange
#     "#FFB000"   # Gold
# ]

default_root_colors = [ROOT.TColor.GetColor(c) for c in default_colors]

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

def plot_information_criteria(
    criteria_by_x,
    x_values=None,
    figsize=(8, 3),
    x_label="Number of Components (k)",
    aic_key="AIC",
    bic_key="BIC",
    aic_label="$\Delta\\text{AIC}$",
    bic_label="$\Delta\\text{BIC}$",
    aic_color="red",
    bic_color="blue",
    bic_marker="x",
    aic_marker_size=100,
    bic_marker_size=100,
    label_size=18,
    tick_label_size=None,
    legend_label_size=None,
    legend_loc="upper center",
    sort_x=True,
    show=True,
):
    if isinstance(criteria_by_x, dict):
        items = list(criteria_by_x.items())
    else:
        if x_values is None:
            raise ValueError("x_values is required when criteria_by_x is not a dictionary")
        items = list(zip(x_values, criteria_by_x))

    if not items:
        raise ValueError("No information criteria were provided")

    if sort_x:
        try:
            items = sorted(items, key=lambda item: item[0])
        except TypeError:
            pass

    plot_values = [item[0] for item in items]
    aic_values = []
    bic_values = []
    for x_value, criteria in items:
        if aic_key not in criteria or bic_key not in criteria:
            raise KeyError(
                f"Missing {aic_key} or {bic_key} for {x_value}"
            )
        aic_values.append(criteria[aic_key])
        bic_values.append(criteria[bic_key])

    aic_values = np.array(aic_values)
    bic_values = np.array(bic_values)

    aic_values -= np.min(aic_values)
    bic_values -= np.min(bic_values)

    use_numeric_axis = all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in plot_values
    )
    if use_numeric_axis:
        x_coords = plot_values
        x_tick_labels = None
    else:
        x_coords = np.arange(len(plot_values))
        x_tick_labels = plot_values

    fig, ax1 = plt.subplots(1, figsize=figsize)

    ax1.scatter(x_coords, aic_values, color=aic_color, label=aic_label, s=aic_marker_size)
    ax1.set_xlabel(x_label, fontsize=label_size)
    ax1.set_ylabel(aic_label, color=aic_color, fontsize=label_size)
    ax1.tick_params(axis='y', labelcolor=aic_color, labelsize=tick_label_size)
    ax1.set_xticks(x_coords)
    ax1.tick_params(axis='x', labelsize=tick_label_size)
    if x_tick_labels is not None:
        ax1.set_xticklabels(x_tick_labels)

    ax2 = ax1.twinx()
    ax2.scatter(
        x_coords,
        bic_values,
        color=bic_color,
        label=bic_label,
        marker=bic_marker,
        s=bic_marker_size,
    )
    ax2.set_ylabel(bic_label, color=bic_color, fontsize=label_size)
    ax2.tick_params(axis='y', labelcolor=bic_color, labelsize=tick_label_size)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc=legend_loc,
        fontsize=legend_label_size,
    )

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax1, ax2

def roo_hist_to_th1(roo_hist, name):
    n_points = roo_hist.GetN()
    x_values = list(roo_hist.GetX())
    y_values = list(roo_hist.GetY())

    if n_points == 0:
        return ROOT.TH1D(name, "", 1, 0.0, 1.0)

    if n_points == 1:
        width = 1.0
        edges = [x_values[0] - width / 2.0, x_values[0] + width / 2.0]
    else:
        edges = [x_values[0] - (x_values[1] - x_values[0]) / 2.0]
        for i in range(n_points - 1):
            edges.append((x_values[i] + x_values[i + 1]) / 2.0)
        edges.append(x_values[-1] + (x_values[-1] - x_values[-2]) / 2.0)

    hist = ROOT.TH1D(name, "", len(edges) - 1, array('d', edges))
    hist.SetDirectory(0)
    for i, value in enumerate(y_values):
        hist.SetBinContent(i + 1, value)
    return hist

def plot_fits(
    data, x, #bins,
    models,
    labels,
    title=None,
    logx=False,
    x_label=None,
    nbins=None,
    plot_range=None,
    pull_range=(-4, 4),
    binning=None,
    colors = default_root_colors,
    linestyles=None,
    y_min=1.1e-1,
    markersize=1.5,
    linewidth=2,
    title_size=0.12, label_size=0.09,
    logy=True,
    legend_bounds=(0.55, 0.56, 0.95, 0.85),
    legend_columns=2,
    legend_text_size=0.064,
    legend_margin=0.18,
    pull_fill_style=1001,
    pull_fill_alpha=0.3,
    ):

    c = ROOT.TCanvas(random_string(), "canvas", 1600, 800)
    c.cd()

    # Set binning
    # x.setRange(plot_range)
    # Get range from x (RooRealVar)
    # if plot_range:
    #     x.setRange(*plot_range)
    if nbins is not None:
        x_min = x.getMin()
        x_max = x.getMax()
        if plot_range is not None:
            x_min, x_max = plot_range
        binning = ROOT.RooBinning(nbins, x_min, x_max)
        x.setBinning(binning)

    # Frames
    main_frame = x.frame()

    if title is not None:
        main_frame.SetTitle(title)
        main_frame.SetTitleSize(0.08)
    else:
        main_frame.SetTitle("")

    main_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())
    main_frame.GetYaxis().SetTitle("Events / bin")
    main_frame.GetYaxis().CenterTitle(True)
    main_frame.GetYaxis().SetTitleSize(title_size)
    main_frame.GetYaxis().SetTitleOffset(0.42)
    main_frame.GetYaxis().SetLabelSize(label_size)

    pull_frame = x.frame()
    pull_frame.SetTitle("")  # Remove title
    pull_frame.GetYaxis().SetTitleSize(title_size)
    pull_frame.GetYaxis().SetTitleOffset(0.37)
    pull_frame.GetYaxis().SetTitle("#frac{data - fit}{#sigma_{data}}")
    pull_frame.GetYaxis().CenterTitle(True)
    pull_frame.GetYaxis().SetLabelSize(label_size)
    pull_frame.GetYaxis().SetNdivisions(104, False)

    # Add dashed line at y=0
    min_x = x.getMin()
    max_x = x.getMax()
    if plot_range is not None:
        min_x, max_x = plot_range
    line = ROOT.TLine(min_x, 0, max_x, 0)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    pull_frame.addObject(line)

    pull_frame.GetXaxis().SetTitleSize(title_size + 0.02)
    pull_frame.GetXaxis().SetTitleOffset(1.1)
    pull_frame.GetXaxis().CenterTitle(True)
    pull_frame.GetXaxis().SetLabelSize(label_size + 0.01)
    pull_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())

    
    if plot_range is not None:
        main_frame.GetXaxis().SetRangeUser(*plot_range)
        pull_frame.GetXaxis().SetRangeUser(*plot_range)

    if logx:
        # Improve readability on log-x by drawing labels on intermediate ticks.
        main_frame.GetXaxis().SetMoreLogLabels(True)
        pull_frame.GetXaxis().SetMoreLogLabels(True)
        main_frame.GetXaxis().SetNoExponent(True)
        pull_frame.GetXaxis().SetNoExponent(True)

    # Plot data and fits
    data_reference_name = "plot_data_reference"
    data_name = "plot_data"
    data.plotOn(
        main_frame,
        ROOT.RooFit.Name(data_reference_name),
        ROOT.RooFit.MarkerSize(0),
        ROOT.RooFit.LineColor(0),
    )

    curve_specs = []
    pull_specs = []
    for i, model in enumerate(models):
        if hasattr(model, "pdf"):
            pdf = model.pdf
        else:
            pdf = model

        model_label = labels[i]
        if colors is not None:
            if len(colors) == 0:
                raise ValueError("colors must not be empty when provided")
            model_color = colors[i % len(colors)]
        else:
            model_color = colors[i % len(colors)]
        if isinstance(model_color, str):
            model_color = ROOT.TColor.GetColor(model_color)

        if linestyles is not None:
            if len(linestyles) == 0:
                raise ValueError("linestyles must not be empty when provided")
            line_style = linestyles[i % len(linestyles)]
            if isinstance(line_style, str):
                line_style_lookup = {
                    "solid": 1,
                    "dashed": 2,
                    "dotted": 3,
                    "dashdotted": 4,
                    "dashdot": 4,
                }
                line_style = line_style_lookup.get(line_style.strip().lower())
                if line_style is None:
                    raise ValueError(f"Unsupported line style: {model_linestyles[i % len(model_linestyles)]}")
        else:
            line_style = ROOT.kSolid

        curve_name = f"plot_curve_{i}"

        pdf.plotOn(
            main_frame,
            ROOT.RooFit.Precision(1e-5),
            ROOT.RooFit.LineColor(model_color),
            ROOT.RooFit.LineStyle(line_style),
            ROOT.RooFit.LineWidth(int(linewidth)),
            ROOT.RooFit.Name(curve_name),
            ROOT.RooFit.DrawOption("L"),  # Use "L" for line only
        )
        curve_specs.append((curve_name, model_label, model_color))

    data.plotOn(
        main_frame,
        ROOT.RooFit.Name(data_name),
        ROOT.RooFit.MarkerSize(markersize),
    )

    for curve_name, model_label, model_color in curve_specs:
        pull_hist = main_frame.pullHist(data_reference_name, curve_name)
        pull_hist.SetLineColor(model_color)
        pull_hist.SetMarkerColor(model_color)
        pull_hist.SetMarkerSize(markersize)
        pull_hist.SetLineWidth(int(linewidth))
        pull_specs.append((pull_hist, model_color, pull_fill_style))

    # Plot the histograms
    main_pad = ROOT.TPad("main_pad", "Main Pad", 0, 0.5, 1, 1)
    pull_pad = ROOT.TPad("pull_pad", "Pull Pad", 0, 0, 1, 0.5)

    if logy:
        main_pad.SetLogy()
    if logx:
        main_pad.SetLogx()
    main_pad.SetTopMargin(0.12)
    main_pad.SetLeftMargin(0.12)
    main_pad.SetRightMargin(0.04)
    main_pad.SetBottomMargin(0)
    main_pad.Draw()

    if logx:
        pull_pad.SetLogx()
    pull_pad.SetLeftMargin(0.12)
    pull_pad.SetRightMargin(0.04)
    pull_pad.SetTopMargin(0)
    pull_pad.SetBottomMargin(0.38)
    pull_pad.Draw()

    legend_x1, legend_y1, legend_x2, legend_y2 = legend_bounds
    legend = ROOT.TLegend(legend_x1, legend_y1, legend_x2, legend_y2)
    legend.SetNColumns(legend_columns)
    legend.SetTextFont(42)
    legend.SetTextSize(legend_text_size)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # Transparent legend background
    legend.SetMargin(legend_margin)

    data_obj = main_frame.findObject(data_name)
    if data_obj:
        legend.AddEntry(data_obj, "Data", "lp")

    for curve_name, model_label, model_color in curve_specs:
        curve_obj = main_frame.findObject(curve_name)
        if curve_obj:
            legend.AddEntry(curve_obj, model_label, "l")

    main_pad.cd()
    main_frame.SetMinimum(y_min)
    if logy:
        main_frame.SetMaximum(main_frame.GetMaximum() * 10)
    else:
        main_frame.SetMaximum(main_frame.GetMaximum() * 1.2)
    main_frame.Draw()

    legend.Draw()
    ROOT.SetOwnership(legend, False)

    pull_pad.cd()
    pull_frame.Draw()

    pull_hist_frame = next(
        (prim for prim in pull_pad.GetListOfPrimitives() if isinstance(prim, ROOT.TH1)),
        None,
    )
    if pull_hist_frame is not None:
        pull_hist_frame.SetMinimum(pull_range[0])
        pull_hist_frame.SetMaximum(pull_range[1])
        pull_hist_frame.GetYaxis().SetNdivisions(104, False)
        pull_hist_frame.GetYaxis().SetTitle("#frac{data - fit}{#sigma_{data}}")
        pull_hist_frame.GetYaxis().CenterTitle(True)
        pull_hist_frame.GetYaxis().SetTitleSize(title_size)
        pull_hist_frame.GetYaxis().SetTitleOffset(0.37)
        pull_hist_frame.GetYaxis().SetLabelSize(label_size)
        pull_hist_frame.GetXaxis().SetTitle(x_label if x_label is not None else x.GetTitle())
        pull_hist_frame.GetXaxis().SetTitleSize(title_size + 0.02)
        pull_hist_frame.GetXaxis().SetTitleOffset(1.1)
        pull_hist_frame.GetXaxis().CenterTitle(True)
        pull_hist_frame.GetXaxis().SetLabelSize(label_size + 0.01)

    filled_pull_hists = []
    for i, (pull_hist, model_color, fill_style) in enumerate(pull_specs):
        filled_hist = roo_hist_to_th1(pull_hist, f"pull_fill_{i}_{random_string()}")
        # filled_hist.SetMinimum(
        # filled_hist.SetMaximum(4.0)
        filled_hist.SetLineColor(model_color)
        filled_hist.SetLineWidth(1)
        filled_hist.SetFillColorAlpha(model_color, pull_fill_alpha)
        filled_hist.SetFillStyle(fill_style)
        filled_hist.Draw("HIST SAME")
        filled_pull_hists.append(filled_hist)

    line.Draw("SAME")

    c._pull_filled_hists = filled_pull_hists
    c._pull_line = line

    c.Update()
    # c.Modified()
    c.Draw()

    return c
    # # Save and draw
    # png_filename = f"{plot_cache}/fit_{random_string()}.png"

    # c.SaveAs(png_filename)
    # print(f"Saved fit to {png_filename}")
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # img = plt.imread(png_filename)
    # ax.imshow(img)
    # ax.axis('off')
    # plt.show()

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
        tasks.append(so.Task(fit_points, ws_cache, batch, constant))

    df = so.run_tasks(
        tasks,
        n_cores=n_batches,
        merge_results_fn=concatenate_dfs,
        use_condor=use_condor,
        cache_results=True if use_condor else False,
        condor_job_name=cache_name,
        env_wrapper=so.run_in_mamba,
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

def plot_2D_profile(
        df: pd.DataFrame,
        p1_name, p2_name,
        ax=None, fig=None,
        plot_contours=True,
        worst_case=False,
        logz=False,
        random_restarts=None,
    ):

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
    if logz:
        cf = ax.contourf(X_grid, Y_grid, np.log1p(delta_nll + 1e-6), levels=50, cmap='viridis')
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('log10(ΔNLL)')
    else:
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

    # Add random restart points if provided
    if random_restarts is not None:
        for i, fit_result in enumerate(random_restarts):
            x_val = fit_result['final_pars'][p1_name]
            y_val = fit_result['final_pars'][p2_name]

            if i == len(random_restarts) - 1:
                ax.plot(x_val, y_val, 'x', color='red', markersize=5, label='Best Solution')
            else:
                ax.plot(x_val, y_val, 'x', color='black', markersize=5, label="A Solution")
        ax.legend()

def plot_pair_profiles(
        df: pd.DataFrame,
        pset: Dict[str, tuple],
        plot_contours=True,
        worst_case=False,
        logz=False,
    ):
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
            plot_2D_profile(df, param_x, param_y, ax=ax, fig=fig, plot_contours=plot_contours, worst_case=worst_case, logz=logz)
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
