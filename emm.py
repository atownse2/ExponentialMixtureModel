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
emm_cache = f"{top_dir}/cache/emm"
if not os.path.exists(emm_cache):
    os.makedirs(emm_cache)

# Data
data_dir = f"{top_dir}/data/high_mass_diphoton/"

def get_data(normalize=False):
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
def stick_breaking_weights(n_components):
    """
    Generate stick-breaking weights for a mixture model.
    """
    stick_proportions = [
        ROOT.RooRealVar(
            f"v_{i}",
            f"Stick proportion {i}",
            0.5,  # Initial value
            0, 1  # Bounds
        ) for i in range(n_components - 1)
    ]
    stick_proportions.append(ROOT.RooRealVar(
        f"v_{n_components-1}",
        f"Stick proportion {n_components-1}",
        1 # Fixed to 1
    ))

    weights = []
    for i in range(n_components):
        prod_terms = [f"(1-v_{j})" for j in range(i)]
        prod_str = "*".join(prod_terms) if prod_terms else "1"
        weight = ROOT.RooFormulaVar(
                    f"weight_{i}",
                    f"Weight for component {i}",
                    f"v_{i} * {prod_str}",
                    ROOT.RooArgList(*stick_proportions[:i+1])
        )
        weights.append(weight)
    
    return weights, stick_proportions

def normalization_weights(n_components):
    """
    Generate normalization construction weights for a mixture model.
    """
    unnormalized_weights = [
        ROOT.RooRealVar(
            f"unnormalized_weight_{i}",
            f"Unnormalized weight for component {i}",
            1, 0, 1000
        ) for i in range(n_components)
    ]
    unnormalized_weights[-1].setConstant(True)  # Last weight is fixed to 1

    sum_weights_str = "+".join([w.GetName() for w in unnormalized_weights])
    weights = [
        ROOT.RooFormulaVar(
            f"weight_{i}",
            f"Weight for component {i}",
            f"{unnormalized_weights[i].GetName()}/({sum_weights_str})",
            ROOT.RooArgList(*unnormalized_weights)
        ) for i in range(n_components)
    ]
    return weights, unnormalized_weights


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

class MixtureModel:
    name = "GenericMixtureModel"
    def __init__(self, x, n_components, **kwargs):
        self.n_components = n_components
        self.kwargs = kwargs
        
        self.init_weights()
        self.init_pdfs(x)
        self.pdf = mixture_pdf(self.weights, self.pdfs, name="pdf")
    
    def init_weights(self):
        if "use_normalization_construction" in self.kwargs and self.kwargs["use_normalization_construction"]:
            self.weights, self.raw_weights = normalization_weights(self.n_components)
        else:
            self.weights, self.raw_weights = stick_breaking_weights(self.n_components)
    
    def init_pdfs(self, x):
        """
        Initialize the PDFs for the mixture model.
        The PDF names should be in the format "pdf_{i}" where i is the index of the component.
        """
        raise NotImplementedError("Subclasses must implement init_pdfs")
    
    def print_base(self):
        """
        Print the model parameters.
        """
        print(f"Model: {self.name}")
        print(f"Number of components: {self.n_components}")
        print("Weights:")
        for w in self.weights:
            print(f"  {w.GetName()}: {w.getVal()} ± {w.getError() if hasattr(w, 'getError') else 'N/A'}")
    
    def print(self):
        raise NotImplementedError("Subclasses must implement print method")
        

class ExponentialMixtureModel(MixtureModel):
    name = "ExponentialMixtureModel"

    def init_rates(self, x):
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        self.raw_rates = [
            ROOT.RooRealVar(
                f"raw_rate_{i}",
                f"Raw rate for exponential {i}",
                i, 0, 1000
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
    
    def print(self):
        self.print_base()
        print("Raw Rates:")
        for r in self.raw_rates:
            print(f"  {r.GetName()}: {r.getVal()} ± {r.getError() if hasattr(r, 'getError') else 'N/A'}")

class ExponentialMixtureModel_Ordered(ExponentialMixtureModel):
    def init_rates(self, x):
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        self.raw_rate_diffs = [
            ROOT.RooRealVar(
                f"raw_rate_diff_{i}",
                f"Unordered rate difference {i}",
                1, 0, 1000
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

    def print(self):
        self.print_base()
        print("Raw Rate Differences:")
        for r in self.raw_rate_diffs:
            print(f"  {r.GetName()}: {r.getVal()} ± {r.getError() if hasattr(r, 'getError') else 'N/A'}")

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

def fit(model, data, penalty=None):
    if penalty is not None:
        nll_base = model.pdf.createNLL(data)
        nll = ROOT.RooFormulaVar(
            "nll_penalty", "nll + penalty",
            "@0 + @1", ROOT.RooArgList(nll_base, penalty)
        )
    else:
        nll = model.pdf.createNLL(data)
    
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.minimize("Minuit2", "migrad")
    minimizer.hesse()
    # minimizer.minos()

    fit_result = minimizer.save()

    if penalty:
        print(nll_base.getVal(), penalty.getVal(), nll.getVal())

    return fit_result

def fit_and_plot(model, data, x, fit_result=None, penalty=None):

    if fit_result is None:
        fit_result = fit(model, data, penalty=penalty)

    plot_fits(
        data, x,
        [model],
        [model.name],
        [fit_result, None, None],
        logx=True,
        nbins=128,
    )
    plot_correlation_matrix(fit_result)
    if hasattr(model, 'print'):
        model.print()


def plot_fits(
    data, x, #bins,
    models,
    model_labels,
    fit_results,
    logx=False,
    nbins=128,
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kMagenta, ROOT.kCyan],
    ):

    c = ROOT.TCanvas(random_string(), "canvas", 1600, 800)
    c.cd()

    # Set binning
    binning = ROOT.RooBinning(nbins, x.getMin(), x.getMax())
    x.setBinning(binning)

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
                ROOT.RooFit.MoveToBack(),
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
    cache_file = f"{emm_cache}/fit.png"
    c.SaveAs(cache_file)

    # Show png from cache file
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Replace this with your actual file path
    img = mpimg.imread(cache_file)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(img)
    ax.axis('off')  # Optional: hides axis ticks and labels
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
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