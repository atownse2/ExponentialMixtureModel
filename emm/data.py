import os
import ROOT
import numpy as np
from array import array

from tools import root_tools
from tools import storage

top_dir = storage.top_dir
cache_dir = f"{top_dir}/cache"
data_dir = f"{top_dir}/data/high_mass_diphoton"

def get_diphoton_data(tree=False):
    npy_file = f"{data_dir}/diphoton_data.npy"
    
    # Load the npy file if it exists
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"{npy_file} does not exist.")

    mgg = np.load(npy_file)
    if tree:
        t_mgg = root_tools.to_root_tree([mgg], "mgg", ["x"], index=True)
        return t_mgg
    
    return mgg


def get_diphoton_binning():
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

    return bins

def get_dijet_data():
    import yaml

    yaml_file = f'{top_dir}/data/dijet_ATLAS/HEPData-ins1759712-v1-Table_2.yaml'
    # Load the YAML content
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    # 1. Extract Bin Edges from independent_variables
    # We take the 'low' from all bins and the 'high' from the last bin
    mjj_bins = data['independent_variables'][0]['values']
    bin_edges = [float(b['low']) for b in mjj_bins]
    bin_edges.append(float(mjj_bins[-1]['high']))
    bin_edges.append(10_000.0)  # Add a bin of zeros beyond the last bin edge
    
    # Convert to a double array for ROOT
    edges_array = array('d', bin_edges)
    n_bins = len(bin_edges) - 1

    # 2. Initialize Histograms
    h_observed = ROOT.TH1D("h_observed", "Observed Events;Dijet Mass [GeV];Events", n_bins, edges_array)
    h_fit = ROOT.TH1D("h_fit", "Fit Results;Dijet Mass [GeV];Events", n_bins, edges_array)

    # 3. Fill Observed Data (First dependent variable)
    obs_data = data['dependent_variables'][0]['values']
    obs_data.append({'value': 0})  # Add a zero entry for the extra bin
    for i, entry in enumerate(obs_data):
        # ROOT bins are 1-indexed
        h_observed.SetBinContent(i + 1, float(entry['value']))
        # Poisson error is standard for observed counts if not provided
        # h_observed.SetBinError(i + 1, ROOT.TMath.Sqrt(float(entry['value'])))

    # 4. Fill Fit Data (Second dependent variable)
    fit_data = data['dependent_variables'][1]['values']
    for i, entry in enumerate(fit_data):
        val = float(entry['value'])
        err = float(entry['errors'][0]['symerror'])
        h_fit.SetBinContent(i + 1, val)
        h_fit.SetBinError(i + 1, err)

    # # 5. Save to a ROOT file
    # output_file = "HEPData_Output.root"
    # f_out = ROOT.TFile(output_file, "RECREATE")
    # h_observed.Write()
    # h_fit.Write()
    # f_out.Close()
    return h_observed, h_fit
