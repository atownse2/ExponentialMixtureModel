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
