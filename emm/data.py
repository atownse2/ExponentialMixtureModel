import ROOT
import numpy as np
from array import array

from tools import root_tools
from tools import storage

top_dir = storage.top_dir
cache_dir = f"{top_dir}/cache"
data_dir = f"{top_dir}/data/high_mass_diphoton"

def get_diphoton_data(normalize=False, sort_and_index=False, tree=False):

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
