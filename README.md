Code accompanying the paper [Modeling Falling Backgrounds with Exponential Mixtures](https://arxiv.org/abs/2607.00884).

## Overview

This repository is the working analysis code used for the fits, figures, and
simulation studies in the paper. The central implementation is a
PyROOT/RooFit-based finite exponential mixture which is applied to two high-energy physics examples: the ATLAS Run 2 dijet search and the Run 2 CMS high-mass diphoton search. 

The repository should be read primarily as a record of the study rather than as
a polished software package. The parts most likely to be useful to future work
are the notebooks in `notebooks/`, which track the examples and validation
studies in the paper, and the reusable modules in `emm/`.

## What Is Here

- `notebooks/`: analysis notebooks corresponding closely to the paper's example
	datasets and follow-up studies
- `emm/`: PyROOT/RooFit code for mixture models, fitting, plotting,
	profiling, bias studies, and coverage studies
- `data/`: public input data and auxiliary files for the ATLAS dijet and CMS
	high-mass diphoton examples
- `tools/`: helpers for ROOT utilities, storage, local multiprocessing,
	and HTCondor execution.

## Paper Map

If you are looking for a particular part of the study, the notebooks are the
fastest way in.

- `notebooks/dijet.ipynb`: the ATLAS Run 2 dijet example from Section 3.1 of
	the paper. It reads the public HEPData table in `data/dijet_ATLAS/`, builds
	the binned RooFit objects, fits the standard dijet function and exponential
	mixtures, and compares model selection and residuals. See [Search for New
	Resonances in Mass Distributions of Jet Pairs Using 139 fb^-1 of pp
	Collisions at sqrt(s)=13 TeV with the ATLAS Detector](https://dx.doi.org/10.1007/JHEP03%282020%29145)
	and [HEPData](https://doi.org/10.17182/hepdata.91126).
- `notebooks/diphoton.ipynb`: the CMS high-mass diphoton example from Section
	3.2. It loads the EBEB sample from `data/high_mass_diphoton/`, fits 1 to 4
	component exponential mixtures with random restarts, and compares them to the
	background functions used in the CMS analysis. See [Search for New Physics in
	High-Mass Diphoton Events from Proton-Proton Collisions at sqrt(s)=13
	TeV](https://dx.doi.org/10.1007/JHEP08%282024%29215) and
	[HEPData](https://doi.org/10.17182/hepdata.150677).
- `notebooks/bias.ipynb`, `notebooks/coverage.ipynb`, and
	`notebooks/spurious_signal.ipynb`: the pseudo-dataset studies from Section 4,
	covering bias, bootstrap coverage, and spurious signal.
- `notebooks/likelihood_profiling.ipynb` and
	`notebooks/global_optimization.ipynb`: supporting studies on parameter scans
	and optimization behavior.

If you only want examples to adapt, `notebooks/diphoton.ipynb` is the smallest
end-to-end example and `notebooks/dijet.ipynb` is the clearest example of the
binned workflow used in the paper.

## Code

- `emm.models`: the finite exponential mixture classes, RooFit wrappers,
	various additional models, and signal-plus-background model builders
- `emm.fitting`: maximum-likelihood fits through RooFit and MINUIT, retries and
	random restarts, information criteria, and chi-squared helpers
- `emm.profiling`: parameter-scan and profile-plot utilities
- `emm.data`: diphoton data loaders and binning helpers
- `emm.bias` and `emm.coverage`: pseudo-dataset study pipelines used in the
	validation section of the paper
- `tools/scale_out.py`: local multiprocessing and batch execution helpers for
	larger scans and simulation studies

## Setup

The environment used for the notebooks and scripts can be recreated with:

```bash
conda env create -f emm-env.yml
conda activate emm-env
python -m pip install -e .
```

## Sanity Check

From the repository root after activating the environment:

```bash
python -c "import emm.models, emm.fitting, emm.plotting, emm.profiling, emm.bias, emm.coverage, emm.data; print('OK')"
```

## Notes

- ROOT is provided by conda in `emm-env.yml`.
- The repository was developed around notebooks and batch jobs, so many
  top-level directories are analysis outputs rather than source code.
- Batch execution helpers in `tools/scale_out.py` can run locally or dispatch
	to HTCondor and TaskVine when those services are available.
