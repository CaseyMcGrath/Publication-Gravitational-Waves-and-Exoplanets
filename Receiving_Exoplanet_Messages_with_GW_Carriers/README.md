# Materials Availability

This repository contains the code/calculations for reproducing all of the plots in:

<div align="center">
<strong>Receiving Exoplanet Messages with Gravitational Wave Carriers:</strong> 
</div>
<div align="center">
<strong>The Gravitational Wave Radial Velocity Method</strong>
</div>


_Publication Journal_:

_ArXiv_:



# 3rd-Pary Code and Versions:
The following 3rd-party codes are needed for running the codes in this repository.  The specific verion numbers used for the calculations presented in this paper are indicated:

- Python (3.12.9)
- NumPy (2.2.4)
- SciPy (1.15.2)
- Matplotlib (3.10.1)
- AstroPy (7.0.1)
- h5py (3.13.0)
- LISAOrbits (2.4.2)
- LISAGWResponse (2.4)
- PyTDI (1.3.1)




# Repository Structure

The relevant code used to generate each figure is organized into separate folders.

> __Figure_1__
> - ___Binary_Masses.ipynb___: A Jupyter Notebook for generating the discovery space plot for the ultra-compact binaries.
> - ___binary_types_chirpmass.pdf___: The PDF figure.

> __Figure_2__
>
> - __sim_data_files__
>     - ___LISA1_PSD1s_chirp_fromFT.txt___: Data saved of the phase chirping signal in the top left panel of the final figure.
>     - ___LISA1_PSD1s_mono_fromFT.txt___: Data saved of the monochromatic signal in the top left panel of the final figure.
>     - ___LISA2_PSD1s_chirp_fromFT.txt___: Data saved of the phase chirping signal in the bottom left panel of the final figure.
>     - ___LISA2_PSD1s_mono_fromFT.txt___: Data saved of the monochromatic signal in the bottom left panel of the final figure.
>     - ___LVK1_PSD1s_chirp_fromFT.txt___: Data saved of the phase chirping signal in the top right panel of the final figure.
>     - ___LVK1_PSD1s_mono_fromFT.txt___: Data saved of the monochromatic signal in the top right panel of the final figure.
>     - ___LVK2_PSD1s_chirp_fromFT.txt___: Data saved of the phase chirping signal in the bottom right panel of the final figure.
>     - ___LVK2_PSD1s_mono_fromFT.txt___: Data saved of the monochromatic signal in the bottom right panel of the final figure.
> - ___signal_power.pdf___: The PDF figure.
> - ___signal_power.py___: Python script for calculating and generating the PSDs of the H+ strain for each of the six systems.

> __Figure_3__
> - ___PSD_RelDiff.ipynb___: A Jupyter Notebook for generating the relative difference in the power lost (with vs. without the exoplanet) in the GW carrier frequency.
> - ___PSD_RelDiff.pdf___: The PDF figure.

> __Figure_4__
>
> - __LISAsim_data_files__
>     - ___lisasim1.h5___: Data saved from the LISA simulation used in the left and top right panels of the final figure.
>     - ___lisasim2.h5___: Data saved from the LISA simulation used in the middle right panel of the final figure.
>     - ___lisasim3.h5___: Data saved from the LISA simulation used in the bottom right panel of the final figure.
>     - ___orbits_equalarm_20350101.h5___: This file contains the data for a LISA orbit that was generated using equal-armlength orbits, given a start time of 2035/01/01.  This orbit information is read in by the _lisasim.py_ script in order to perform the LISA simulation.
> - ___lisa_sim.pdf___: The PDF figure.
> - ___lisasim_plot.py___: Python script for taking the data generated from running _lisasim.py_ and generating the final figure.
> - ___lisasim.py___: Python script for running each of the LISA simulations.  This script is run first, and generates data as it runs which is saved to the _LISAsim_data_files_ directory.  That data is used by the _lisasim_plot.py_ script to generate the final figure.

> __Figure_5__
> - ___bias_ratio.ipynb___: A Jupyter Notebook for generating the SNR bias ratio.
> - ___bias_ratio.pdf___: The PDF figure.

> __Figures_6-7__
> - ___spectral_features.ipynb___: A Jupyter Notebook for generating the two figures that demonstrate the features of frequency modulated signal spectra, for monochromatic and chirping signals.
> - ___inter-intra_signal_confusion.pdf___: The PDF figure 7.
> - ___spectral_features.pdf___: The PDF figure 6.

> ___functions.py___
>
> All of the main functions for the various formulae presented in the paper are contained in this file.  These functions are imported by the other jupyter notebooks and python scripts.

> ___lisasim_snr_calculations.ipynb___
>
> A Jupyter Notebook for performing some of the SNR calculations that were referenced throughout the paper.

