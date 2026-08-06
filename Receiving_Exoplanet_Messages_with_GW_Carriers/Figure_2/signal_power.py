"""
   Copyright 2025 Casey McGrath

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

File: signal_power.py

Purpose: Run the simulations and generate Figure 2
"""

import numpy as np
from astropy import constants as const, units as u
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


import sys
sys.path.append('../')
from functions import *



def tick_label_writer(nlist):
    labellist = []
    for n in nlist:
        if n < 0:
            if n == -1:
                labellist += [r'-$f_p$']
            else:
                labellist += [r'{0}$f_p$'.format(n)]
        if n == 0:
            labellist += [r'$f_\mathrm{GW,0}$']
        if n > 0:
            if n == 1:
                labellist += [r'+$f_p$']
            else:
                labellist += [r'+{0}$f_p$'.format(n)]
    return labellist






# -------------------------------------------------------
# Constant across all simulations 
M1   = 0.7*const.M_sun.value    # White dwarf mass 1
M2   = 0.7*const.M_sun.value    # White dwarf mass 2
Msys = M1 + M2                  # Binary system mass

Iz   = 1e38                     # mass quadrupole [m^2 kg]
ep   = 1e-6                     # ellipticity
Mtri = 1.4 * const.M_sun.value  # Neutron star mass


inc    = np.pi/2                # Exoplanet inclination
R      = 2*u.kpc.to(u.m)        # source distance
iota   = 0
psi    = 0
phi0   = 0
phi0_p = 0

norm = "backward"
# -------------------------------------------------------






plt.rcParams['font.family']      = 'serif'
plt.rcParams['font.serif']       = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # <--- default: dejavusans
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12


fig, ax = plt.subplot_mosaic([['tl','tr'],['bl','br']], figsize=(20,10))
plt.subplots_adjust(wspace=0.1)

ax['tl'].set_ylabel(r'PSD $\left[\mathrm{Hz}^{-1}\right]$', fontsize=16), ax['bl'].set_ylabel(r'PSD $\left[\mathrm{Hz}^{-1}\right]$', fontsize=16)
ax['bl'].set_xlabel(r'$f \ \left[\mathrm{mHz}\right]$',     fontsize=16), ax['br'].set_xlabel(r'$f \ \left[\mathrm{Hz}\right]$',      fontsize=16)

ax['tl'].set_title('Binary System', fontsize=20, pad=10), ax['tr'].set_title('Triaxial Body', fontsize=20, pad=10)

ax['tl'].xaxis.set_major_formatter(FormatStrFormatter('%.3f')), ax['tr'].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax['bl'].xaxis.set_major_formatter(FormatStrFormatter('%.4f')), ax['br'].xaxis.set_major_formatter(FormatStrFormatter('%.6f'))



 





# ----------------------------------------------------------------------------
# Binary System panel simulations 
# ----------------------------------------------------------------------------
# Initial GW frequency
fgw0 = 5e-3

# Observation period
t0   = 0
Tobs = 10*u.year.to(u.s)


# Compute required GW waveform parameters: Chirp-mass, GW amplitude, initial chirp rate
Mc       = Mchirp(M1, M2)
A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)

print("Binary System initial chirp rate = {0:0.2f} nHz/year".format((dfgw0*u.Hz**2).to(u.nHz/u.year).value), "\n")





# -------------------------------------------------------
# Binary #1
# -------------------------------------------------------
Mp  = 10*const.M_jup.value   # Exoplanet mass
T_p = 0.01*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the semi-amplitude
K = K_semiamp(T_p, Msys, Mp, inc)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)


# -------------- Fourier Transforms ----------------
# --> Calculate the Fourier Transform of Hplus

# Number of terms in the summation to include in the calculation
N = 100

# Frequency range to calculate over.  For a realistic simulation, the frequency resolution is 1/Tobs.
# NOTE: if any of the frequencies land exactly on the carrier frequency or one of the sideband message frequencies,
# then you get a "RuntimeWarning" due to the singularity in the Fourier transform function.  Introducing a
# very *slight* offset can help mitigate that.
freqs = np.arange(fgw0 - 20*f_p, fgw0 + 20*f_p, 1/Tobs) * 1.00000000000001

# ---------------------------------------------------------------------------------------------------
'''
Uncomment the following lines to calculate the waveform and save the data to the text file.
'''
# # Monochromatic signal
# Hp_FT_mono = M_mono_GW_FT(waveform_mono.Hp_A1, waveform_mono.Hp_A2, waveform_mono.fgw0, waveform_mono.f_p, waveform_mono.epsilon, waveform_mono.phi0, waveform_mono.phi_p, waveform_mono.t0, freqs, Tobs, N)
# # Chirping signal
# Hp_FT_chirp = M_chirp_GW_FT(waveform_chirp.Hp_A1, waveform_chirp.Hp_A2, waveform_chirp.fgw0, waveform_chirp.f_p, waveform_chirp.dfgw0, waveform_chirp.epsilon, waveform_chirp.phi0, waveform_chirp.phi_p, waveform_chirp.t0, freqs, Tobs, N)

# # -------------- PSDs ----------------
# PSD1s_mono_fromFT  = PSD_1s_fromFT(Hp_FT_mono, Tobs).real
# PSD1s_chirp_fromFT = PSD_1s_fromFT(Hp_FT_chirp, Tobs).real

# np.savetxt('./sim_data_files/Binary1_PSD1s_mono_fromFT.txt', PSD1s_mono_fromFT)
# np.savetxt('./sim_data_files/Binary1_PSD1s_chirp_fromFT.txt', PSD1s_chirp_fromFT)
# ---------------------------------------------------------------------------------------------------

PSD1s_mono_fromFT = np.loadtxt('./sim_data_files/Binary1_PSD1s_mono_fromFT.txt')
PSD1s_chirp_fromFT = np.loadtxt('./sim_data_files/Binary1_PSD1s_chirp_fromFT.txt')



# -------------- Plot ----------------
ax['tl'].semilogy(freqs*u.Hz.to(u.mHz), PSD1s_mono_fromFT,        color='#4b026c', label='Monochromatic')
ax['tl'].semilogy(freqs*u.Hz.to(u.mHz), PSD1s_chirp_fromFT, '-', color='#ffa700', label='Chirping')

# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(Mp*u.kg.to(const.M_jup))+"\n"+r"$P$ = {0:0.2f} years".format(T_p*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.4f}".format(waveform_mono.epsilon)
ax['tl'].text(0.81, 0.05, simulation_info, fontsize=12, transform=ax['tl'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

n = 4
ax['tl'].set_xlim([10**(np.log10(fgw0 - n*f_p))*u.Hz.to(u.mHz), 10**(np.log10(fgw0 + n*f_p))*u.Hz.to(u.mHz)])
ax['tl'].set_ylim([1e-54, 1e-33])

# Top x-axis
ax2 = ax['tl'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-3,-2,-1,0,1,2,3]
ax_tick_list       = [(fgw0 + n*f_p)*u.Hz.to(u.mHz) for n in n_ticks]
x_tick_list_labels = tick_label_writer(n_ticks)
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-2,0,2]
ax_tick_list = [(fgw0 + n*f_p)*u.Hz.to(u.mHz) for n in n_ticks]
ax['tl'].set_xticks(ax_tick_list), ax['tl'].minorticks_off()
ax['tl'].grid(axis='x')

# ax['tl'].legend(loc='upper right', fontsize=14, ncol=2)
ax['tl'].legend(bbox_to_anchor=(0.75,-1.35), loc='upper left', fontsize=14, ncol=2)

print("K       =", waveform_mono.K)
print("epsilon =", waveform_mono.epsilon, "\n")



# -------------------------------------------------------
# Binary #2
# -------------------------------------------------------
Mp  = 10*const.M_jup.value    # Exoplanet mass
T_p = 0.4*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the semi-amplitude
K = K_semiamp(T_p, Msys, Mp, inc)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)


# -------------- Fourier Transforms ----------------
# --> Calculate the Fourier Transform of Hplus

# Number of terms in the summation to include in the calculation
N = 100

# Frequency range to calculate over.  For a realistic simulation, the frequency resolution is 1/Tobs.
# NOTE: if any of the frequencies land exactly on the carrier frequency or one of the sideband message frequencies,
# then you get a "RuntimeWarning" due to the singularity in the Fourier transform function.  Introducing a
# very *slight* offset can help mitigate that.
freqs = np.arange(fgw0 - 20*f_p, fgw0 + 20*f_p, 1/Tobs) * 1.0000000001

# ---------------------------------------------------------------------------------------------------
'''
Uncomment the following lines to calculate the waveform and save the data to the text file.
'''
# # Monochromatic signal
# Hp_FT_mono = M_mono_GW_FT(waveform_mono.Hp_A1, waveform_mono.Hp_A2, waveform_mono.fgw0, waveform_mono.f_p, waveform_mono.epsilon, waveform_mono.phi0, waveform_mono.phi_p, waveform_mono.t0, freqs, Tobs, N)
# # Chirping signal
# Hp_FT_chirp = M_chirp_GW_FT(waveform_chirp.Hp_A1, waveform_chirp.Hp_A2, waveform_chirp.fgw0, waveform_chirp.f_p, waveform_chirp.dfgw0, waveform_chirp.epsilon, waveform_chirp.phi0, waveform_chirp.phi_p, waveform_chirp.t0, freqs, Tobs, N)

# # -------------- PSDs ----------------
# PSD1s_mono_fromFT  = PSD_1s_fromFT(Hp_FT_mono, Tobs).real
# PSD1s_chirp_fromFT = PSD_1s_fromFT(Hp_FT_chirp, Tobs).real

# np.savetxt('./sim_data_files/Binary2_PSD1s_mono_fromFT.txt', PSD1s_mono_fromFT)
# np.savetxt('./sim_data_files/Binary2_PSD1s_chirp_fromFT.txt', PSD1s_chirp_fromFT)
# ---------------------------------------------------------------------------------------------------

PSD1s_mono_fromFT = np.loadtxt('./sim_data_files/Binary2_PSD1s_mono_fromFT.txt')
PSD1s_chirp_fromFT = np.loadtxt('./sim_data_files/Binary2_PSD1s_chirp_fromFT.txt')



# -------------- Plot ----------------
ax['bl'].semilogy(freqs*u.Hz.to(u.mHz), PSD1s_mono_fromFT,        color='#4b026c')
ax['bl'].semilogy(freqs*u.Hz.to(u.mHz), PSD1s_chirp_fromFT, '-', color='#ffa700')

# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(Mp*u.kg.to(const.M_jup))+"\n"+r"$P$ = {0:0.1f} years".format(T_p*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(waveform_mono.epsilon)
ax['bl'].text(0.825, 0.79, simulation_info, fontsize=12, transform=ax['bl'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

#n = 7
ax['bl'].set_xlim([10**(np.log10(fgw0 - 4*f_p))*u.Hz.to(u.mHz), 10**(np.log10(fgw0 + 5*f_p))*u.Hz.to(u.mHz)])
ax['bl'].set_ylim([1e-45, 2e-33])

# Top x-axis
ax2 = ax['bl'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-3,-2,-1,0,1,2]
ax_tick_list       = [(fgw0 + n*f_p)*u.Hz.to(u.mHz) for n in n_ticks] + [(fgw0 + dfgw0*Tobs)*u.Hz.to(u.mHz)]
x_tick_list_labels = tick_label_writer(n_ticks) + [r'$+ \dot{f}_\mathrm{GW,0} T_\mathrm{obs}$']
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-2,0,2]
ax_tick_list = [(fgw0 + n*f_p)*u.Hz.to(u.mHz) for n in n_ticks] + [(fgw0 + dfgw0*Tobs)*u.Hz.to(u.mHz)]
ax['bl'].set_xticks(ax_tick_list), ax['bl'].minorticks_off()
ax['bl'].grid(axis='x')


print("K       =", waveform_mono.K)
print("epsilon =", waveform_mono.epsilon, "\n")








# ----------------------------------------------------------------------------
# Triaxial panel simulations 
# ----------------------------------------------------------------------------
# Initial GW frequency
fgw0 = 100

# Observation period
t0   = 0
Tobs = 4*u.year.to(u.s)      # NOTE: 10 months = 0.8333 yrs


# Compute required GW waveform parameters: Chirp-mass, GW amplitude, initial chirp rate
A, dfgw0 = Amp_tri(ep, Iz, R, fgw0), dfgw_dt_tri(ep, Iz, fgw0, 0, t0)

print("Triaxial Body initial chirp rate = {0:0.2f} nHz/year".format((dfgw0*u.Hz**2).to(u.nHz/u.year).value), "\n")






# -------------------------------------------------------
# Triaxial #1
# -------------------------------------------------------
Mp  = 10*const.M_jup.value  # Exoplanet mass
T_p = 0.01*u.year.to(u.s)  # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the semi-amplitude
K = K_semiamp(T_p, Mtri, Mp, inc)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)


# -------------- Fourier Transforms ----------------
# --> Calculate the Fourier Transform of Hplus

# Number of terms in the summation to include in the calculation
N = 200

# Frequency range to calculate over.  For a realistic simulation, the frequency resolution is 1/Tobs.
# NOTE: if any of the frequencies land exactly on the carrier frequency or one of the sideband message frequencies,
# then you get a "RuntimeWarning" due to the singularity in the Fourier transform function.  Introducing a
# very *slight* offset can help mitigate that.
freqs = np.arange(fgw0 - 150*f_p, fgw0 + 150*f_p, 1/Tobs) * 1.0000000001

# ---------------------------------------------------------------------------------------------------
'''
Uncomment the following lines to calculate the waveform and save the data to the text file.
'''
# # Monochromatic signal
# Hp_FT_mono = M_mono_GW_FT(waveform_mono.Hp_A1, waveform_mono.Hp_A2, waveform_mono.fgw0, waveform_mono.f_p, waveform_mono.epsilon, waveform_mono.phi0, waveform_mono.phi_p, waveform_mono.t0, freqs, Tobs, N)
# # Chirping signal
# Hp_FT_chirp = M_chirp_GW_FT(waveform_chirp.Hp_A1, waveform_chirp.Hp_A2, waveform_chirp.fgw0, waveform_chirp.f_p, waveform_chirp.dfgw0, waveform_chirp.epsilon, waveform_chirp.phi0, waveform_chirp.phi_p, waveform_chirp.t0, freqs, Tobs, N)

# # -------------- PSDs ----------------
# PSD1s_mono_fromFT  = PSD_1s_fromFT(Hp_FT_mono, Tobs).real
# PSD1s_chirp_fromFT = PSD_1s_fromFT(Hp_FT_chirp, Tobs).real

# np.savetxt('./sim_data_files/Triaxial1_PSD1s_mono_fromFT.txt', PSD1s_mono_fromFT)
# np.savetxt('./sim_data_files/Triaxial1_PSD1s_chirp_fromFT.txt', PSD1s_chirp_fromFT)
# ---------------------------------------------------------------------------------------------------

PSD1s_mono_fromFT = np.loadtxt('./sim_data_files/Triaxial1_PSD1s_mono_fromFT.txt')
PSD1s_chirp_fromFT = np.loadtxt('./sim_data_files/Triaxial1_PSD1s_chirp_fromFT.txt')



# -------------- Plot ----------------
ax['tr'].semilogy(freqs, PSD1s_mono_fromFT,        color='#4b026c')
ax['tr'].semilogy(freqs, PSD1s_chirp_fromFT, '-', color='#ffa700')

# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(Mp*u.kg.to(const.M_jup))+"\n"+r"$P$ = {0:0.2f} years".format(T_p*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(waveform_mono.epsilon)
ax['tr'].text(0.02, 0.05, simulation_info, fontsize=12, transform=ax['tr'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

n = 130
ax['tr'].set_xlim([10**(np.log10(fgw0 - n*f_p)), 10**(np.log10(fgw0 + n*f_p))])
ax['tr'].set_ylim([1e-52, 4e-47])

# Top x-axis
ax2 = ax['tr'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-100,-50,0,50,100]
ax_tick_list       = [(fgw0 + n*f_p) for n in n_ticks]
x_tick_list_labels = tick_label_writer(n_ticks)
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-100,-50,0,50,100]
ax_tick_list = [(fgw0 + n*f_p) for n in n_ticks]
ax['tr'].set_xticks(ax_tick_list), ax['tr'].minorticks_off()
ax['tr'].grid(axis='x')


# Create the inset graph
axins = ax['tr'].inset_axes([0.53, 0.04, 0.45, 0.45], xlim=(fgw0 - 5*f_p, fgw0 + 5*f_p), ylim=(1e-54, 1e-45)) # <--[x0, y0, width, height]
axins.semilogy(freqs, PSD1s_mono_fromFT,        color='#4b026c')
axins.semilogy(freqs, PSD1s_chirp_fromFT, '-', color='#ffa700')

n_ticks = [-4,-2,0,2,4]
ax_tick_list       = [(fgw0 + n*f_p) for n in n_ticks]
x_tick_list_labels = tick_label_writer(n_ticks)
axins.set_xticks(ax_tick_list), axins.set_xticklabels(x_tick_list_labels), axins.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, direction='in', pad=-20)
axins.set_yticks([]); axins.minorticks_off(); 
# ax['tr'].indicate_inset_zoom(axins);


print("K       =", waveform_mono.K)
print("epsilon =", waveform_mono.epsilon, "\n")






# -------------------------------------------------------
# Triaxial #2
# -------------------------------------------------------
Mp   = 1*const.M_earth.value #0.001*const.M_jup.value  # Exoplanet mass
T_p  = 0.4*u.year.to(u.s)     # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the semi-amplitude
K = K_semiamp(T_p, Mtri, Mp, inc)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)


# -------------- Fourier Transforms ----------------
# --> Calculate the Fourier Transform of Hplus

# Number of terms in the summation to include in the calculation
N = 100

# Frequency range to calculate over.  For a realistic simulation, the frequency resolution is 1/Tobs.
# NOTE: if any of the frequencies land exactly on the carrier frequency or one of the sideband message frequencies,
# then you get a "RuntimeWarning" due to the singularity in the Fourier transform function.  Introducing a
# very *slight* offset can help mitigate that.
freqs = np.arange(fgw0 - 60*f_p, fgw0 + 60*f_p, 1/Tobs) * 1.0000000001

# ---------------------------------------------------------------------------------------------------
'''
Uncomment the following lines to calculate the waveform and save the data to the text file.
'''
# # Monochromatic signal
# Hp_FT_mono = M_mono_GW_FT(waveform_mono.Hp_A1, waveform_mono.Hp_A2, waveform_mono.fgw0, waveform_mono.f_p, waveform_mono.epsilon, waveform_mono.phi0, waveform_mono.phi_p, waveform_mono.t0, freqs, Tobs, N)
# # Chirping signal
# Hp_FT_chirp = M_chirp_GW_FT(waveform_chirp.Hp_A1, waveform_chirp.Hp_A2, waveform_chirp.fgw0, waveform_chirp.f_p, waveform_chirp.dfgw0, waveform_chirp.epsilon, waveform_chirp.phi0, waveform_chirp.phi_p, waveform_chirp.t0, freqs, Tobs, N)

# # -------------- PSDs ----------------
# PSD1s_mono_fromFT  = PSD_1s_fromFT(Hp_FT_mono, Tobs).real
# PSD1s_chirp_fromFT = PSD_1s_fromFT(Hp_FT_chirp, Tobs).real

# np.savetxt('./sim_data_files/Triaxial2_PSD1s_mono_fromFT.txt', PSD1s_mono_fromFT)
# np.savetxt('./sim_data_files/Triaxial2_PSD1s_chirp_fromFT.txt', PSD1s_chirp_fromFT)
# ---------------------------------------------------------------------------------------------------

PSD1s_mono_fromFT = np.loadtxt('./sim_data_files/Triaxial2_PSD1s_mono_fromFT.txt')
PSD1s_chirp_fromFT = np.loadtxt('./sim_data_files/Triaxial2_PSD1s_chirp_fromFT.txt')



# -------------- Plot ----------------
ax['br'].semilogy(freqs, PSD1s_mono_fromFT,        color='#4b026c')
ax['br'].semilogy(freqs, PSD1s_chirp_fromFT, '-', color='#ffa700')

# plot text box info

# simulation_info = r"$M_p$ = {0:0.2f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(Mp*u.kg.to(const.M_jup))+"\n"+r"$P$ = {0:0.2f} years".format(T_p*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(waveform_mono.epsilon)
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\oplus$".format(Mp*u.kg.to(const.M_earth))+"\n"+r"$P$ = {0:0.1f} years".format(T_p*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(waveform_mono.epsilon)
ax['br'].text(0.02, 0.79, simulation_info, fontsize=12, transform=ax['br'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

# n = 10
ax['br'].set_xlim([10**(np.log10(fgw0 - 29*f_p)), 10**(np.log10(fgw0 + 3*f_p))])
ax['br'].set_ylim([1e-49, 2e-45])

# Top x-axis
ax2 = ax['br'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-2,0,2]
ax_tick_list       = [(fgw0 + n*f_p) for n in n_ticks] + [fgw0 + dfgw0*Tobs]
x_tick_list_labels = tick_label_writer(n_ticks) + [r'$+ \dot{f}_\mathrm{GW,0} T_\mathrm{obs}$']
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [0]
ax_tick_list = [(fgw0 + n*f_p) for n in n_ticks] + [fgw0 + dfgw0*Tobs]
ax['br'].set_xticks(ax_tick_list), ax['br'].minorticks_off()
ax['br'].grid(axis='x')



print("K       =", waveform_mono.K)
print("epsilon =", waveform_mono.epsilon, "\n")






# plt.savefig('./signal_power.pdf', bbox_inches='tight')
plt.show();


