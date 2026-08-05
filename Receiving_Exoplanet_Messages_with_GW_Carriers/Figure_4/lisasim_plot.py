#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:43:04 2025

@author: cdmcgrat
"""

import numpy as np
from astropy import constants as const, units as u
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import h5py



import sys
sys.path.append('../')
from functions import PSD_1s



# Convenient class for making the Legend 'patch' appear with both the line and fill:
# --> https://stackoverflow.com/questions/69836527/create-a-rectangular-patch-with-upper-and-lower-edge-in-matplotlib
class HandlerFilledBetween(mpl.legend_handler.HandlerPolyCollection):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)[0]
        x0, y0 = p.get_x(), p.get_y()
        x1 = x0 + p.get_width()
        y1 = y0 + p.get_height()
        line_upper = mpl.lines.Line2D([x0, x1], [y1, y1], color='k')
        return [p, line_upper]



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


def tick_label_writer2(nlist):
    labellist = []
    for n in nlist:
        if n < 0:
            if n == -1:
                labellist += [r'$f_\mathrm{{LISA}}$']
            else:
                labellist += [r'{0}$f_\mathrm{{LISA}}$'.format(n)]
        if n == 0:
            labellist += [r'$f_\mathrm{GW,0}$']
        if n > 0:
            if n == 1:
                labellist += [r'+$f_\mathrm{{LISA}}$']
            else:
                labellist += [r'+{0}$f_\mathrm{{LISA}}$'.format(n)]
    return labellist



def S_OMS(freqs):
    # Aoms = 7.9e-12  # <-- Sangria
    # Aoms = 15e-12
    Aoms = 12e-12
    return (Aoms * 2*np.pi*freqs/const.c.value)**2 * (1 + (2e-3/freqs)**4)

def S_acc(freqs):
    # Aacc = 2.4e-15  # <-- Sangria
    # Aacc = 3e-15
    Aacc = 2.4e-15
    return (Aacc / (2*np.pi*freqs*const.c.value))**2 * (1 + (0.4e-3/freqs)**2) * (1 + (freqs/8e-3)**4)

def S_X20_EQarm(freqs, L):
    omega = 2*np.pi*freqs
    Soms = S_OMS(freqs)
    Sacc = S_acc(freqs)
    return 64 * np.sin(2*omega*L/const.c.value)**2 * np.sin(omega*L/const.c.value)**2 * ( Soms + (3+np.cos(2*omega*L/const.c.value))*Sacc )



# Frequency range for the LISA sensitivity curve
freqs_sens = np.logspace(-4,-1,200)


Larm = 2.5e9








plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # <--- default: dejavusans
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12


fig, ax = plt.subplot_mosaic([['l','tr'],['l','mr'],['l','br']], width_ratios=[1/4,1], figsize=(20,14))
plt.subplots_adjust(wspace=0.1)

ax['l'].set_ylabel(r'TDI PSD $\left[\mathrm{Hz}^{-1}\right]$', fontsize=16)
ax['l'].set_xlabel(r'$f \ \left[\mathrm{Hz}\right]$',          fontsize=16), ax['br'].set_xlabel(r'$f \ \left[\mathrm{mHz}\right]$', fontsize=16)


ax['l'].set_title('Full Sensitivity Space', y=1.02, fontsize=20), ax['tr'].set_title('Zoom In', fontsize=20)



ax['tr'].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax['mr'].xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax['br'].xaxis.set_major_formatter(FormatStrFormatter('%.5f'))








# Set the normalization convention
norm = "backward"


Tlisa  = 1*u.year.to(u.s)
f_lisa = 1/Tlisa






# -------------------------------------------------------
# LISA #1
# -------------------------------------------------------
# Load the data and look at the keys
hf = h5py.File('./LISAsim_data_files/lisasim1.h5', 'r')

print("H5 file keys: ", hf.keys())

# Store the data in arrays, then close the h5 file
data_sim  = dict(zip( [i for i in hf['simulation_metadata'].keys()], [hf['simulation_metadata/'+i][()] for i in hf['simulation_metadata'].keys()] ))
data_X_mono  = hf['monochromatic'][:]
data_X_chirp = hf['chirping'][:]
data_t       = hf['times'][:]
hf.close()

print("Binary System initial chirp rate = {0:0.2f} nHz/year".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value))


# Compute the one-dimensional discrete Fourier Transform for real input (output is still complex!)
# (and cut first f=0 frequency from the dataset)
X_mono_DFT  = np.fft.rfft(data_X_mono, norm=norm)[1:]  
X_chirp_DFT = np.fft.rfft(data_X_chirp, norm=norm)[1:]  

# Return the Discrete Fourier Transform sample frequencies and their spacing
# (and cut first f=0 frequency from the dataset)
freqs = np.fft.rfftfreq(data_t.size, data_sim['dt'])[1:]
freqs *= u.Hz.to(u.mHz)


# -------------- PSDs ----------------
PSD_mono  = PSD_1s(X_mono_DFT, data_sim['dt'], data_t.size, norm).real
PSD_chirp = PSD_1s(X_chirp_DFT, data_sim['dt'], data_t.size, norm).real




# ------------ LEFT PLOT --------------
ax['l'].fill_between(freqs_sens, S_X20_EQarm(freqs_sens, Larm), 0, color='gray', alpha=0.2, label='LISA Noise')
ax['l'].loglog(freqs*u.mHz.to(u.Hz), PSD_mono,        color='#4b026c', label='Monochromatic')
ax['l'].loglog(freqs*u.mHz.to(u.Hz), PSD_chirp, '--', color='#ffa700', label='Chirping')
ax['l'].loglog(freqs_sens, S_X20_EQarm(freqs_sens, Larm), color='k')
ax['l'].set_xlim([freqs_sens.min(), freqs_sens.max()])
ax['l'].set_ylim([2e-45, 2e-37])
ax['l'].legend(loc='upper left', fontsize=14, handler_map={mpl.collections.PolyCollection: HandlerFilledBetween()})



# ------------ RIGHT PLOT --------------
ax['tr'].fill_between(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), 0, color='gray', alpha=0.2)
ax['tr'].semilogy(freqs, PSD_mono,        color='#4b026c')
ax['tr'].semilogy(freqs, PSD_chirp, '--', color='#ffa700')
ax['tr'].semilogy(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), color='k')


# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(data_sim['Mp']*u.kg.to(const.M_jup))+"\n"+r"$T_p$ = {0:0.2f} years".format(data_sim['T_p']*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(data_sim['epsilon'])#+"\n"+r"$\dot{{f}}_\mathrm{{GW,0}}$ = {0:0.2f} $\ \frac{{\mathrm{{nHz}}}}{{\mathrm{{year}}}}$".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value)
ax['tr'].text(0.875, 0.74, simulation_info, fontsize=12, transform=ax['tr'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

n = 3
ax['tr'].set_xlim([10**(np.log10(data_sim['fgw0'] - n*data_sim['f_p']))*u.Hz.to(u.mHz), 10**(np.log10(data_sim['fgw0'] + n*data_sim['f_p']))*u.Hz.to(u.mHz)])
ax['tr'].set_ylim([1e-50, 1e-37])

# Top x-axis
ax2 = ax['tr'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-2,-1,0,1,2]
ax_tick_list       = [(data_sim['fgw0'] + n*data_sim['f_p'])*u.Hz.to(u.mHz) for n in n_ticks]
x_tick_list_labels = tick_label_writer(n_ticks)
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-2,-1,0,1,2]
ax_tick_list = [(data_sim['fgw0'] + n*data_sim['f_p'])*u.Hz.to(u.mHz) for n in n_ticks]
ax['tr'].set_xticks(ax_tick_list), ax['tr'].minorticks_off()
ax['tr'].grid(axis='x')






# -------------------------------------------------------
# LISA #2
# -------------------------------------------------------
# Load the data and look at the keys
hf = h5py.File('./LISAsim_data_files/lisasim2.h5', 'r')

print("H5 file keys: ", hf.keys())

# Store the data in arrays, then close the h5 file
data_sim  = dict(zip( [i for i in hf['simulation_metadata'].keys()], [hf['simulation_metadata/'+i][()] for i in hf['simulation_metadata'].keys()] ))
data_X_mono  = hf['monochromatic'][:]
data_X_chirp = hf['chirping'][:]
data_t       = hf['times'][:]
hf.close()

print("Binary System initial chirp rate = {0:0.2f} nHz/year".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value))


# Compute the one-dimensional discrete Fourier Transform for real input (output is still complex!)
# (and cut first f=0 frequency from the dataset)
X_mono_DFT  = np.fft.rfft(data_X_mono, norm=norm)[1:]  
X_chirp_DFT = np.fft.rfft(data_X_chirp, norm=norm)[1:]  

# Return the Discrete Fourier Transform sample frequencies and their spacing
# (and cut first f=0 frequency from the dataset)
freqs = np.fft.rfftfreq(data_t.size, data_sim['dt'])[1:]
freqs *= u.Hz.to(u.mHz)


# -------------- PSDs ----------------
PSD_mono  = PSD_1s(X_mono_DFT, data_sim['dt'], data_t.size, norm).real
PSD_chirp = PSD_1s(X_chirp_DFT, data_sim['dt'], data_t.size, norm).real




# # ------------ LEFT PLOT --------------
# ax['l'].loglog(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), color='k', label='LISA')
# ax['l'].loglog(freqs, PSD_mono,        label='Monochromatic')
# ax['l'].loglog(freqs, PSD_chirp, '--', label='Chirping')
# ax['l'].set_xlim([freqs_sens.min()*u.Hz.to(u.mHz), freqs_sens.max()*u.Hz.to(u.mHz)])
# ax['l'].set_ylim([1e-45, 1e-36])
# ax['l'].legend(loc='upper left')



# ------------ RIGHT PLOT --------------
ax['mr'].fill_between(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), 0, color='gray', alpha=0.2)
ax['mr'].semilogy(freqs, PSD_mono,        color='#4b026c')
ax['mr'].semilogy(freqs, PSD_chirp, '--', color='#ffa700')
ax['mr'].semilogy(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), color='k')

# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(data_sim['Mp']*u.kg.to(const.M_jup))+"\n"+r"$T_p$ = {0:0.1f} years".format(data_sim['T_p']*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(data_sim['epsilon'])#+"\n"+r"$\dot{{f}}_\mathrm{{GW,0}}$ = {0:0.2f} $\ \frac{{\mathrm{{nHz}}}}{{\mathrm{{year}}}}$".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value)
ax['mr'].text(0.885, 0.74, simulation_info, fontsize=12, transform=ax['mr'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

#n = 30
ax['mr'].set_xlim([10**(np.log10(data_sim['fgw0'] - 30*f_lisa))*u.Hz.to(u.mHz), 10**(np.log10(data_sim['fgw0'] + 35*f_lisa))*u.Hz.to(u.mHz)])
ax['mr'].set_ylim([1e-48, 1e-37])

# Top x-axis
ax2 = ax['mr'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
# ax_tick_list       = [(data_sim['fgw0'] + n*data_sim['f_p'])*u.Hz.to(u.mHz) for n in n_ticks]
ax_tick_list       = [(data_sim['fgw0'] + n*f_lisa)*u.Hz.to(u.mHz) for n in n_ticks]
x_tick_list_labels = tick_label_writer2(n_ticks)
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-20,-10,0,10,20]
ax_tick_list = [(data_sim['fgw0'] + n*f_lisa)*u.Hz.to(u.mHz) for n in n_ticks]
ax['mr'].set_xticks(ax_tick_list), ax['mr'].minorticks_off()
ax['mr'].grid(axis='x')










# -------------------------------------------------------
# LISA #3
# -------------------------------------------------------
# Load the data and look at the keys
hf = h5py.File('./LISAsim_data_files/lisasim3.h5', 'r')

print("H5 file keys: ", hf.keys())

# Store the data in arrays, then close the h5 file
data_sim  = dict(zip( [i for i in hf['simulation_metadata'].keys()], [hf['simulation_metadata/'+i][()] for i in hf['simulation_metadata'].keys()] ))
data_X_mono  = hf['monochromatic'][:]
data_X_chirp = hf['chirping'][:]
data_t       = hf['times'][:]
hf.close()

print("Binary System initial chirp rate = {0:0.2f} nHz/year".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value))


# Compute the one-dimensional discrete Fourier Transform for real input (output is still complex!)
# (and cut first f=0 frequency from the dataset)
X_mono_DFT  = np.fft.rfft(data_X_mono, norm=norm)[1:]  
X_chirp_DFT = np.fft.rfft(data_X_chirp, norm=norm)[1:]  

# Return the Discrete Fourier Transform sample frequencies and their spacing
# (and cut first f=0 frequency from the dataset)
freqs = np.fft.rfftfreq(data_t.size, data_sim['dt'])[1:]
freqs *= u.Hz.to(u.mHz)


# -------------- PSDs ----------------
PSD_mono  = PSD_1s(X_mono_DFT, data_sim['dt'], data_t.size, norm).real
PSD_chirp = PSD_1s(X_chirp_DFT, data_sim['dt'], data_t.size, norm).real




# # ------------ LEFT PLOT --------------
# ax['l'].loglog(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), color='k', label='LISA')
# ax['l'].loglog(freqs, PSD_mono,        label='Monochromatic')
# ax['l'].loglog(freqs, PSD_chirp, '--', label='Chirping')
# ax['l'].set_xlim([freqs_sens.min()*u.Hz.to(u.mHz), freqs_sens.max()*u.Hz.to(u.mHz)])
# ax['l'].set_ylim([1e-45, 1e-36])
# ax['l'].legend(loc='upper left')



# ------------ RIGHT PLOT --------------
ax['br'].fill_between(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), 0, color='gray', alpha=0.2)
ax['br'].semilogy(freqs, PSD_mono,        color='#4b026c')
ax['br'].semilogy(freqs, PSD_chirp, '--', color='#ffa700')
ax['br'].semilogy(freqs_sens*u.Hz.to(u.mHz), S_X20_EQarm(freqs_sens, Larm), color='k')

# plot text box info
simulation_info = r"$M_p$ = {0:0.0f} $\mathrm{{M}}_\mathrm{{Jup}}$".format(data_sim['Mp']*u.kg.to(const.M_jup))+"\n"+r"$T_p$ = {0:0.1f} years".format(data_sim['T_p']*u.s.to(u.year))+"\n"+r"$\epsilon$ = {0:0.3f}".format(data_sim['epsilon'])#+"\n"+r"$\dot{{f}}_\mathrm{{GW,0}}$ = {0:0.2f} $\ \frac{{\mathrm{{nHz}}}}{{\mathrm{{year}}}}$".format((data_sim['dfgw0']*u.Hz**2).to(u.nHz/u.year).value)
ax['br'].text(0.885, 0.74, simulation_info, fontsize=12, transform=ax['br'].transAxes, bbox=dict(alpha=1,color='white',boxstyle='round',ec='k'));

#n = 30
ax['br'].set_xlim([10**(np.log10(data_sim['fgw0'] - 30*f_lisa))*u.Hz.to(u.mHz), 10**(np.log10(data_sim['fgw0'] + 35*f_lisa))*u.Hz.to(u.mHz)])
ax['br'].set_ylim([1e-46, 1e-37])

# Top x-axis
ax2 = ax['br'].secondary_xaxis('top')
ax2.tick_params(direction="in")

n_ticks = [-25,-20,-15,-10,-5,0,5,10,15,20,25]
# ax_tick_list       = [(data_sim['fgw0'] + n*data_sim['f_p'])*u.Hz.to(u.mHz) for n in n_ticks]
ax_tick_list       = [(data_sim['fgw0'] + n*f_lisa)*u.Hz.to(u.mHz) for n in n_ticks]
x_tick_list_labels = tick_label_writer2(n_ticks)
ax2.set_xticks(ax_tick_list), ax2.set_xticklabels(x_tick_list_labels)
ax2.minorticks_off()

n_ticks = [-20,-10,0,10,20]
ax_tick_list = [(data_sim['fgw0'] + n*f_lisa)*u.Hz.to(u.mHz) for n in n_ticks]
ax['br'].set_xticks(ax_tick_list), ax['br'].minorticks_off()
ax['br'].grid(axis='x')












#plt.savefig('./lisa_sim.pdf', bbox_inches='tight')

plt.show()


