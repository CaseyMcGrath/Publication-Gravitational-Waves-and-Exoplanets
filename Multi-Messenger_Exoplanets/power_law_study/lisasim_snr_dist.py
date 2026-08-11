"""
   Copyright 2026 Casey McGrath

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

File: lisasim_snr_dist.py

Purpose: Run the LISA simulations used to test the SNR residual power law dependence on the GW source distance.


This script is for generating the LISA response in **fractional frequency deviation** TDI given just a specified orbit and GW response.  All LISA noise sources are therefore not simulated.

Required:
- pre-made orbit file
- parameters for desired white dwarf binary and exoplanet

Output:
- text file containing the exoplanet's distance and the calculated SNR residual 
- saved to the 'LISAsim_data_files' folder
"""

import numpy as np
import h5py
from astropy import constants as const, units as u
import os

import time


# GWResponse
from lisagwresponse import ReadStrain
# PyTDI
from pytdi.michelson import X2
from pytdi import Data

import sys
sys.path.append('../')
from functions import *





# -------------------------------------------------------
# LISA Sensitivity (Michelson TDI X 2.0) functions
# -------------------------------------------------------
def S_OMS(freqs):
    Aoms = 12e-12
    return (Aoms * 2*np.pi*freqs/const.c.value)**2 * (1 + (2e-3/freqs)**4)

def S_acc(freqs):
    Aacc = 2.4e-15
    return (Aacc / (2*np.pi*freqs*const.c.value))**2 * (1 + (0.4e-3/freqs)**2) * (1 + (freqs/8e-3)**4)

def S_X20_EQarm(freqs, L):
    omega = 2*np.pi*freqs
    Soms = S_OMS(freqs)
    Sacc = S_acc(freqs)
    return 64 * np.sin(2*omega*L/const.c.value)**2 * np.sin(omega*L/const.c.value)**2 * ( Soms + (3+np.cos(2*omega*L/const.c.value))*Sacc )




# -------------------------------------------------------
#  Orbits 
# -------------------------------------------------------
#  Provide the orbit file
orbit_file = '../LISAsim_data_files/orbits_equalarm_20350101.h5'

f = h5py.File(orbit_file, 'r')
orbit_starttime = f.attrs['t0']

# LISA arm-length
Larm = 2.5e9  # units: [m]


# -------------------------------------------------------
#  Timing / Observation 
# -------------------------------------------------------
fs = 0.02
dt = 1/fs

start_sim = orbit_starttime
pad_sim   = int(10)  # the number of "dt"s before and after our simulation

starttime = start_sim + pad_sim*dt
Tobs      = 10*u.year.to(u.s)
endtime   = starttime + Tobs
times     = np.arange(starttime, endtime, dt)
Nt        = len(times)

end_sim   = endtime + pad_sim*dt
times_sim = np.arange(start_sim, end_sim, dt)

# frequency resolution
df = 1/Tobs


# -------------------------------------------------------
# Constant across all simulations 
M1   = 0.335*const.M_sun.value   # White dwarf mass 1
M2   = 0.323*const.M_sun.value   # White dwarf mass 2
Msys = M1 + M2                   # Binary system mass

fgw0 = 3.79e-3                   # Initial GW frequency

inc    = np.pi/2                 # Exoplanet inclination
iota   = 0
psi    = 0
phi0   = 0
phi0_p = 0
t0     = starttime      # initial time (which corresponds to fgw0, phi0, and phi0_p)

T_p = 0.1*u.year.to(u.s)       # Exoplanet period
Mp  = 7*const.M_jup.value      # Exoplanet mass

# Sky-angles for the source
gw_beta   = 53.96*u.deg.to(u.rad)
gw_lambda = 13.24*u.deg.to(u.rad)

norm = "backward"
# -------------------------------------------------------



# -------------------------------------------------------
# Exoplanet discovery space: GW source distance range
Rvals = np.logspace(np.log10(50),np.log10(1500),10)*u.pc.to(u.m)      # units: [m]
# -------------------------------------------------------





if os.path.exists('../LISAsim_data_files/snrdiff_dist.txt'):
    os.remove('../LISAsim_data_files/snrdiff_dist.txt')




# ---------------------------------
#  SIMULATION
# ---------------------------------

loop_start_time = time.time()

for R in Rvals:

    # Exoplanet frequency
    f_p = 1/T_p
    
    # Compute the required GW waveform parameters: Chirp-mass, GW amplitude, Semi-amplitude, initial chirp rate
    Mc, K    = Mchirp(M1, M2), K_semiamp(T_p, Msys, Mp, inc)
    A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)
    
    # Chirping Waveform - NO EXOPLANET
    waveform_chirp_noexo = GWexo(A,iota,psi,phi0,fgw0,dfgw0,0,f_p,phi0_p,t0)
    waveform_chirp_noexo_Hplus, waveform_chirp_noexo_Hcross = waveform_chirp_noexo.H_plus(times_sim), waveform_chirp_noexo.H_cross(times_sim)
    
    # Chirping Waveform
    waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)
    waveform_chirp_Hplus, waveform_chirp_Hcross = waveform_chirp.H_plus(times_sim), waveform_chirp.H_cross(times_sim)
    
    
    ############################# Observer-Frame Waveforms #############################
    # generate the GW Response
    gwsource_chirp_noexo = ReadStrain(times_sim, waveform_chirp_noexo_Hplus, waveform_chirp_noexo_Hcross,
                                      orbits  = orbit_file,
                                      gw_beta = gw_beta, gw_lambda = gw_lambda)
    
    gwsource_chirp = ReadStrain(times_sim, waveform_chirp_Hplus, waveform_chirp_Hcross,
                                orbits  = orbit_file,
                                gw_beta = gw_beta, gw_lambda = gw_lambda)
    
    gw_file_chirp_noexo = '../LISAsim_data_files/gw_file_chirp_noexo.h5'
    gw_file_chirp       = '../LISAsim_data_files/gw_file_chirp.h5'
    
    
    # --> if the file exists from a previous run, first delete it:
    if os.path.exists(gw_file_chirp_noexo):
        os.remove(gw_file_chirp_noexo)
        
    if os.path.exists(gw_file_chirp):
        os.remove(gw_file_chirp)
    
    
    gwsource_chirp_noexo.write(path = gw_file_chirp_noexo,
                               t0   = starttime,
                               dt   = dt,
                               size = Nt)
    
    gwsource_chirp.write(path = gw_file_chirp,
                         t0   = starttime,
                         dt   = dt,
                         size = Nt)
    
    
    ################################### PyTDI ###################################
    # Directly pass the orbit and GW Response to PyTDI.
    
    # --> CRITICAL NOTE: remember, when PyTDI outputs .from_gws(), the output is 
    #                    in **fractional frequency fluctuations**
    
    #------------------------------------------------------------------------------------------
    data = Data.from_gws(gw_file_chirp_noexo, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')
    
    TDI_X  = X2.build(**data.args)
    X_data_noexo = TDI_X(data.measurements)
    
    
    #------------------------------------------------------------------------------------------
    data = Data.from_gws(gw_file_chirp, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')
    
    TDI_X  = X2.build(**data.args)
    X_data = TDI_X(data.measurements)
    
    
    #------------------------------------------------------------------------------------------
    # Delete the four GW files, no longer needed
    os.remove(gw_file_chirp_noexo)
    os.remove(gw_file_chirp)
    
    
    
    #------------------------------------------------------------------------------------------
    # Compute the one-dimensional discrete Fourier Transform for real input (output is still complex!)
    # (and cut first f=0 frequency from the dataset)
    # chirp_DFT       = np.fft.rfft(X_data, norm=norm)[1:]  
    # chirp_noexo_DFT = np.fft.rfft(X_data_noexo, norm=norm)[1:]
    
    X_data_diff    = X_data - X_data_noexo
    chirp_diff_DFT = np.fft.rfft(X_data_diff, norm=norm)[1:]
    
    # Return the Discrete Fourier Transform sample frequencies and their spacing
    # (and cut first f=0 frequency from the dataset)
    freqs = np.fft.rfftfreq(Nt, dt)[1:]
    
    
    
    # -------------- PSDs ----------------
    PSD_chirp_diff = PSD_1s(chirp_diff_DFT, dt, Nt, norm).real
    
    
    # -------------- SNR residual ----------------
    SNRdiff = np.sqrt( 2 * np.sum(PSD_chirp_diff / S_X20_EQarm(freqs, Larm)) )
    
    # -------------------------------------
    # Save the output   
    f = open('../LISAsim_data_files/snrdiff_dist.txt', 'a')
    np.savetxt(f, np.array([[R],[SNRdiff]]).T)
    f.close()



loop_end_time = time.time()
loop_elapsed_time = (loop_end_time - loop_start_time)*u.s.to(u.min)
print(f"\nRuntime: {loop_elapsed_time:.4f} minutes")







