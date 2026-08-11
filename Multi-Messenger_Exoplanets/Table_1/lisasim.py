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

File: lisasim.py

Purpose: Run the LISA simulations used to calculate the "Derived System Properties" in Table 1.


This script is for generating the LISA response in **fractional frequency deviation** TDI given just a specified orbit and GW response.  All LISA noise sources are therefore not simulated.

- First the GW response is generated given the desired input parameters, then the orbit + GW response are passed directly to PyTDI.
- Four simulations are run back-to-back: both the monochromatic and phase chirping models of the DWD system with and without the exoplanet.
    

Required:
- pre-made orbit file
- parameters for desired white dwarf binary and exoplanet

Output:
- (intermediate) four GW response files are created but then deleted after the PyTDI step is complete
- h5 file containing the metadata of the simulation + the TDI outputs for the simulations
"""

import numpy as np
import h5py
from astropy import constants as const, units as u
import os


# GWResponse
from lisagwresponse import ReadStrain
# PyTDI
from pytdi.michelson import X2
from pytdi import Data

import sys
sys.path.append('../')
from functions import *



# -------------------------------------------------------
#  Orbits 
# -------------------------------------------------------
#  Provide the orbit file
orbit_file = '../LISAsim_data_files/orbits_equalarm_20350101.h5'

f = h5py.File(orbit_file, 'r')
orbit_starttime = f.attrs['t0']


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


# -------------------------------------------------------
# Constant across all simulations 
M1   = 0.335*const.M_sun.value   # White dwarf mass 1
M2   = 0.323*const.M_sun.value   # White dwarf mass 2
Msys = M1 + M2                   # Binary system mass

fgw0 = 3.79e-3                   # Initial GW frequency

inc    = np.pi/2                 # Exoplanet inclination
R      = 150*u.pc.to(u.m)        # source distance
iota   = 0
psi    = 0
phi0   = 0
phi0_p = 0
t0     = starttime      # initial time (which corresponds to fgw0, phi0, and phi0_p)

# Sky-angles for the source: Ecliptic latitude (beta) and longitude (lambda)
gw_beta   = 53.96*u.deg.to(u.rad)
gw_lambda = 13.24*u.deg.to(u.rad)

norm = "backward"
# -------------------------------------------------------





# -------------------------------------------------------
# LISA 
# -------------------------------------------------------
Mp   = 10*const.M_jup.value    # Exoplanet mass
T_p  = 0.1*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the required GW waveform parameters: Chirp-mass, GW amplitude, Semi-amplitude, initial chirp rate
Mc, K    = Mchirp(M1, M2), K_semiamp(T_p, Msys, Mp, inc)
A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)
ep       = epsilon(Mp, inc, f_p, fgw0, Msys)

print(f"phase modulation index = {ep:0.4f}")


# Monochromatic Waveform - NO EXOPLANET
waveform_mono_noexo = GWexo(A,iota,psi,phi0,fgw0,0,0,f_p,phi0_p,t0)
waveform_mono_noexo_Hplus, waveform_mono_noexo_Hcross = waveform_mono_noexo.H_plus(times_sim), waveform_mono_noexo.H_cross(times_sim)
# Chirping Waveform - NO EXOPLANET
waveform_chirp_noexo = GWexo(A,iota,psi,phi0,fgw0,dfgw0,0,f_p,phi0_p,t0)
waveform_chirp_noexo_Hplus, waveform_chirp_noexo_Hcross = waveform_chirp_noexo.H_plus(times_sim), waveform_chirp_noexo.H_cross(times_sim)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
waveform_mono_Hplus, waveform_mono_Hcross = waveform_mono.H_plus(times_sim), waveform_mono.H_cross(times_sim)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)
waveform_chirp_Hplus, waveform_chirp_Hcross = waveform_chirp.H_plus(times_sim), waveform_chirp.H_cross(times_sim)



############################# Observer-Frame Waveforms #############################
# generate the GW Response
print('\nBeginning GW Response calculation (monochromatic - NO EXOPLANET)')
gwsource_mono_noexo = ReadStrain(times_sim, waveform_mono_noexo_Hplus, waveform_mono_noexo_Hcross,
                                 orbits  = orbit_file,
                                 gw_beta = gw_beta, gw_lambda = gw_lambda)
print('\nBeginning GW Response calculation (chirping - NO EXOXPLANET)')
gwsource_chirp_noexo = ReadStrain(times_sim, waveform_chirp_noexo_Hplus, waveform_chirp_noexo_Hcross,
                                  orbits  = orbit_file,
                                  gw_beta = gw_beta, gw_lambda = gw_lambda)

print('\nBeginning GW Response calculation (monochromatic)')
gwsource_mono = ReadStrain(times_sim, waveform_mono_Hplus, waveform_mono_Hcross,
                           orbits  = orbit_file,
                           gw_beta = gw_beta, gw_lambda = gw_lambda)
print('\nBeginning GW Response calculation (chirping)')
gwsource_chirp = ReadStrain(times_sim, waveform_chirp_Hplus, waveform_chirp_Hcross,
                            orbits  = orbit_file,
                            gw_beta = gw_beta, gw_lambda = gw_lambda)

gw_file_mono_noexo  = '../LISAsim_data_files/gw_file_mono_noexo.h5'
gw_file_chirp_noexo = '../LISAsim_data_files/gw_file_chirp_noexo.h5'

gw_file_mono  = '../LISAsim_data_files/gw_file_mono.h5'
gw_file_chirp = '../LISAsim_data_files/gw_file_chirp.h5'


# --> if the file exists from a previous run, first delete it:
if os.path.exists(gw_file_mono_noexo):
    os.remove(gw_file_mono_noexo)
if os.path.exists(gw_file_chirp_noexo):
    os.remove(gw_file_chirp_noexo)
    
if os.path.exists(gw_file_mono):
    os.remove(gw_file_mono)
if os.path.exists(gw_file_chirp):
    os.remove(gw_file_chirp)

# NOTE: simulation data will be shortened down to 'final' data AFTER the PyTDI step!
gwsource_mono_noexo.write(path = gw_file_mono_noexo,
                          t0   = starttime,
                          dt   = dt,
                          size = Nt)
gwsource_chirp_noexo.write(path = gw_file_chirp_noexo,
                           t0   = starttime,
                           dt   = dt,
                           size = Nt)

gwsource_mono.write(path = gw_file_mono,
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

# Location of PyTDI output file
tdi_file = '../LISAsim_data_files/lisasim_10yr.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(tdi_file):
    os.remove(tdi_file)


hf = h5py.File(tdi_file, 'w')  # <-- Use if initially creating the document
hf.create_dataset('times', data=times)
hf.close()


#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_mono_noexo, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (monochromatic - NO EXOPLANET)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

hf.create_dataset('monochromatic, no exoplanet', data=X_data)
hf.close()

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_chirp_noexo, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (chirping - NO EXOPLANET)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

hf.create_dataset('chirping, no exoplanet', data=X_data)
hf.close()


#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_mono, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (monochromatic)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

hf.create_dataset('monochromatic', data=X_data)
hf.close()

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_chirp, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (chirping)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

hf.create_dataset('chirping', data=X_data)
hf.close()



# Delete the four GW files, no longer needed
os.remove(gw_file_mono_noexo)
os.remove(gw_file_chirp_noexo)

os.remove(gw_file_mono)
os.remove(gw_file_chirp)


################################### Save Metadata ###################################
# Finally, save all important metadata to the h5 file
hf = h5py.File(tdi_file, 'a')    # <-- Use if appending to the document!
# --> primary binary parameters
hf.create_dataset("simulation_metadata/A",         data = A)
hf.create_dataset("simulation_metadata/fgw0",      data = fgw0)
hf.create_dataset("simulation_metadata/dfgw0",     data = dfgw0)
hf.create_dataset("simulation_metadata/iota",      data = iota)
hf.create_dataset("simulation_metadata/psi",       data = psi)
hf.create_dataset("simulation_metadata/phi0",      data = phi0)
hf.create_dataset("simulation_metadata/gw_beta",   data = gw_beta)
hf.create_dataset("simulation_metadata/gw_lambda", data = gw_lambda)
# --> primary exoplanet parameters
hf.create_dataset("simulation_metadata/K",      data = K)
hf.create_dataset("simulation_metadata/f_p",    data = f_p)
hf.create_dataset("simulation_metadata/phi0_p", data = phi0_p)
# --> secondary binary + exoplanet parameters
hf.create_dataset("simulation_metadata/M1",      data = M1)
hf.create_dataset("simulation_metadata/M2",      data = M2)
hf.create_dataset("simulation_metadata/Mp",      data = Mp)
hf.create_dataset("simulation_metadata/inc",     data = inc)
hf.create_dataset("simulation_metadata/R",       data = R)
hf.create_dataset("simulation_metadata/T_p",     data = T_p)
hf.create_dataset("simulation_metadata/Mc",      data = Mc)
hf.create_dataset("simulation_metadata/epsilon", data = waveform_mono.epsilon)
# --> timing parameters
hf.create_dataset("simulation_metadata/Tobs", data = Tobs)
hf.create_dataset("simulation_metadata/fs",   data = fs)
hf.create_dataset("simulation_metadata/dt",   data = dt)

hf.close()



