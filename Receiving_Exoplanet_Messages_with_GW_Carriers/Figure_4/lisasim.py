"""
This script is for generating the LISA response in **fractional frequency deviation** TDI given just a specified orbit and GW response.  All LISA noise sources are therefore not simulated.

- First the GW response is generated given the desired input parameters, then the orbit + GW response are passed directly to PyTDI.
- Two simulations are run back-to-back: (1) binary + exoplanet, (2) binary

Required:
- pre-made orbit file
- parameters for desired white dwarf binary and exoplanet

Output:
- (intermediate) two GW response files are created but then deleted after the PyTDI step is complete
- h5 file containing the metadata of the simulation + the TDI outputs for both simulations (1) and (2)

Run:
>>> python3 script_lisasim.py

"""
import numpy as np
import h5py
from astropy import constants as const, units as u
import os


# Orbits
from datetime import datetime
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
orbit_file = './LISAsim_data_files/orbits_equalarm_20350101.h5'

f = h5py.File(orbit_file, 'r')
orbit_starttime = f.attrs['t0']


# -------------------------------------------------------
#  Timing / Observation 
# -------------------------------------------------------
fs = 1.1e-2
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
M1   = 0.6*const.M_sun.value   # White dwarf mass 1
M2   = 0.6*const.M_sun.value   # White dwarf mass 2
Msys = M1 + M2                 # Binary system mass

fgw0 = 5e-3                    # Initial GW frequency

inc    = np.pi/2               # Exoplanet inclination
R      = 2*u.kpc.to(u.m)       # source distance
iota   = 0
psi    = 0
phi0   = 0
phi0_p = 0
t0     = starttime      # initial time (which corresponds to fgw0, phi0, and phi0_p)

# Sky-angles for the source
gw_beta   = 0
gw_lambda = 0

norm = "backward"
# -------------------------------------------------------







# -------------------------------------------------------
# LISA #1
# -------------------------------------------------------
Mp   = 100*const.M_jup.value    # Exoplanet mass
T_p  = 0.01*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the required GW waveform parameters: Chirp-mass, GW amplitude, Semi-amplitude, initial chirp rate
Mc, K    = Mchirp(M1, M2), K_semiamp(T_p, Msys, Mp, inc)
A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
waveform_mono_Hplus, waveform_mono_Hcross = waveform_mono.H_plus(times_sim), waveform_mono.H_cross(times_sim)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)
waveform_chirp_Hplus, waveform_chirp_Hcross = waveform_chirp.H_plus(times_sim), waveform_chirp.H_cross(times_sim)



############################# Observer-Frame Waveforms #############################
# generate the GW Response
print('\nBeginning GW Response calculation (monochromatic)')
gwsource_mono = ReadStrain(times_sim, waveform_mono_Hplus, waveform_mono_Hcross,
                           orbits  = orbit_file,
                           gw_beta = gw_beta, gw_lambda = gw_lambda)
print('\nBeginning GW Response calculation (chirping)')
gwsource_chirp = ReadStrain(times_sim, waveform_chirp_Hplus, waveform_chirp_Hcross,
                            orbits  = orbit_file,
                            gw_beta = gw_beta, gw_lambda = gw_lambda)

gw_file_mono  = './LISAsim_data_files/gw_file_mono.h5'
gw_file_chirp = './LISAsim_data_files/gw_file_chirp.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(gw_file_mono):
    os.remove(gw_file_mono)
if os.path.exists(gw_file_chirp):
    os.remove(gw_file_chirp)

# NOTE: simulation data will be shortened down to 'final' data AFTER the PyTDI step!
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
#                    in **fractional frequency fluctuations**, so we don't
#                    divide by the central frequency (unlike .from_instrument())

# Location of PyTDI output file
tdi_file = './LISAsim_data_files/lisasim1.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(tdi_file):
    os.remove(tdi_file)


hf = h5py.File(tdi_file, 'w')  # <-- Use if initially creating the document
hf.create_dataset('times', data=times)
hf.close()

# Use the desired timing array to "mask out" the simulation data to only the 
# portion of the data that we want.
# --> This is important for removing the numerical artifacts that filters in 
#     PyTDI will introduce at the start of the data.
#mask = np.isin(times_sim, times)

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_mono, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (monochromatic)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('monochromatic', data=X_data[mask])
hf.create_dataset('monochromatic', data=X_data)
hf.close()

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_chirp, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (chirping)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('chirping', data=X_data[mask])
hf.create_dataset('chirping', data=X_data)
hf.close()


# Delete the two GW files, no longer needed
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




# -------------------------------------------------------
# LISA #2
# -------------------------------------------------------
Mp   = 30*const.M_jup.value    # Exoplanet mass
T_p  = 0.8*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the required GW waveform parameters: Chirp-mass, GW amplitude, Semi-amplitude, initial chirp rate
Mc, K    = Mchirp(M1, M2), K_semiamp(T_p, Msys, Mp, inc)
A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
waveform_mono_Hplus, waveform_mono_Hcross = waveform_mono.H_plus(times_sim), waveform_mono.H_cross(times_sim)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)
waveform_chirp_Hplus, waveform_chirp_Hcross = waveform_chirp.H_plus(times_sim), waveform_chirp.H_cross(times_sim)



############################# Observer-Frame Waveforms #############################
# generate the GW Response
print('\nBeginning GW Response calculation (monochromatic)')
gwsource_mono = ReadStrain(times_sim, waveform_mono_Hplus, waveform_mono_Hcross,
                           orbits  = orbit_file,
                           gw_beta = gw_beta, gw_lambda = gw_lambda)
print('\nBeginning GW Response calculation (chirping)')
gwsource_chirp = ReadStrain(times_sim, waveform_chirp_Hplus, waveform_chirp_Hcross,
                            orbits  = orbit_file,
                            gw_beta = gw_beta, gw_lambda = gw_lambda)

gw_file_mono  = './LISAsim_data_files/gw_file_mono.h5'
gw_file_chirp = './LISAsim_data_files/gw_file_chirp.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(gw_file_mono):
    os.remove(gw_file_mono)
if os.path.exists(gw_file_chirp):
    os.remove(gw_file_chirp)

# NOTE: simulation data will be shortened down to 'final' data AFTER the PyTDI step!
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
#                    in **fractional frequency fluctuations**, so we don't
#                    divide by the central frequency (unlike .from_instrument())

# Location of PyTDI output file
tdi_file = './LISAsim_data_files/lisasim2.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(tdi_file):
    os.remove(tdi_file)


hf = h5py.File(tdi_file, 'w')  # <-- Use if initially creating the document
hf.create_dataset('times', data=times)
hf.close()

# Use the desired timing array to "mask out" the simulation data to only the 
# portion of the data that we want.
# --> This is important for removing the numerical artifacts that filters in 
#     PyTDI will introduce at the start of the data.
#mask = np.isin(times_sim, times)

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_mono, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (monochromatic)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('monochromatic', data=X_data[mask])
hf.create_dataset('monochromatic', data=X_data)
hf.close()

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_chirp, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (chirping)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('chirping', data=X_data[mask])
hf.create_dataset('chirping', data=X_data)
hf.close()


# Delete the two GW files, no longer needed
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




# -------------------------------------------------------
# LISA #3
# -------------------------------------------------------
Mp   = 8*const.M_jup.value    # Exoplanet mass
T_p  = 2.3*u.year.to(u.s)      # Exoplanet period
# -------------------------------------------------------
# Exoplanet frequency
f_p = 1/T_p

# Compute the required GW waveform parameters: Chirp-mass, GW amplitude, Semi-amplitude, initial chirp rate
Mc, K    = Mchirp(M1, M2), K_semiamp(T_p, Msys, Mp, inc)
A, dfgw0 = Amp_binary(Mc, R, fgw0), dfgw_dt_binary(Mc, fgw0, 0, t0)

# Monochromatic Waveform
waveform_mono = GWexo(A,iota,psi,phi0,fgw0,0,K,f_p,phi0_p,t0)
waveform_mono_Hplus, waveform_mono_Hcross = waveform_mono.H_plus(times_sim), waveform_mono.H_cross(times_sim)
# Chirping Waveform
waveform_chirp = GWexo(A,iota,psi,phi0,fgw0,dfgw0,K,f_p,phi0_p,t0)
waveform_chirp_Hplus, waveform_chirp_Hcross = waveform_chirp.H_plus(times_sim), waveform_chirp.H_cross(times_sim)



############################# Observer-Frame Waveforms #############################
# generate the GW Response
print('\nBeginning GW Response calculation (monochromatic)')
gwsource_mono = ReadStrain(times_sim, waveform_mono_Hplus, waveform_mono_Hcross,
                           orbits  = orbit_file,
                           gw_beta = gw_beta, gw_lambda = gw_lambda)
print('\nBeginning GW Response calculation (chirping)')
gwsource_chirp = ReadStrain(times_sim, waveform_chirp_Hplus, waveform_chirp_Hcross,
                            orbits  = orbit_file,
                            gw_beta = gw_beta, gw_lambda = gw_lambda)

gw_file_mono  = './LISAsim_data_files/gw_file_mono.h5'
gw_file_chirp = './LISAsim_data_files/gw_file_chirp.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(gw_file_mono):
    os.remove(gw_file_mono)
if os.path.exists(gw_file_chirp):
    os.remove(gw_file_chirp)

# NOTE: simulation data will be shortened down to 'final' data AFTER the PyTDI step!
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
#                    in **fractional frequency fluctuations**, so we don't
#                    divide by the central frequency (unlike .from_instrument())

# Location of PyTDI output file
tdi_file = './LISAsim_data_files/lisasim3.h5'

# --> if the file exists from a previous run, first delete it:
if os.path.exists(tdi_file):
    os.remove(tdi_file)


hf = h5py.File(tdi_file, 'w')  # <-- Use if initially creating the document
hf.create_dataset('times', data=times)
hf.close()

# Use the desired timing array to "mask out" the simulation data to only the 
# portion of the data that we want.
# --> This is important for removing the numerical artifacts that filters in 
#     PyTDI will introduce at the start of the data.
#mask = np.isin(times_sim, times)

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_mono, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (monochromatic)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('monochromatic', data=X_data[mask])
hf.create_dataset('monochromatic', data=X_data)
hf.close()

#------------------------------------------------------------------------------------------
data = Data.from_gws(gw_file_chirp, orbit_file, gw_dataset='tcb', orbit_dataset='tcb/ltt')

print('\nBeginning TDI-X calculation (chirping)')
TDI_X  = X2.build(**data.args)
X_data = TDI_X(data.measurements)

hf = h5py.File(tdi_file, 'a')  # <-- Use if appending to the document!

# hf.create_dataset('chirping', data=X_data[mask])
hf.create_dataset('chirping', data=X_data)
hf.close()


# Delete the two GW files, no longer needed
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




