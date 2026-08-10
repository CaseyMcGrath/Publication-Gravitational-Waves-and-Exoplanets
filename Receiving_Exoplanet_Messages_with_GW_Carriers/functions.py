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

File: functions.py

Purpose: Functions for modeling the Doppler Modulation of a gravitational wave source. 
"""

import numpy as np
import scipy.special as sp
from astropy import constants as const




class GWexo:
    """
    Class to construct and calculate the GW+exoplanet strain waveform.

    Attributes
    ----------
    A : float
        GW Amplitude (units: [-]).
    iota : float
        GW Inclination angle (units: [rad]).
    psi : float
        GW Polarization angle (units: [rad]).
    phi0 : float
        Initial GW phase (units: [rad]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).
    dfgw0 : float
        Initial GW frequency derivative (units: [Hz^2]).
    K : float
        Exoplanet Semi-amplitude (units: [m/s]).
    f_p : float
        Exoplanet frequency (units: [Hz]).
    phi_p : float
        Exoplanet initial phase (units: [rad]).
    t0 : float
        Initial time (units: [s]).
    chirp_doppler : bool, optional
        Flag for including the chirp Doppler effect in the calculation, by default False.
    omega0 : float
        Initial GW (orbital) angular frequency (units: [rad/s]).
    domega0 : float
        Initial GW (orbital) angular frequency derivative (units: [rad/s^2]).
    omega_p : float
        Exoplanet orbital frequency (units: [rad/s]).
    epsilon : float
        Modulation index (units: [rad]).
    epsilon_chirp : float
        Chirp modulation index (units: [rad]).
    Hp_A1 : float
        Plus H strain amplitude 1 term (units: [-]).
    Hp_A2 : float
        Plus H strain amplitude 2 term (units: [-]).
    Hx_A1 : float
        Cross H strain amplitude 1 term (units: [-]).
    Hx_A2 : float
        Cross H strain amplitude 2 term (units: [-]).

    Methods
    -------
    V_r(t):
        Radial velocity function.
    omega(t):
        GW (orbital) angular frequency function.
    Phi_gw(t):
        GW phase function.
    h_plus(t):
        GW plus h strain function.
    h_cross(t):
        GW cross h strain function.
    H_plus(t):
        GW plus H strain function.
    H_cross(t):
        GW cross H strain function.
    """
    
    def __init__(self, A, iota, psi, phi0, fgw0, dfgw0, K, f_p, phi_p, t0, chirp_doppler=False):
        """Constructs the necessary attributes (including ones derived from the input parameters) for the GWexo object.

        Parameters
        ----------
        A : float
            GW Amplitude (units: [-]).
        iota : float
            GW Inclination angle (units: [rad]).
        psi : float
            GW Polarization angle (units: [rad]).
        phi0 : float
            Initial GW phase (units: [rad]).
        fgw0 : float
            Initial GW frequency (units: [Hz]).
        dfgw0 : float
            Initial GW frequency derivative (units: [Hz^2]).
        K : float
            Exoplanet Semi-amplitude (units: [m/s]).
        f_p : float
            Exoplanet frequency (units: [Hz]).
        phi_p : float
            Exoplanet initial phase (units: [rad]).
        t0 : float
            Initial time (units: [s]).
        chirp_doppler : bool, optional
            Flag for including the chirp Doppler effect in the calculation, by default False.
        """
        
        # GW binary parameters
        self.A     = A
        self.iota  = iota
        self.psi   = psi
        self.phi0  = phi0
        self.fgw0  = fgw0
        self.dfgw0 = dfgw0
        
        # Exoplanet parameters
        self.K     = K
        self.f_p   = f_p
        self.phi_p = phi_p
        
        # Additional parameters
        self.t0 = t0
        self.chirp_doppler = chirp_doppler

        # Derived quantities
        self.omega0        = np.pi * fgw0
        self.domega0       = np.pi * dfgw0
        self.omega_p       = 2*np.pi * f_p
        self.epsilon       = K/const.c.value * (fgw0/f_p)
        self.epsilon_chirp = K/const.c.value * (dfgw0/(2*np.pi*f_p**2))

        self.Hp_A1 = -2*A * (1 + np.cos(iota)**2)/2 * np.cos(2*psi)
        self.Hp_A2 = -2*A * np.cos(iota) * np.sin(2*psi)
        self.Hx_A1 = -2*A * (1 + np.cos(iota)**2)/2 * np.sin(2*psi)
        self.Hx_A2 = -2*A * np.cos(iota) * np.cos(2*psi)

    
    def V_r(self, t):
        """Radial velocity function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            Radial velocity calculated along the time array (units: [m/s]).
        """
        return self.K * np.cos(self.phi_p + self.omega_p*(t-self.t0))

    def omega(self, t):
        """GW (orbital) angular frequency function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW (orbital) angular frequency calculated along the time array (units: [rad/s]).
        """
        return (1 - self.V_r(t)/const.c) * (self.omega0 + self.domega0*(t-self.t0))
    
    def Phi_gw(self, t):
        """GW phase function.

        Note: by default the chirp Doppler effect is excluded from the computation.  However, this can be enabled
        by setting the chirp_doppler=True class attribute.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW phase calculated along the time array (units: [rad]).
        """
        monochromatic_gw = self.phi0 + 2*self.omega0*(t-self.t0)
        
        chirp = self.domega0 * (t-self.t0)**2

        exoplanet_doppler_effect_monochromatic = self.epsilon * ( np.sin(self.phi_p + self.omega_p*(t-self.t0)) - np.sin(self.phi_p) )

        if self.chirp_doppler:
            exoplanet_doppler_effect_chirp = self.epsilon_chirp * ( self.omega_p * (t-self.t0) * np.sin(self.phi_p + self.omega_p*(t-self.t0))
                                                                   + np.cos(self.phi_p + self.omega_p*(t-self.t0))
                                                                   - np.cos(self.phi_p) 
                                                                  )
            return monochromatic_gw + chirp + exoplanet_doppler_effect_monochromatic + exoplanet_doppler_effect_chirp

        else:     
            return monochromatic_gw + chirp + exoplanet_doppler_effect_monochromatic
    
    def h_plus(self, t):
        """GW plus h strain function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW plus h strain calculated along the time array (units: [-]).
        """
        return -2 * self.A * (1 + np.cos(self.iota)**2)/2 * np.cos(self.Phi_gw(t))
        
    def h_cross(self, t):
        """GW cross h strain function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW cross h strain calculated along the time array (units: [-]).
        """
        return -2 * self.A * np.cos(self.iota) * np.sin(self.Phi_gw(t))

    def H_plus(self, t):
        """GW plus H strain function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW plus H strain calculated along the time array (units: [-]).
        """
        return self.h_plus(t)*np.cos(2*self.psi) - self.h_cross(t)*np.sin(2*self.psi) 
        
    def H_cross(self, t):
        """GW cross H strain function.

        Parameters
        ----------
        t : array
            Time array (units: [s]).

        Returns
        -------
        array
            GW cross H strain calculated along the time array (units: [-]).
        """
        return self.h_plus(t)*np.sin(2*self.psi) + self.h_cross(t)*np.cos(2*self.psi) 






# -------------------------------------------- DOPPLER-RELATED FUNCTIONS --------------------------------------------

def K_semiamp(T_p, M_sys, M_p, inc):
    """Exoplanet semi-amplitude parameter.

    Parameters
    ----------
    T_p : float
        Exoplanet's period (units: [s]).
    M_sys : float
        GW source system mass (units: [kg]).
    M_p : float
        Exoplanet's mass (units: [kg]).
    inc : float
        Inclination of the exoplanet's orbit to the line perpendicular to the line-of-sight (units: [rad]).

    Returns
    -------
    float
        Exoplanet semi-amplitude (units: [m/s]).
    """
    return (2*np.pi*const.G.value / T_p)**(1/3) * (M_p / (M_sys + M_p)**(2/3)) * np.sin(inc)


def epsilon(M_p, inc, f_p, fgw0, M_sys):
    """Exoplanet modulation index parameter.

    Parameters
    ----------
    M_p : float
        Exoplanet's mass (units: [kg]).
    inc : float
        Inclination of the exoplanet's orbit to the line perpendicular to the line-of-sight (units: [rad]).
    f_p : float
        Exoplanet's orbital frequency (units: [Hz]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).
    M_sys : float
        GW source system mass (units: [kg]).

    Returns
    -------
    float
        Exoplanet modulation index (units: [rad]).
    """
    return (2*np.pi*const.G.value)**(1/3) / const.c.value * (M_p*np.sin(inc))/(M_sys + M_p)**(2/3) * fgw0 / f_p**(2/3)





# -------------------------------------------- SOURCE AMPLITUDE FUNCTIONS --------------------------------------------

def Mchirp(M1, M2):
    """GW binary chirp mass.

    Parameters
    ----------
    M1 : float
        Binary object mass 1 (units: [kg])
    M2 : float
        Binary object mass 2 (units: [kg])

    Returns
    -------
    float
        Chirp mass (units: [kg]).
    """
    Mb = M1 + M2
    return (M1*M2)**(3/5) / Mb**(1/5)
    

def Amp_binary(Mchirp, R, fgw0):
    """GW Amplitude parameter for a binary system.

    Parameters
    ----------
    Mchirp : float
        Chirp mass (units: [kg]).
    R : float
        GW source distance (units: [m]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).

    Returns
    -------
    float
        Binary system GW amplitude parameter (units: [-]). 
    """
    return 2 * (const.G.value * Mchirp)**(5/3) / (const.c.value**4 * R) * (np.pi * fgw0)**(2/3)


def Amp_tri(ep, Iz, R, fgw0):
    """GW Amplitude parameter for a triaxial body.

    Parameters
    ----------
    ep : float
        Ellipticity (units: [-]).
    Iz : float
        Principle moment of inertia (units: [kg m^2]).
    R : float
        GW source distance (units: [m]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).

    Returns
    -------
    float
        Triaxial body GW amplitude parameter (units: [-]). 
    """
    return 2 * (const.G.value * ep * Iz) / (const.c.value**4 * R) * (np.pi * fgw0)**2





# -------------------------------------------- SOURCE FREQUENCY FUNCTIONS --------------------------------------------

def dTau_c(Mchirp, fgw0):
    """Time of coalescence (binary system).

    Parameters
    ----------
    Mchirp : float
        Chirp mass (units: [kg]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).

    Returns
    -------
    float
        Time of coalescence (units: [s]).
    """
    omega0 = np.pi * fgw0
    return 5/256 * (const.c.value**3 / (const.G.value * Mchirp))**(5/3) / omega0**(8/3)
    

def dTau(ep, Iz, fgw0):
    """Dynamical time parameter (triaxial body).

    Parameters
    ----------
    ep : float
        Ellipticity (units: [-]).
    Iz : float
        Principle moment of inertia (units: [kg m^2]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).

    Returns
    -------
    float
        Dynamical time parameter (units: [s]).
    """
    omega0 = np.pi * fgw0
    return 5/128 * (const.c.value**5 / (const.G.value * ep**2 * Iz)) / omega0**4


def dfgw_dt_binary(Mchirp, fgw0, t, t0):
    """Rate of change of the GW frequency for a binary system, driven purely by GW radiation.

    Parameters
    ----------
    Mchirp : float
        Chirp mass (units: [kg]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).
    t : array
        Time (units: [s]).
    t0 : float
        Initial time (units: [s]).

    Returns
    -------
    array
        Rate of change of the GW frequency [units: [Hz^2]].
    """
    dTc = dTau_c(Mchirp, fgw0)
    return 3/8 * (fgw0 / dTc) * (1 - (t-t0)/dTc)**(-11/8)


def dfgw_dt_tri(ep, Iz, fgw0, t, t0):
    """Rate of change of the GW frequency for a triaxial body, driven purely by GW radiation.

    Parameters
    ----------
    ep : float
        Ellipticity (units: [-]).
    Iz : float
        Principle moment of inertia (units: [kg m^2]).
    fgw0 : float
        Initial GW frequency (units: [Hz]).
    t : array
        Time (units: [s]).
    t0 : float
        Initial time (units: [s]).

    Returns
    -------
    array
        Rate of change of the GW frequency [units: [Hz^2]].
    """
    dT = dTau(ep, Iz, fgw0)
    return -1/4 * (fgw0 / dT) * (1 + (t-t0)/dT)**(-1/4)
    




# -------------------------------------------- UTILITY FUNCTIONS --------------------------------------------

def PSD_1s(x_DFT, dt, Nt, norm):
    """Calculate the discrete **1-SIDED** PSD of a times series.  The discrete definition approaches the analytic
    solution in the limit that the number of data points N --> infty (i.e., Delta t --> 0).

    Parameters
    ----------
    x_DFT : array
        1-sided DFT of a time series.
    dt : float
        Timing data time step.
    Nt : int
        Number of time data.
    norm : str
        The normalization convention that was used to generate the input DFT.
        Options: {“backward”, “ortho”, “forward”}

    Returns
    -------
    array
        The discrete 1-sided PSD of a time series.
    """
    if norm == 'backward':
        G, H = 1, 1/Nt
    if norm == 'forward':
        G, H = 1/Nt, 1
    if norm == 'ortho':
        G, H = 1/np.sqrt(Nt), 1/np.sqrt(Nt)

    return 2 * dt * H/G * x_DFT * np.conj(x_DFT)


def PSD_1s_fromFT(x_FT, Tobs):
    """Calculate the discrete **1-SIDED** PSD of a times series from an ANALYTIC expression.  The definition 
    approaches the analytic solution in the limit that the observation time --> infinity.

    Parameters
    ----------
    x_FT : array
        1-sided DFT of a time series.
    Tobs : float
        Observation time.

    Returns
    -------
    array
        The discrete 1-sided PSD of a time series.
    """
    return 2 / Tobs * x_FT * np.conj(x_FT)


def DFT_to_FT(x_DFT, freqs, dt, starttime, Nt, norm):
    """Convert from a Discrete Fourier Transform (DFT) to a Fourier Transform (FT).

    Parameters
    ----------
    x_DFT : array
        1-sided DFT of a time series.
    freqs : array
        Frequencies.
    dt : float
        Time step of time series.
    starttime : float
        Starting time of the time series.
    Nt : int
        Number of data in the time series.
    norm : str
        The normalization convention that was used to generate the input DFT.
        Options: {“backward”, “ortho”, “forward”}

    Returns
    -------
    array
        The 1-sided FT.
    """
    if norm == 'backward':
        G = 1
    if norm == 'forward':
        G = 1/Nt
    if norm == 'ortho':
        G = 1/np.sqrt(Nt)
    
    return np.exp(-1j*2*np.pi*freqs*starttime) * dt / G * x_DFT


def SNR(signal_FT_list, Sn_model, df):
    """Multi-data channel Signal-to-Noise Ratio, from a set of analytic FTs.

    Parameters
    ----------
    signal_FT_list : list/array-like
        1-sided set of FTs
        Shape: [number of FTs, number of frequencies]
    Sn_model : array
        1-sided CSD matrix model
        Shape: [number of FTs, number of FTs, number of frequencies]
    df : float
        Frequency bin width.

    Returns
    -------
    float
        The signal-to-noise ratio
    """
    # Broadcast inversion trick:  Sn^-1 = ((Sn^-1)^T)^T = ((Sn^T)^-1)^T
    # --> needed b/c np.linalg.inv() preserves the first dimension, and inverts over the 2nd two!
    # --> shape: [Nch, Nch, Nf]
    Sn_model_inv = np.linalg.inv(Sn_model.T).T
    
    # integrand channel dot product and frequecy summation: 
    # (1) dot product over all of the data channels first (for every frequency)
    # (2) sum up all of the frequencies
    # --> np.einsum('ijk,jk->ik', Sn_inv, s*) does the first channel dot product (collapses [Nch,Nch,Nf].[Nch,Nf] to [Nch,Nf])
    # --> np.einsum('ij,ij', s, ...) does the second channel dot product AND sums up all the frequencies (collapses [Nch,Nf].[Nch,Nf] to [Nf] 
    #     and sums them up!)
    integrand_sum = np.einsum('ij,ij', signal_FT_list, np.einsum('ijk,jk->ik', Sn_model_inv, np.conj(signal_FT_list)) )
    
    # --> sqrt( Re[ 4 * integrand_sum * df ] )
    return np.sqrt( np.real( 4 * integrand_sum * df) )





# -------------------------------------------- MODEL FUNCTIONS --------------------------------------------

# Monochromatic: FM (time domain)
def M_mono_FM(A, fc, fm, epsilon, phi0, phip, t0, times):
    """Monochromatic, frequency modulated model.

    Parameters
    ----------
    A : float
        Amplitude.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    times : array
        Times.

    Returns
    -------
    array
        Model calculated along the input time array.
    """
    wc = 2*np.pi*fc
    wm = 2*np.pi*fm
    return A*np.cos(phi0 + wc*(times-t0) + epsilon*np.sin(phip + wm*(times-t0)))


# Monochromatic: FM (frequency domain)
def M_mono_FM_FT(A, fc, fm, epsilon, phi0, phip, t0, freqs, Tobs, N):
    """Fourier Transform of the monochromatic, frequency modulated model.

    Parameters
    ----------
    A : float
        Amplitude.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    freqs : array
        Frequencies.
    Tobs : float
        Observation time.
    N : int
        The +/- N terms to be included in the summation calculation.

    Returns
    -------
    array
        Fourier transform calculated along the input frequencies array.
    """
    wc = 2*np.pi*fc
    wm = 2*np.pi*fm

    n = np.arange(-N,N+1,1)
    FT = []
    for fval in freqs:
        w = 2*np.pi*fval

        term1 = np.exp(1j*(phi0 + n*phip - w*t0 + (wc+n*wm-w)*Tobs/2)) * np.sin((wc+n*wm-w)*Tobs/2)/(wc+n*wm-w)
        term2 = np.exp(1j*(-phi0 + n*phip - w*t0 + (-wc+n*wm-w)*Tobs/2)) * np.sin((-wc+n*wm-w)*Tobs/2)/(-wc+n*wm-w)
        
        FT_f = A * sp.jv(n,epsilon) * ( term1 + ((-1.0)**n)*term2 )     
        FT_f = np.sum(FT_f)
        
        FT += [FT_f]
    return np.asarray(FT)



# Monochromatic: GW + Exoplanet (time domain)
def M_mono_GW(A1, A2, fc, fm, epsilon, phi0, phip, t0, times):
    """GW + exoplanet model: Monochromatic, frequency modulated

    Parameters
    ----------
    A1 : float
        Amplitude 1 term.
    A2 : float
        Amplitude 2 term.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    times : array
        Times.

    Returns
    -------
    array
        Model calculated along the input time array.
    """
    wc = 2*np.pi*fc
    wm = 2*np.pi*fm

    Phase = phi0 + wc*(times-t0) + epsilon*(np.sin(phip + wm*(times-t0)) - np.sin(phip))
    
    return A1*np.cos(Phase) - A2*np.sin(Phase)


# Monochromatic: GW + Exoplanet (frequency domain)
def M_mono_GW_FT(A1, A2, fc, fm, epsilon, phi0, phip, t0, freqs, Tobs, N):
    """Wrapper function to calculate the Fourier Transform for the GW + exoplanet monochromatic, frequency modulated model.

    Parameters
    ----------
    A1 : float
        Amplitude 1 term.
    A2 : float
        Amplitude 2 term.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    freqs : array
        Frequencies.
    Tobs : float
        Observation time.
    N : int
        The +/- N terms to be included in the summation calculation.

    Returns
    -------
    array
        Fourier transform calculated along the input frequencies array.
    """
    Phi0_1 = phi0 - epsilon*np.sin(phip)
    Phi0_2 = phi0 - epsilon*np.sin(phip) + np.pi/2
    
    term1 = M_mono_FM_FT(A1, fc, fm, epsilon, Phi0_1, phip, t0, freqs, Tobs, N)
    term2 = M_mono_FM_FT(A2, fc, fm, epsilon, Phi0_2, phip, t0, freqs, Tobs, N)
    
    return term1 + term2



# Chirping: FM (time domain)
def M_chirp_FM(A, fc, fm, dfprime, epsilon, phi0, phip, t0, times):
    """Phase chirping, frequency modulated model.

    Parameters
    ----------
    A : float
        Amplitude.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    dfprime : float
        Frequency derivative.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    times : array
        Times.

    Returns
    -------
    array
        Model calculated along the input time array.
    """
    wc      = 2*np.pi*fc
    wm      = 2*np.pi*fm
    dwprime = 2*np.pi*dfprime
    return A*np.cos(phi0 + wc*(times-t0) + epsilon*np.sin(phip + wm*(times-t0)) + 1/2*dwprime*(times-t0)**2)


# Chirping: FM (frequency domain)
def M_chirp_FM_FT(A, fc, fm, dfprime, epsilon, phi0, phip, t0, freqs, Tobs, N):
    """Fourier Transform of the phase chirping, frequency modulated model.

    Parameters
    ----------
    A : float
        Amplitude.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    dfprime : float
        Frequency derivative.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    freqs : array
        Frequencies.
    Tobs : float
        Observation time.
    N : int
        The +/- N terms to be included in the summation calculation.

    Returns
    -------
    array
        Fourier transform calculated along the input frequencies array.
    """
    wc      = 2*np.pi*fc
    wm      = 2*np.pi*fm
    dwprime = 2*np.pi*dfprime
    
    n = np.arange(-N,N+1,1)
    FT = []

    # spin-up
    if dwprime > 0:
        for fval in freqs:
            w = 2*np.pi*fval
    
            alpham = (wc + n*wm - w)/np.sqrt(2*dwprime)
            alphap = (wc - n*wm + w)/np.sqrt(2*dwprime)
            
            Xm_l = (wc + n*wm - w)/np.sqrt(np.pi*dwprime)
            Xp_l = (wc - n*wm + w)/np.sqrt(np.pi*dwprime)
            Xm_u = np.sqrt(dwprime/np.pi)*Tobs + Xm_l
            Xp_u = np.sqrt(dwprime/np.pi)*Tobs + Xp_l
            
            Sm_u, Cm_u = sp.fresnel(Xm_u)
            Sm_l, Cm_l = sp.fresnel(Xm_l)
            Sp_u, Cp_u = sp.fresnel(Xp_u)
            Sp_l, Cp_l = sp.fresnel(Xp_l)
            
            term1 = np.exp(1j*(phi0-alpham**2)) * ((Cm_u - Cm_l) + 1j*(Sm_u - Sm_l))
            term2 = np.exp(-1j*(phi0-alphap**2)) * ((Cp_u - Cp_l) - 1j*(Sp_u - Sp_l))
            
            FT_f = np.exp(-1j*w*t0)*A/2*np.sqrt(np.pi/dwprime) * np.exp(1j*n*phip) * sp.jv(n,epsilon) * (term1 + ((-1.0)**n)*term2)
            FT_f = np.sum(FT_f)
            
            FT += [FT_f]
    
    # spin-down
    if dwprime < 0:
        dwprime = np.abs(dwprime)
        
        for fval in freqs:  
            w = 2*np.pi*fval
            
            alpham = (wc + n*wm - w)/np.sqrt(2*dwprime)
            alphap = (wc - n*wm + w)/np.sqrt(2*dwprime)
            
            Xm_l = -(wc + n*wm - w)/np.sqrt(np.pi*dwprime)
            Xp_l = -(wc - n*wm + w)/np.sqrt(np.pi*dwprime)
            Xm_u = np.sqrt(dwprime/np.pi)*Tobs + Xm_l
            Xp_u = np.sqrt(dwprime/np.pi)*Tobs + Xp_l
            
            Sm_u, Cm_u = sp.fresnel(Xm_u)
            Sm_l, Cm_l = sp.fresnel(Xm_l)
            Sp_u, Cp_u = sp.fresnel(Xp_u)
            Sp_l, Cp_l = sp.fresnel(Xp_l)
            
            term1 = np.exp(1j*(phi0+alpham**2)) * ((Cm_u - Cm_l) - 1j*(Sm_u - Sm_l))
            term2 = np.exp(-1j*(phi0+alphap**2)) * ((Cp_u - Cp_l) + 1j*(Sp_u - Sp_l))
            
            FT_f = np.exp(-1j*w*t0)*A/2*np.sqrt(np.pi/dwprime) * np.exp(1j*n*phip) * sp.jv(n,epsilon) * (term1 + ((-1.0)**n)*term2)
            FT_f = np.sum(FT_f)
            
            FT += [FT_f]
            
    return np.asarray(FT)



# Chirping: GW + Exoplanet (time domain)
def M_chirp_GW(A1, A2, fc, fm, dfgw0, epsilon, phi0, phip, t0, times):
    """GW + exoplanet model: phase chirping, frequency modulated

    Parameters
    ----------
    A1 : float
        Amplitude 1 term.
    A2 : float
        Amplitude 2 term.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    dfgw0 : float
        GW frequency derivative.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    times : array
        Times.

    Returns
    -------
    array
        Model calculated along the input time array.
    """
    wc  = 2*np.pi*fc
    wm  = 2*np.pi*fm
    dw0 = np.pi*dfgw0

    Phase = phi0 + wc*(times-t0) + epsilon*(np.sin(phip + wm*(times-t0)) - np.sin(phip)) + dw0*(times-t0)**2
    
    return A1*np.cos(Phase) - A2*np.sin(Phase)


# Chirping: GW + Exoplanet (frequency domain)
def M_chirp_GW_FT(A1, A2, fc, fm, dfgw0, epsilon, phi0, phip, t0, freqs, Tobs, N):
    """Wrapper function to calculate the Fourier Transform for the GW + exoplanet phase chirping, frequency modulated model.

    Parameters
    ----------
    A1 : float
        Amplitude 1 term.
    A2 : float
        Amplitude 2 term.
    fc : float
        Carrier frequency.
    fm : float
        Message frequency.
    dfgw0 : float
        GW frequency derivative.
    epsilon : float
        Modulation index.
    phi0 : float
        Initial phase.
    phip : float
        Message initial phase.
    t0 : float
        Initial time.
    freqs : array
        Frequencies.
    Tobs : float
        Observation time.
    N : int
        The +/- N terms to be included in the summation calculation.

    Returns
    -------
    array
        Fourier transform calculated along the input frequencies array.
    """
    Phi0_1  = phi0 - epsilon*np.sin(phip)
    Phi0_2  = phi0 - epsilon*np.sin(phip) + np.pi/2
    dfprime = dfgw0
    
    term1 = M_chirp_FM_FT(A1, fc, fm, dfprime, epsilon, Phi0_1, phip, t0, freqs, Tobs, N)
    term2 = M_chirp_FM_FT(A2, fc, fm, dfprime, epsilon, Phi0_2, phip, t0, freqs, Tobs, N)
    
    return term1 + term2








