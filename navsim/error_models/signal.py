import numpy as np
from numba import njit

from navtools.constants import SPEED_OF_LIGHT, BOLTZMANN


@njit(cache=True)
def compute_carrier_to_noise(
    range: float,
    transmit_power: float,
    transmit_antenna_gain: float,
    fcarrier: float,
    js: float = 0,
    temperature: float = 290,
):
    """computes carrier-to-noise density ratio from free space path loss (derived from Friis equation)

    Parameters
    ----------
    range : float
        range to emitter [m]
    transmit_power : float
        trasmit power [dBW]
    transmit_antenna_gain : float
        isotropic antenna gain [dBi]
    fcarrier : float
        signal carrier frequency [Hz]
    js : float, optional
        jammer-to-signal ratio [dB], by default 0
    temperature : float, optional
        noise temperature [K], by default 290

    Returns
    -------
    float
        carrier-to-noise ratio [dB-Hz]

    Reference
    -------
    A. Joseph, “Measuring GNSS Signal Strength: What is the difference between SNR and C/N0?,” InsideGNSS, Nov. 2010.
    """

    ADDITIONAL_NOISE_FIGURE = 3  # [dB-Hz] cascaded + band-limiting/quantization noise

    EIRP = transmit_power + transmit_antenna_gain  # [dBW]
    wavelength = SPEED_OF_LIGHT / fcarrier  # [m]
    FSPL = 20 * np.log10(4 * np.pi * range / wavelength)  # [dB] free space path loss

    received_carrier_power = EIRP - FSPL - js  # [dBW]
    thermal_noise_density = 10 * np.log10(BOLTZMANN * temperature)  # [dBW/Hz]

    nominal_cn0 = received_carrier_power - thermal_noise_density  # [dB-Hz]
    cn0 = nominal_cn0 - ADDITIONAL_NOISE_FIGURE

    return cn0
