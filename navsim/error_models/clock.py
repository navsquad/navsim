"""all clock models were retrieved from the following source:

R. G. Brown and P. Y. C. Hwang, Introduction to random signals and applied Kalman filtering: with MATLAB exercises, 4. ed. Hoboken, NJ: Wiley, 2012.
ch. 9.3, pg. 327
"""

import numpy as np
from dataclasses import dataclass
from numba import njit

from navtools.constants import SPEED_OF_LIGHT


@dataclass(frozen=True)
class NavigationClock:
    """dataclass of navigation clock Allan variance values"""

    h0: float
    h1: float
    h2: float


LOW_QUALITY_TCXO = NavigationClock(h0=2e-19, h1=7e-21, h2=2e-20)
HIGH_QUALITY_TCXO = NavigationClock(h0=2e-21, h1=1e-22, h2=2e-20)
OCXO = NavigationClock(h0=2e-25, h1=7e-25, h2=6e-25)
RUBIDIUM = NavigationClock(h0=2e-22, h1=4.5e-26, h2=1e-30)
CESIUM = NavigationClock(h0=2e-22, h1=5e-27, h2=1.5e-33)


@njit(cache=True)
def compute_clock_states(
    h0: float, h2: float, T: float, nperiods: int = 1
) -> tuple[np.array, np.array]:
    """computes clock bias and drift using two-state clock model for specified period and number of periods

    Parameters
    ----------
    h0 : float
    h2 : float
    T : float
        sampling period
    nperiods : int, optional
        number of total sampling periods, by default 1

    Returns
    -------
    tuple[np.array, np.array]
        clock bias and drift

    Reference
    -------
    L. Galleani, “A tutorial on the two-state model of the atomic clock noise,” Metrologia, vol. 45, no. 6, p. S175, Dec. 2008, doi: 10.1088/0026-1394/45/6/S23.
    """
    # two-state clock model white noise spectral amplitudes
    sf = h0 / 2
    sg = h2 * 2 * np.pi**2

    # error variances
    sigma_phase_error = np.sqrt(sf * T + (1 / 3) * sg * T**3)  # [s]
    sigma_freq_error = np.sqrt(sg * T)  # [s/s]

    drift_ss = np.cumsum(sigma_freq_error * np.random.randn(nperiods))
    bias_s = np.cumsum(drift_ss * T) + np.cumsum(
        sigma_phase_error * np.random.randn(nperiods)
    )

    drift_ms = drift_ss * SPEED_OF_LIGHT
    bias_m = bias_s * SPEED_OF_LIGHT

    return bias_m, drift_ms


def get_clock_allan_variance_values(clock_name: str):
    """factory function that retrieves requested clock Allan variance values

    Parameters
    ----------
    clock_name : str
        name of clock

    Returns
    -------
    NavigationClock
        clock Allan variance values
    """
    CLOCKS = {
        "lowqualitytcxo": LOW_QUALITY_TCXO,
        "highqualitytcxo": HIGH_QUALITY_TCXO,
        "ocxo": OCXO,
        "rubidium": RUBIDIUM,
        "cesium": CESIUM,
    }

    clock_name = "".join([i for i in clock_name if i.isalnum()]).casefold()
    clock = CLOCKS.get(clock_name.casefold(), CESIUM)  # defaults to cesium

    return clock
