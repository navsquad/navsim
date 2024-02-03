import numpy as np

from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import ClassVar
from numba import njit

from laika import AstroDog
from laika.gps_time import GPSTime
from navtools.conversions.coordinates import ecef2lla
from navtools.constants import SPEED_OF_LIGHT


# Ionosphere
@dataclass
class IonosphereModelParameters:
    time: datetime
    rx_pos: np.ndarray
    emitter_pos: np.ndarray
    az: float
    el: float
    fcarrier: float
    alpha: ClassVar[np.ndarray[float]] = np.array([2.6768e-08, 4.4914e-09, -3.2658e-07, -5.2153e-07])
    beta: ClassVar[np.ndarray[float]] = np.array([1.3058e05, -1.1203e05, -7.0416e05, -6.4865e06])


class IonosphereModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_delay(self, params: IonosphereModelParameters):
        pass


class TotalElectronCountMapModel(IonosphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: IonosphereModelParameters):
        gps_time = GPSTime.from_datetime(datetime=params.time)

        if not hasattr(self, "dog"):
            self.dog = AstroDog()
            self.ionex = self.dog.get_ionex(time=gps_time)

        delay = self.ionex.get_delay(
            time=gps_time,
            freq=params.fcarrier,
            rcv_pos=params.rx_pos,
            sat_pos=params.emitter_pos,
            az=params.az,
            el=params.el,
        )

        return delay


class KlobucharModel(IonosphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: IonosphereModelParameters):
        gps_time = GPSTime.from_datetime(datetime=params.time)
        lla = ecef2lla(params.rx_pos)

        delay = compute_klobuchar_delay(
            lat=lla[0],
            lon=lla[1],
            az=params.az,
            el=params.el,
            tow=gps_time.tow,
            alpha=params.alpha,
            beta=params.beta,
        )

        return delay


@njit(cache=True)
def compute_klobuchar_delay(
    lat: float,
    lon: float,
    az: float,
    el: float,
    tow: float,
    alpha: np.array,
    beta: np.array,
):
    psi = 0.0137 / (el / np.pi + 0.11) - 0.022
    phi = lat / np.pi + psi * np.cos(az)

    if phi > 0.416:
        phi = 0.416
    elif phi < -0.416:
        phi = -0.416
    lam = lon / np.pi + psi * np.sin(az) / np.cos(phi * np.pi)

    local_tod = 43200.0 * lam + tow
    local_tod -= np.floor(local_tod / 86400.0) * 86400.0  # [s]

    slant_factor = 1.0 + 16.0 * np.power(0.53 - el / np.pi, 3.0)

    amplitude = alpha[0] + phi * (alpha[1] + phi * (alpha[2] + phi * alpha[3]))
    period = beta[0] + phi * (beta[1] + phi * (beta[2] + phi * beta[3]))
    if amplitude < 0.0:
        amplitude = 0.0
    if period < 72000.0:
        period = 72000.0

    phase = 2.0 * np.pi * (local_tod - 50400.0) / period

    multiple = 5e-9
    if np.abs(phase) < 1.57:
        multiple += amplitude * (1.0 + phase * phase * (-0.5 + phase * phase / 24.0))

    delay = SPEED_OF_LIGHT * slant_factor * multiple

    return delay


# Troposphere
@dataclass
class TroposphereModelParameters:
    rx_pos: np.array
    el: float


class TroposphereModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_delay(self, params: TroposphereModelParameters):
        pass


class SaastamoinenModel(TroposphereModel):
    def __init__(self) -> None:
        super().__init__()

    def get_delay(self, params: TroposphereModelParameters):
        delay = compute_saastamoinen_delay(rx_pos=params.rx_pos, el=params.el)

        return delay


@njit(cache=True)
def compute_saastamoinen_delay(
    rx_pos, el, humidity=0.75, temperature_at_sea_level=15.0
):
    """function from RTKlib: https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c#L3362-3362
        with no changes by way of laika: https://github.com/commaai/laika

    Parameters
    ----------
    rx_pos : _type_
        receiver ECEF position
    el : _type_
        elevation to emitter [rad]
    humidity : float, optional
        relative humidity, by default 0.75
    temperature_at_sea_level : float, optional
        temperature at sea level [C], by default 15.0

    Returns
    -------
    _type_
        sum of wet and dry tropospheric delay [m]
    """
    # TODO: clean this up
    rx_pos_lla = ecef2lla(rx_pos)
    if rx_pos_lla[2] < -1e3 or 1e4 < rx_pos_lla[2] or el <= 0:
        return 0.0

    hgt = 0.0 if rx_pos_lla[2] < 0.0 else rx_pos_lla[2]  # standard atmosphere

    pres = 1013.25 * pow(1.0 - 2.2557e-5 * hgt, 5.2568)
    temp = temperature_at_sea_level - 6.5e-3 * hgt + 273.16
    e = 6.108 * humidity * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))

    # /* saastamoninen model */
    z = np.pi / 2.0 - el
    trph = (
        0.0022768
        * pres
        / (1.0 - 0.00266 * np.cos(2.0 * rx_pos_lla[0]) - 0.00028 * hgt / 1e3)
        / np.cos(z)
    )
    trpw = 0.002277 * (1255.0 / temp + 0.05) * e / np.cos(z)
    return trph + trpw


# Factories
def get_ionosphere_model(model_name: str):
    """factory function that retrieves requested ionosphere model

    Parameters
    ----------
    model_name : str
        name of ionosphere model

    Returns
    -------
    IonosphereModel
        ionosphere model
    """
    IONOSPHERE_MODELS = {
        "tecmap": TotalElectronCountMapModel(),
        "klobuchar": KlobucharModel(),
    }

    model_name = "".join([i for i in model_name if i.isalnum()]).casefold()
    model = IONOSPHERE_MODELS.get(
        model_name.casefold(),
        TotalElectronCountMapModel(),  # defaults to tec map
    )

    return model


def get_troposphere_model(model_name: str):
    """factory function that retrieves requested troposphere model

    Parameters
    ----------
    model_name : str
        name of troposphere model

    Returns
    -------
    TroposphereModel
        troposphere model
    """
    TROPOSPHERE_MODELS = {"saastamoinen": SaastamoinenModel()}

    model_name = "".join([i for i in model_name if i.isalnum()]).casefold()
    model = TROPOSPHERE_MODELS.get(
        model_name.casefold(), SaastamoinenModel()
    )  # defaults to saastamoinen

    return model
