__all__ = ["CorrelatorErrors", "CorrelatorOutputs", "CorrelatorSimulation"]

import navtools as nt
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from navtools.constants import SPEED_OF_LIGHT
from navsim.configuration import SimulationConfiguration
from navsim.error_models import compute_range_error, compute_range_rate_error


@dataclass(frozen=True)
class CorrelatorErrors:
    code_prange: np.ndarray
    carr_prange: np.ndarray
    prange_rate: np.ndarray
    chip: np.ndarray
    carr_phase: np.ndarray
    freq: np.ndarray


@dataclass(frozen=True)
class CorrelatorOutputs:
    inphase: np.ndarray
    subinphase: np.ndarray
    quadrature: np.ndarray
    subquadrature: np.ndarray


class CorrelatorSimulation:
    @property
    def chip_errors(self):
        return dict(self.__chip_err_log)

    @property
    def code_prange_errors(self):
        return dict(self.__code_prange_err_log)

    @property
    def cphase_errors(self):
        return dict(self.__cphase_err_log)

    @property
    def carrier_prange_errors(self):
        return dict(self.__carr_prange_err_log)

    @property
    def ferrors(self):
        return dict(self.__ferr_log)

    @property
    def prange_rate_errors(self):
        return dict(self.__prange_rate_err_log)

    def __init__(self, configuration: SimulationConfiguration) -> None:
        constellations = configuration.constellations

        # signals
        self.__signals = {
            constellation: signal
            for constellation, signal in constellations.emitters.items()
        }
        self.__observables = None

        # correlators
        self.__correlators = {
            constellation: nt.get_correlator_model(
                correlator_name=emitter.correlator_model
            )
            for constellation, emitter in constellations.emitters.items()
        }

        # errors
        self.__errors = None
        self.__chip_err_log = defaultdict(lambda: [])
        self.__code_prange_err_log = defaultdict(lambda: [])
        self.__ferr_log = defaultdict(lambda: [])
        self.__prange_rate_err_log = defaultdict(lambda: [])
        self.__cphase_err_log = defaultdict(lambda: [])
        self.__carr_prange_err_log = defaultdict(lambda: [])

        # time
        self.T = 1 / configuration.time.fsim

    def compute_errors(
        self, observables: dict, est_pranges: np.ndarray, est_prange_rates: np.ndarray
    ):
        self.__observables = observables
        self.__nemitters = len(self.__observables)

        # extract necessary observables
        carr_pranges = np.array(
            [emitter.carrier_pseudorange for emitter in self.__observables.values()]
        )
        code_pranges = np.array(
            [emitter.code_pseudorange for emitter in self.__observables.values()]
        )
        prange_rates = np.array(
            [emitter.pseudorange_rate for emitter in self.__observables.values()]
        )

        chip_length, wavelength = self.__compute_cycle_lengths(observables=observables)

        code_prange_err, chip_err = compute_range_error(
            true_prange=code_pranges,
            est_prange=est_pranges,
            cycle_length=chip_length,
        )
        carrier_prange_err, cphase_err = compute_range_error(
            true_prange=carr_pranges,
            est_prange=est_pranges,
            cycle_length=wavelength,
        )
        prange_rate_err, ferr = compute_range_rate_error(
            true_prange_rate=prange_rates,
            est_prange_rate=est_prange_rates,
            wavelength=wavelength,
        )

        self.__errors = CorrelatorErrors(
            code_prange=self.__sort_constellation_errors(code_prange_err),
            carr_prange=self.__sort_constellation_errors(carrier_prange_err),
            prange_rate=self.__sort_constellation_errors(prange_rate_err),
            chip=self.__sort_constellation_errors(chip_err),
            carr_phase=self.__sort_constellation_errors(cphase_err),
            freq=self.__sort_constellation_errors(ferr),
        )
        self.epoch_errors = CorrelatorErrors(
            code_prange=code_prange_err,
            carr_prange=carrier_prange_err,
            prange_rate=prange_rate_err,
            chip=chip_err,
            carr_phase=cphase_err,
            freq=ferr,
        )

        return self.epoch_errors

    def correlate(
        self,
        tap_spacing: float = 0.0,
        nsubcorrelators: int = 2,
        include_subcorrelators: bool = True,
        include_noise: bool = True,
    ):
        inphase = []
        subinphase = []
        quadrature = []
        subquadrature = []

        for constellation, correlator in self.__correlators.items():
            cn0 = np.array(
                [
                    emitter.cn0
                    for emitter in self.__observables.values()
                    if emitter.constellation == constellation
                ]
            )

            chip_error = np.array(self.__errors.chip.get(constellation))
            ferror = np.array(self.__errors.freq.get(constellation))
            phase_error = np.array(self.__errors.carr_phase.get(constellation))

            subtime_T = np.flip(
                self.T * np.arange(0, nsubcorrelators) / nsubcorrelators
            )

            # * assumes linear ferror over integration period *
            subphase_deltas = np.array([ferror * T for T in subtime_T])
            subphase_errors = phase_error - subphase_deltas
            mean_phase_errors = np.mean(subphase_errors, axis=0)

            I, Q = correlator(
                T=self.T,
                cn0=cn0,
                chip_error=chip_error,
                ferror=ferror,
                phase_error=mean_phase_errors,
                tap_spacing=tap_spacing,
                include_noise=include_noise,
            )

            if include_subcorrelators:
                subI, subQ = correlator(
                    T=self.T / nsubcorrelators,
                    cn0=cn0,
                    chip_error=chip_error,
                    ferror=ferror,
                    phase_error=subphase_errors,
                    tap_spacing=tap_spacing,
                    include_noise=include_noise,
                )
                subinphase.append(subI)
                subquadrature.append(subQ)

            inphase.append(I)
            quadrature.append(Q)

        inphase = np.hstack(inphase)
        quadrature = np.hstack(quadrature)

        if include_subcorrelators:
            subinphase = np.hstack(subinphase)
            subquadrature = np.hstack(subquadrature)
        else:
            subinphase = None
            subquadrature = None

        outputs = CorrelatorOutputs(
            inphase=inphase,
            subinphase=subinphase,
            quadrature=quadrature,
            subquadrature=subquadrature,
        )

        return outputs

    def clear_errors(self):
        self.__errors = None

        self.__chip_err_log = defaultdict(lambda: [])
        self.__code_prange_err_log = defaultdict(lambda: [])
        self.__ferr_log = defaultdict(lambda: [])
        self.__prange_rate_err_log = defaultdict(lambda: [])
        self.__cphase_err_log = defaultdict(lambda: [])
        self.__carr_prange_err_log = defaultdict(lambda: [])

    def log_errors(self):
        self.__log_by_emitter(data=self.epoch_errors.chip, log=self.__chip_err_log)
        self.__log_by_emitter(
            data=self.epoch_errors.code_prange, log=self.__code_prange_err_log
        )
        self.__log_by_emitter(
            data=self.epoch_errors.carr_phase, log=self.__cphase_err_log
        )
        self.__log_by_emitter(
            data=self.epoch_errors.carr_prange, log=self.__carr_prange_err_log
        )
        self.__log_by_emitter(data=self.epoch_errors.freq, log=self.__ferr_log)
        self.__log_by_emitter(
            data=self.epoch_errors.prange_rate, log=self.__prange_rate_err_log
        )

    def __compute_cycle_lengths(self, observables: dict):
        chip_length = []
        wavelength = []

        for emitter in observables.values():
            properties = self.__signals.get(emitter.constellation).properties
            fcarrier = properties.fcarrier
            # ! assumes tracking data channel (fchip_data) !
            fchip = properties.fchip_data

            fratio = fchip / fcarrier
            carrier_doppler = emitter.carrier_doppler
            code_doppler = carrier_doppler * fratio

            chip_length.append(SPEED_OF_LIGHT / (fchip + code_doppler))
            wavelength.append(SPEED_OF_LIGHT / (fcarrier + carrier_doppler))

        chip_length = np.array(chip_length)
        wavelength = np.array(wavelength)

        return chip_length, wavelength

    def __sort_constellation_errors(self, errors: np.ndarray):
        errors = nt.smart_transpose(
            col_size=self.__nemitters, transformed_array=errors
        ).T  # transposing again to ensure nrows=nemitters

        sorted_errors = defaultdict(lambda: [])

        for emitter_index, emitter in enumerate(self.__observables.values()):
            sorted_errors[emitter.constellation].append(errors[emitter_index])

        return sorted_errors

    def __log_by_emitter(self, data: np.ndarray, log: defaultdict):
        for idx, (emitter_id, emitter) in enumerate(self.__observables.items()):
            log[(emitter.constellation, emitter_id)].append(data[idx])
