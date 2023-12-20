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
    carrier_prange: np.ndarray
    prange_rate: np.ndarray
    chip: np.ndarray
    carrier_phase: np.ndarray
    frequency: np.ndarray


@dataclass(frozen=True)
class CorrelatorOutputs:
    inphase: np.ndarray
    quadrature: np.ndarray


class CorrelatorSimulation:
    @property
    def errors(self):
        return self.__errors

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
        self.__errors = None

        # time
        self.T = 1 / configuration.time.fsim

    def compute_errors(
        self, observables: dict, est_pranges: np.ndarray, est_prange_rates: np.ndarray
    ):
        self.__observables = observables
        self.__nemitters = len(self.__observables)

        # extract necessary observables
        carrier_pranges = np.array(
            [emitter.carrier_pseudorange for emitter in self.__observables.values()]
        )
        code_pranges = np.array(
            [emitter.code_pseudorange for emitter in self.__observables.values()]
        )
        prange_rates = np.array(
            [emitter.pseudorange_rate for emitter in self.__observables.values()]
        )

        chip_length, wavelength = self.__compute_cycle_lengths(observables=observables)

        code_prange_error, chip_error = compute_range_error(
            true_prange=carrier_pranges,
            est_prange=est_pranges,
            cycle_length=chip_length,
        )
        carrier_prange_error, cphase_error = compute_range_error(
            true_prange=code_pranges,
            est_prange=est_pranges,
            cycle_length=wavelength,
        )
        prange_rate_error, ferror = compute_range_rate_error(
            true_prange_rate=prange_rates,
            est_prange_rate=est_prange_rates,
            wavelength=wavelength,
        )

        a = self.__sort_errors(code_prange_error)

        self.__errors = CorrelatorErrors(
            code_prange=self.__sort_errors(code_prange_error),
            carrier_prange=self.__sort_errors(carrier_prange_error),
            prange_rate=self.__sort_errors(prange_rate_error),
            chip=self.__sort_errors(chip_error),
            carrier_phase=self.__sort_errors(cphase_error),
            frequency=self.__sort_errors(ferror),
        )

    def correlate(self, tap_spacing: float = 0.0):
        inphase = []
        quadrature = []

        for constellation, correlator in self.__correlators.items():
            cn0 = np.array(
                [
                    emitter.cn0
                    for emitter in self.__observables.values()
                    if emitter.constellation == constellation
                ]
            )

            chip_error = np.array(self.__errors.chip.get(constellation))
            ferror = np.array(self.__errors.frequency.get(constellation))
            phase_error = np.array(self.__errors.carrier_phase.get(constellation))

            I, Q = correlator(
                T=self.T,
                cn0=cn0,
                chip_error=chip_error,
                ferror=ferror,
                phase_error=phase_error,
                tap_spacing=tap_spacing,
            )

            inphase.append(I)
            quadrature.append(Q)

        inphase = np.hstack(inphase)
        quadrature = np.hstack(quadrature)

        outputs = CorrelatorOutputs(inphase=inphase, quadrature=quadrature)

        return outputs

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

    def __sort_errors(self, errors: np.ndarray):
        errors = nt.smart_transpose(
            col_size=self.__nemitters, transformed_array=errors
        ).T  # transposing again to ensure nrows=nemitters

        sorted_errors = defaultdict(lambda: [])

        for emitter_index, emitter in enumerate(self.__observables.values()):
            sorted_errors[emitter.constellation].append(errors[emitter_index])

        return sorted_errors
