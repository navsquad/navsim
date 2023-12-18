import numpy as np
import navsim as ns
import navtools as nt
from navtools.constants import SPEED_OF_LIGHT
from navsim.config import SimulationConfiguration
from navsim.error_models.signal import (
    compute_carrier_range_error,
    compute_code_range_error,
    compute_range_rate_error,
)


class CorrelatorResiduals:
    code_pseudorange: np.ndarray
    chip: np.ndarray
    carrier_pseudorange: np.ndarray
    phase: np.ndarray
    pseudorange_rate: np.ndarray
    frequency: np.ndarray


class CorrelatorSimulation:
    def __init__(self, configuration: SimulationConfiguration) -> None:
        self.__measurement_sim = ns.get_signal_simulation(
            simulation_type="measurement", configuration=configuration
        )
        self.__correlator_models = {
            constellation.casefold(): nt.get_correlator_model(
                correlator_name=signal.correlator_model
            )
            for constellation, signal in configuration.constellations.emitters.items()
        }
        self.epoch = 0

    @property
    def emitter_states(self):
        return self.__emitter_states

    def simulate_measurements(self, rx_pos: np.ndarray, rx_vel: np.ndarray = None):
        self.__measurement_sim.simulate(rx_pos=rx_pos, rx_vel=rx_vel)

        self.__emitter_states = self.__measurement_sim.emitter_states
        self.__observables = self.__measurement_sim.observables
        self.__rx_states = self.__measurement_sim.rx_states
        self.__signal_properties = self.__measurement_sim.signal_properties

        return self.__emitter_states, self.__observables, self.__rx_states

    def correlate(
        self,
        predicted_pseudoranges: np.ndarray,
        predicted_pseudorange_rates: np.ndarray,
    ):
        epoch_observables = self.__observables[self.epoch]
        residuals = self.__compute_residuals(
            observables=epoch_observables,
            predicted_pseudoranges=predicted_pseudoranges,
            predicted_pseudorange_rates=predicted_pseudorange_rates,
        )
        self.epoch += 1

    def __compute_residuals(
        self,
        observables: dict,
        predicted_pseudoranges: np.ndarray,
        predicted_pseudorange_rates: np.ndarray,
    ):
        carrier_pseudoranges = np.array(
            [emitter.carrier_pseudorange for emitter in observables.values()]
        )
        code_pseudoranges = np.array(
            [emitter.code_pseudorange for emitter in observables.values()]
        )
        pseudorange_rates = np.array(
            [emitter.pseudorange_rate for emitter in observables.values()]
        )

        chip_length, wavelength = self.__compute_signal_cycle_lengths(
            observables=observables
        )

        code_pseudorange_residual, chip_residual = compute_code_range_error(
            true_pseudorange=carrier_pseudoranges,
            predicted_pseudorange=predicted_pseudoranges,
            chip_length=chip_length,
        )
        carrier_pseudorange_residual, phase_residual = compute_carrier_range_error(
            true_pseudorange=code_pseudoranges,
            predicted_pseudorange=predicted_pseudoranges,
            wavelength=wavelength,
        )
        pseudorange_rate_residual, fresidual = compute_range_rate_error(
            true_pseudorange_rate=pseudorange_rates,
            predicted_pseudorange_rate=predicted_pseudorange_rates,
            wavelength=wavelength,
        )

        residuals = CorrelatorResiduals(
            code_pseudorange=code_pseudorange_residual,
            chip=chip_residual,
            carrier_pseudorange=carrier_pseudorange_residual,
            phase=phase_residual,
            pseudorange_rate=pseudorange_rate_residual,
            frequency=fresidual,
        )

        return residuals

    def __compute_signal_cycle_lengths(self, observables: dict):
        chip_length = []
        wavelength = []

        for emitter in observables.values():
            fcarrier = self.__signal_properties.get(emitter.constellation).fcarrier
            # ! not generalized for pilot tracking although this value is likely the same
            fchip = self.__signal_properties.get(emitter.constellation).fchip_data

            ratio = fchip / fcarrier
            carrier_doppler = emitter.carrier_doppler
            code_doppler = carrier_doppler * ratio

            chip_length.append(SPEED_OF_LIGHT / (fchip + code_doppler))
            wavelength.append(SPEED_OF_LIGHT / (fcarrier + carrier_doppler))

        return np.array(chip_length), np.array(wavelength)
