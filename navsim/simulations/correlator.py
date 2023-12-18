import numpy as np
import navsim as ns
from navsim.config import SimulationConfiguration, ErrorConfiguration
from navsim.simulations.simulation import SignalSimulation
from collections import defaultdict
from dataclasses import dataclass, replace


class CorrelatorSimulation:
    def __init__(self, configuration: SimulationConfiguration) -> None:
        self.__init_measurement_simulation(configuration=configuration)
        self.__init_errors(configuration=configuration.errors)

    @property
    def ephemeris_emitters(self):
        return self.__ephemeris_emitters

    def __init_measurement_simulation(self, configuration: SimulationConfiguration):
        self.__measurement_sim = ns.get_signal_simulation(simulation_type="measurement")
        self.__measurement_sim.initialize(configuration=configuration)

    def __init_errors(self, configuration: ErrorConfiguration):
        self.__is_ephemeris_perturbed = configuration.emitter_ephemeris

    def simulate_observables(self, rx_pos: np.ndarray, rx_vel: np.ndarray = None):
        # measurement simulation
        self.__perform_measurement_simulation(rx_pos=rx_pos, rx_vel=rx_vel)

        if self.__is_ephemeris_perturbed:
            self.__ephemeris_emitters = self.__perturb_emitter_states()
        else:
            self.__ephemeris_emitters = self.__emitter_states

    def __perform_measurement_simulation(self, rx_pos: np.ndarray, rx_vel: np.ndarray):
        self.__measurement_sim.simulate(rx_pos=rx_pos, rx_vel=rx_vel)

        self.__emitter_states = self.__measurement_sim.emitter_states
        self.__observables = self.__measurement_sim.observables
        self.__signal_properties = self.__measurement_sim.signal_properties

    def __perturb_emitter_states(self):
        perturbed_emitter_states = []

        for states in self.__emitter_states:
            new_states = defaultdict()

            for state in states.values():
                new_state = replace(state)

                # clear truth values
                new_state.az = 0
                new_state.el = 0
                new_state.clock_bias = 0
                new_state.clock_drift = 0
                new_state.range = 0
                new_state.range_rate = 0

                # not a great assumption condsidering the noise is correlated
                new_state.pos = state.pos + 0.5 * np.random.randn(*state.pos.shape)
                new_state.vel = state.vel + 0.15 * np.random.randn(*state.vel.shape)

                new_states[new_state.id] = new_state
            perturbed_emitter_states.append(new_states)

        return perturbed_emitter_states
