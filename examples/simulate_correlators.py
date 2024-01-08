import navsim as ns
import numpy as np

from pathlib import Path
from tqdm import tqdm
from navsim import CorrelatorSimulation
from navsim.configuration import SimulationConfiguration

# User-Defined Parameters
IS_EMITTER_TYPE_TRUTH = True
TAP_SPACING = [0]

# Path Parameters
PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "config"
DATA_PATH = PROJECT_PATH / "data"


def simulate():
    rx_pos = np.array([423756, -5361363, 3417705])

    configuration = ns.get_configuration(configuration_path=CONFIG_PATH)

    sim = CorrelatorSimulation(configuration=configuration)
    sim_emitter_states, sim_observables, sim_rx_states = simulate_measurements(
        configuration=configuration, rx_pos=rx_pos
    )

    if IS_EMITTER_TYPE_TRUTH:
        emitters = sim_emitter_states.truth
    else:
        emitters = sim_emitter_states.ephemeris

    for epoch, observables in tqdm(
        enumerate(sim_observables),
        total=len(sim_observables),
        desc="[navsim] simulating correlators",
    ):
        # time update

        # predict observables
        pranges = np.repeat(
            np.array([emitter.range for emitter in emitters[epoch].values()]).reshape(
                1, -1
            ),
            1000,
            axis=0,
        ).T
        prange_rates = np.repeat(
            np.array(
                [emitter.range_rate for emitter in emitters[epoch].values()]
            ).reshape(1, -1),
            1000,
            axis=0,
        ).T

        sim.compute_errors(
            observables=observables,
            est_pranges=pranges,
            est_prange_rates=prange_rates,
        )
        correlators = [
            sim.correlate(tap_spacing=tap_spacing, include_subcorrelators=False)
            for tap_spacing in TAP_SPACING
        ]

        # close loop


def simulate_measurements(
    configuration: SimulationConfiguration,
    rx_pos: np.ndarray,
    rx_vel: np.ndarray = None,
):
    measurement_sim = ns.get_signal_simulation(
        simulation_type="measurement", configuration=configuration
    )
    measurement_sim.generate_truth(rx_pos=rx_pos, rx_vel=rx_vel)
    measurement_sim.simulate()

    return (
        measurement_sim.emitter_states,
        measurement_sim.observables,
        measurement_sim.rx_states,
    )


if __name__ == "__main__":
    simulate()
