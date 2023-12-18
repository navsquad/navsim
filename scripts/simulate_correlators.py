import numpy as np
import navsim as ns

from navsim.simulations import CorrelatorSimulation
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_PATH / "config"
DATA_PATH = PROJECT_PATH / "data"

rx_pos = np.array([423756, -5361363, 3417705])

configuration = ns.get_configuration(configuration_path=CONFIG_PATH)
sim = CorrelatorSimulation(configuration=configuration)

emitter_states, observables, rx_states = sim.simulate_measurements(rx_pos=rx_pos)
ephemeris_emitters = emitter_states.ephemeris_emitter_states

for epoch, emitter_states in enumerate(ephemeris_emitters):
    # time update
    # predict observables
    sim.correlate()
    # close loop


print("")
