import navsim as ns
import numpy as np

from navsim.paths import CONFIG_PATH

rx_pos = np.array([423756, -5361363, 3417705])

configuration = ns.get_configuration(configuration_path=CONFIG_PATH)

sim = ns.get_simulation(simulation_type=configuration.type)
sim.initialize(configuration=configuration)
sim.simulate(rx_pos=rx_pos)

sim.to_hdf(output_dir_path="../data/")
sim.to_mat(output_dir_path="../data/")
