from navsim.simulations.measurement import MeasurementSimulation
from navsim.config import SimulationConfiguration


# Simulation Factory
def get_signal_simulation(simulation_type: str, configuration: SimulationConfiguration):
    SIMULATIONS = {
        "measurement": MeasurementSimulation(configuration=configuration),
    }

    simulation_type = "".join([i for i in simulation_type if i.isalnum()]).casefold()
    simulation = SIMULATIONS.get(
        simulation_type, MeasurementSimulation(configuration=configuration)
    )

    return simulation
