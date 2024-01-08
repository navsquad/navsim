from navsim.simulations.measurement import MeasurementSimulation
from navsim.configuration import SimulationConfiguration


# Simulation Factory
def get_signal_simulation(
    simulation_type: str,
    configuration: SimulationConfiguration,
    disable_progress: bool = False,
):
    SIMULATIONS = {
        "measurement": MeasurementSimulation(
            configuration=configuration, disable_progress=disable_progress
        ),
    }

    simulation_type = "".join([i for i in simulation_type if i.isalnum()]).casefold()
    simulation = SIMULATIONS.get(
        simulation_type,
        MeasurementSimulation(
            configuration=configuration, disable_progress=disable_progress
        ),
    )

    return simulation
