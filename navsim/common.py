from navsim.simulations.measurement import MeasurementSimulation


# Simulation Factory
def get_simulation(simulation_type: str):
    SIMULATIONS = {
        "measurement": MeasurementSimulation(),
    }

    simulation_type = "".join([i for i in simulation_type if i.isalnum()]).casefold()
    simulation = SIMULATIONS.get(simulation_type, MeasurementSimulation())

    return simulation
