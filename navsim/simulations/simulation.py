from abc import ABC, abstractmethod
from navsim.config import SimulationConfiguration


# Simulation Objects
class Simulation(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self, configuration: SimulationConfiguration):
        pass

    @abstractmethod
    def simulate(self):
        pass
