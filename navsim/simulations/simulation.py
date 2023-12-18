from abc import ABC, abstractmethod
from navsim.config import SimulationConfiguration


# Simulation Objects
class SignalSimulation(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def simulate(self):
        pass
