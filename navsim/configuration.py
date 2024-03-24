import os
import yaml
import inspect
import dacite as dc

from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog as fd
from typing import Callable

from navtools.signals.signals import SatelliteSignal
from navtools import get_signal_properties
from navtools.signals.signals import bpsk_correlator
from navsim.exceptions import (
    InvalidConfigurationFormatting,
    EmptyRequiredConfigurationField,
)

try:
    is_readline_available = True
    import readline as rl
except ImportError:
    is_readline_available = False


# Configuration Data Classes
@dataclass(frozen=True)
class TimeConfiguration:
    duration: float
    fsim: float
    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: int = 0


@dataclass(frozen=True)
class ConstellationsConfiguration:
    emitters: dict
    mask_angle: float = 10.0


@dataclass(frozen=True)
class ErrorConfiguration:
    ionosphere: str | None = None
    troposphere: str | None = None
    rx_clock: str | None = None
    pseudorange_awgn_sigma: float = 0.0
    carr_psr_awgn_sigma: float = 0.0
    pseudorange_rate_awgn_sigma: bool = False


@dataclass
class SignalConfiguration:
    signal: str
    js: float = 0.0
    correlator_model: str = "bpsk"

    def __post_init__(self):
        object.__setattr__(
            self, "properties", get_signal_properties(signal_name=self.signal)
        )


@dataclass(frozen=True)
class SimulationConfiguration:
    time: TimeConfiguration
    constellations: ConstellationsConfiguration
    errors: ErrorConfiguration


# Configuration Creation
def get_configuration(configuration_path: str) -> SimulationConfiguration:
    prompt_string = "[navsim] select a simulation configuration: "
    if is_readline_available:
        config_file_name = select_file(
            directory_path=configuration_path,
            prompt_string=prompt_string,
        )
        config_file_path = configuration_path / config_file_name
    else:
        filetypes = (("yaml", "*.yaml"), ("yaml", "*.yml*"))
        config_file_path = fd.askopenfilename(
            initialdir=configuration_path,
            title=prompt_string,
            filetypes=filetypes,
        )

    try:
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

    except Exception as exc:
        raise InvalidConfigurationFormatting(config_name=config_file_name) from exc

    try:
        # time
        time = dc.from_dict(data_class=TimeConfiguration, data=config.get("time"))

        # constellations
        constellations = config.get("constellations")
        constellations["emitters"] = {
            constellation: dc.from_dict(data_class=SignalConfiguration, data=data)
            for constellation, data in constellations.get("emitters").items()
        }
        constellations = dc.from_dict(
            data_class=ConstellationsConfiguration, data=constellations
        )

        # errors
        if "errors" not in config.keys():
            config["errors"] = {}  # handles case when no errors section in config
        errors = dc.from_dict(data_class=ErrorConfiguration, data=config.get("errors"))

        return SimulationConfiguration(
            time=time,
            constellations=constellations,
            errors=errors,
        )

    except dc.exceptions.WrongTypeError as exc:
        last_frame = inspect.trace()[-1]
        class_type = last_frame.frame.f_locals.get("data_class")

        raise EmptyRequiredConfigurationField(
            class_type=class_type, field_name=exc.field_path
        ) from exc


def select_file(directory_path: str | Path, prompt_string="select a file: "):
    if type(directory_path) is str:
        directory_path = Path(directory_path)

    os.chdir(directory_path)
    rl.set_completer_delims(" \t\n=")
    rl.parse_and_bind("tab: complete")

    file_name = input(prompt_string)

    return file_name
