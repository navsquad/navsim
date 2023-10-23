import yaml
import dacite as dc
import inspect
from dataclasses import dataclass

from navtools.io import tab_complete_input
from navsim.exceptions import (
    InvalidConfigurationFormatting,
    EmptyRequiredConfigurationField,
)


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
    ionosphere: str = "klobuchar"
    troposphere: str = "saastamoinen"
    rx_clock: str = "low_quality_tcxo"
    js: float = 0.0


@dataclass(frozen=True)
class SimulationConfiguration:
    type: str
    time: TimeConfiguration
    constellations: ConstellationsConfiguration
    errors: ErrorConfiguration


# Configuration Creation
def get_configuration(configuration_path: str) -> SimulationConfiguration:
    config_file_name = tab_complete_input(
        directory_path=configuration_path,
        prompt_string="[navsim] select a simulation configuration: ",
    )
    config_file_path = configuration_path / config_file_name

    try:
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

    except Exception as exc:
        raise InvalidConfigurationFormatting(config_name=config_file_name) from exc

    try:
        type = config.get("type")
        time = dc.from_dict(data_class=TimeConfiguration, data=config.get("time"))
        constellations = dc.from_dict(
            data_class=ConstellationsConfiguration, data=config.get("constellations")
        )
        errors = dc.from_dict(data_class=ErrorConfiguration, data=config.get("errors"))

        return SimulationConfiguration(
            type=type,
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
