import dacite as dc
from dataclasses import dataclass, fields
from collections import defaultdict
from laika.ephemeris import Ephemeris


@dataclass(frozen=True)
class GPSClock:
    t_gd: float
    iodc: float
    t_oc: float
    a_f2: float
    a_f1: float
    a_f0: float


@dataclass(frozen=True)
class GPSEphemerides:
    m0: float
    delta_n: float
    ecc: float
    a: float
    omega0: float
    i0: float
    omega: float
    omega_dot: float
    i_dot: float
    c_uc: float
    c_us: float
    c_rc: float
    c_rs: float
    c_ic: float
    c_is: float
    t_oe: float
    iode: float


@dataclass(frozen=True)
class GPSIonosphere:
    a0: float
    a1: float
    a2: float
    a3: float
    b0: float
    b1: float
    b2: float
    b3: float


GPS_DATACLASSES = [GPSClock, GPSEphemerides]

DATACLASSES = {"G": GPS_DATACLASSES}


def package_laika_data(constellation_symbol: str, data: Ephemeris):
    data = data.to_dict()
    dataclasses = DATACLASSES.get(constellation_symbol)

    packaged_data = defaultdict()
    for dataclass in dataclasses:
        data_to_package = defaultdict()
        field_names = [field.name for field in fields(dataclass)]

        for key, value in data.items():
            base_key = remove_string_symbols(key)

            for field in field_names:
                if base_key == remove_string_symbols(field):
                    data_to_package[field] = value
                    break

        packaged_dataclass = dc.from_dict(data_class=dataclass, data=data_to_package)
        dataclass_name = packaged_dataclass.__class__.__name__
        packaged_data[dataclass_name] = packaged_dataclass

    return packaged_data


def remove_string_symbols(string: str):
    return "".join([i for i in string if i.isalnum()]).casefold()
