import importlib
import numpy as np
import navtools as nt

from numpy import sin, cos
from laika import AstroDog
from laika.gps_time import GPSTime
from itertools import chain
from skyfield.api import Loader
from skyfield.framelib import itrs
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from navsim.exceptions import InvalidDatetimeLEO


@dataclass(repr=False)
class SatelliteEmitterState:
    # time
    datetime: datetime
    gps_time: GPSTime

    # state
    pos: float
    vel: float
    clock_bias: float
    clock_drift: float
    ephemeris: object

    # line-of-sight
    range: float
    range_rate: float
    az: float
    el: float


class SatelliteEmitters:
    GNSS_SYSTEMS = {
        "gps": "G",
        "glonass": "R",
        "galileo": "E",
        "beidou": "C",
        "qznss": "J",
    }
    LEO_SYSTEMS = {
        "iridium-NEXT": "IRIDIUM",
        "orbcomm": "ORBCOMM",
        "globalstar": "GLOBALSTAR",
        "oneweb": "ONEWEB",
        "starlink": "STARLINK",
    }

    def __init__(
        self, systems: list, mask_angle: float = 10.0, disable_progress: bool = False
    ) -> None:
        self.__sort_systems(systems=systems)

        if self.__laika_systems:
            systems = self.__get_laika_system_names()
            # allows laika to accept SatelliteEmitters systems
            self.__dog = AstroDog(valid_const=systems)

        if self.__sfield_systems:
            self.__load = Loader("/tmp/leo/", verbose=False)
            self.__ts = self.__load.timescale()

            self.__is_sfield_emitters = False

        self.__mask_angle = mask_angle
        self.__disable_progress = disable_progress

        # logging
        self.dop = []

    # public
    def from_datetime(
        self,
        datetime: datetime,
        rx_pos: np.ndarray,
        rx_vel: np.ndarray = np.zeros(3),
        only_visible: bool = True,
    ) -> dict:
        self.datetime = datetime
        self.gps_time = GPSTime.from_datetime(datetime=datetime)
        self.rx_pos = rx_pos
        self.rx_vel = rx_vel

        emitters = {}
        if self.__laika_systems:
            states = self.__dog.get_all_sat_info(time=self.gps_time)
            emitters.update(states)

        if self.__sfield_systems:
            if not self.__is_sfield_emitters:
                self.__sfield_emitters = self.__get_sfield_emitters(
                    initial_datetime=datetime
                )

                self.__is_sfield_emitters = True

            states = self.__get_epoch_sfield_states(datetime=datetime)
            emitters.update(states)

        emitter_states = self.__compute_los_states(
            emitters=emitters, only_visible=only_visible
        )

        return emitter_states

    # private
    def __compute_los_states(self, emitters: dict, only_visible: bool):
        states = defaultdict()
        unit_vectors = []

        for emitter, state in emitters.items():
            system = self.__get_emitter_system(name=emitter)
            pos, vel, cb, cd, eph = state

            visible, az, el = nt.compute_visibility_status(
                rx_pos=self.rx_pos,
                emitter_pos=pos,
                mask_angle=self.__mask_angle,
            )

            if only_visible and not visible:
                continue

            range, unit_vector = nt.compute_range_and_unit_vector(
                rx_pos=self.rx_pos, emitter_pos=pos
            )
            range_rate = nt.compute_range_rate(
                rx_vel=self.rx_vel, emitter_vel=vel, unit_vector=unit_vector
            )

            state = SatelliteEmitterState(
                datetime=self.datetime,
                gps_time=self.gps_time,
                pos=pos,
                vel=vel,
                clock_bias=cb,
                clock_drift=cd,
                ephemeris=eph,
                range=range,
                range_rate=range_rate,
                az=az,
                el=el,
            )
            states[(system, emitter)] = state
            unit_vectors.append(unit_vector)

        self.__compute_dop(unit_vectors=unit_vectors, nemitters=len(unit_vectors))

        return states

    def __compute_dop(self, unit_vectors: list, nemitters: int):
        unit_vectors = np.array(unit_vectors)

        lla = nt.ecef2lla(self.rx_pos[0], self.rx_pos[1], self.rx_pos[2])
        phi = np.radians(lla.lat)
        lam = np.radians(lla.lon)

        R = np.array(
            [
                [-sin(lam), cos(lam), np.zeros_like(lam)],
                [-cos(lam) * sin(phi), -sin(lam) * sin(phi), cos(phi)],
                [cos(lam) * cos(phi), sin(lam) * cos(phi), sin(phi)],
            ]
        )
        H = np.append(-unit_vectors, np.ones(nemitters)[..., None], axis=1)

        try:
            dop = R @ np.linalg.inv(H.T @ H) @ R.T
        except:
            dop = np.zeros_like(H.T @ H)

        self.dop.append(dop)

    def __sort_systems(self, systems: list):
        if isinstance(systems, str):
            systems = systems.split()

        gnss_systems = SatelliteEmitters.GNSS_SYSTEMS
        leo_systems = SatelliteEmitters.LEO_SYSTEMS

        # filters systems by API
        self.__laika_systems = {
            gnss: symbol
            for gnss, symbol in gnss_systems.items()
            for sys in systems
            if sys.casefold() == gnss.casefold()
        }
        self.__sfield_systems = {
            leo: symbol
            for leo, symbol in leo_systems.items()
            for sys in systems
            if sys.casefold() == leo.casefold()
        }

    def __get_laika_system_names(self):
        module = importlib.import_module("laika.helpers")
        obj = getattr(module, "ConstellationId")

        names = []
        for sys in self.__laika_systems:
            name = getattr(obj, sys.upper())
            names.append(name)

        return names

    def __get_sfield_emitters(self, initial_datetime: datetime):
        EARLIEST_DATETIME = datetime(2023, 8, 11)
        URL_BASE_PATH = (
            "https://raw.githubusercontent.com/tannerkoza/celestrak-orbital-data/main/"
        )

        if initial_datetime >= EARLIEST_DATETIME:
            year = initial_datetime.timetuple().tm_year
            day = "%03d" % initial_datetime.timetuple().tm_yday
            # enforces repo naming convention

            system_urls = [
                URL_BASE_PATH + f"{sys}/{year}/{day}/{sys}.tle"
                for sys in self.__sfield_systems
            ]

        else:
            raise InvalidDatetimeLEO(datetime_str=EARLIEST_DATETIME.strftime())

        total_emitters = [
            self.__load.tle_file(
                url=url,
                reload=True,
            )
            for url in system_urls
        ]

        total_emitters = list(chain.from_iterable(total_emitters))
        # flatten to merge multiple systems

        emitters = [
            emitter
            for emitter in total_emitters
            if self.__is_valid_emitter(name=emitter.name)
        ]

        return emitters

    def __get_epoch_sfield_states(self, datetime: datetime):
        time = self.__ts.from_datetime(datetime=datetime.replace(tzinfo=timezone.utc))

        states = defaultdict()
        for emitter in self.__sfield_emitters:
            pos, vel = emitter.at(time).frame_xyz_and_velocity(itrs)
            state = [
                pos.m,
                vel.m_per_s,
                0.0,
                0.0,
                emitter.model,
            ]
            states[emitter.name] = state

        return dict(states)

    def __is_valid_emitter(self, name: str):
        gnss_systems = SatelliteEmitters.GNSS_SYSTEMS
        leo_systems = SatelliteEmitters.LEO_SYSTEMS

        symbol = self.__get_emitter_symbol(name=name)

        if symbol not in gnss_systems.values() and symbol not in leo_systems.values():
            return False

        if "[-]" in name:
            return False
        else:
            return True

    def __get_emitter_system(self, name: str):
        systems = {**SatelliteEmitters.GNSS_SYSTEMS, **SatelliteEmitters.LEO_SYSTEMS}

        symbol = self.__get_emitter_symbol(name=name)

        index = list(systems.values()).index(symbol)
        system = list(systems.keys())[index]

        return system

    @staticmethod
    def __get_emitter_symbol(name: str):
        symbol_chars = []

        for char in name:
            if not char.isalpha():
                break
            symbol_chars.append(char)

        symbol = "".join(symbol_chars)

        return symbol
