import numpy as np
import importlib
import itertools
import warnings

from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timezone
from skyfield.api import Loader
from skyfield.framelib import itrs
from scipy.io import savemat
from datetime import datetime

from laika import AstroDog
from laika.gps_time import GPSTime
from navtools.common import (
    compute_visibility_status,
    compute_range_and_unit_vector,
    compute_range_rate,
)


@dataclass
class SatelliteEmitterState:
    """dataclass that contains states for satellite emitters produced by :class:`navtools.emitters.satellites.SatelliteEmitters`"""

    id: str
    gps_time: GPSTime
    datetime: datetime
    constellation: str
    pos: float
    vel: float
    clock_bias: float
    clock_drift: float
    range: float
    range_rate: float
    az: float
    el: float


class SatelliteEmitters:
    """contains emitters for specified constellations and returns states and observables for given receiver states and time"""

    GNSS = {"gps": "G", "glonass": "R", "galileo": "E", "beidou": "C", "qznss": "J"}
    LEO = {
        "iridium-NEXT": "IRIDIUM",
        "orbcomm": "ORBCOMM",
        "globalstar": "GLOBALSTAR",
        "oneweb": "ONEWEB",
        "starlink": "STARLINK",
    }

    def __init__(self, constellations: list, mask_angle: float = 10.0):
        self._filter_constellations(constellations=constellations)

        if self._laika_constellations:
            laika_literals = self._get_laika_literals()
            self._dog = AstroDog(valid_const=laika_literals)

        if self._skyfield_constellations:
            self._load = Loader("/tmp/leo/", verbose=False)
            self._ts = self._load.timescale()

        self._mask_angle = mask_angle
        self._is_rx_state_unset = True
        self._is_multiple_epoch = False
        self._rx_pos = np.zeros(3)
        self._rx_vel = np.zeros(3)
        self.emitter_ids = set([])

    @property
    def states(self):
        return self._emitter_states

    @property
    def rx_pos(self):
        return self._rx_pos

    @property
    def rx_vel(self):
        return self._rx_vel

    @property
    def ephemerides(self):
        return self._get_ephemerides

    def from_datetime(
        self,
        datetime: datetime,
        rx_pos: np.array,
        rx_vel: np.array = np.zeros(3),
        is_only_visible_emitters: bool = True,
    ) -> dict:
        """computes satellite states and observables for given epoch/time and receiver states

        Parameters
        ----------
        datetime : datetime
            time/epoch associated with current receiver states
        rx_pos : np.array
            receiver position in ECEF reference frame
        rx_vel : np.array, optional
            receiver velocity in ECEF reference frame, by default np.zeros(3)
        is_only_visible_emitters : bool, optional
            determines whether returned emitters are only those in view, by default True

        Returns
        -------
        dict
            emitter states for a particular time/epoch
        """
        emitter_states = {}
        self._time = datetime
        self._gps_time = GPSTime.from_datetime(datetime=datetime)
        self._rx_pos = rx_pos
        self._rx_vel = rx_vel

        if self._laika_constellations:
            laika_states = self._dog.get_all_sat_info(time=self._gps_time)
            emitter_states.update(laika_states)

        if not hasattr(self, "_skyfield_satellites"):
            self._skyfield_satellites = self._get_skyfield_satellites(
                first_datetime=datetime
            )

        if self._skyfield_constellations:
            time = self._ts.from_datetime(
                datetime=datetime.replace(tzinfo=timezone.utc)
            )
            skyfield_states = self._get_single_epoch_skyfield_states(time=time)
            emitter_states.update(skyfield_states)

        self._emitter_states = self._compute_los_states(
            emitter_states=emitter_states,
            is_only_visible_emitters=is_only_visible_emitters,
        )

        return self._emitter_states

    def from_gps_time(
        self,
        gps_time: GPSTime,
        rx_pos: np.array,
        rx_vel: np.array = np.zeros(3),
        is_only_visible_emitters: bool = True,
    ) -> dict:
        """wrapper for :func:`navtools.emitters.satellites.SatelliteEmitters.from_datetime` where `datetime` parameter is `GPSTime`

        Parameters
        ----------
        gps_time : GPSTime
            time/epoch associated with current receiver states
        rx_pos : np.array
            receiver position in ECEF reference frame
        rx_vel : np.array, optional
            receiver velocity in ECEF reference frame, by default np.zeros(3)
        is_only_visible_emitters : bool, optional
            determines whether returned emitters are only those in view, by default True

        Returns
        -------
        dict
            emitter states for a particular time/epoch
        """
        datetime = gps_time.as_datetime()
        emitter_states = self.from_datetime(
            datetime=datetime,
            rx_pos=rx_pos,
            rx_vel=rx_vel,
            is_only_visible_emitters=is_only_visible_emitters,
        )

        return emitter_states

    def from_datetimes(
        self,
        datetimes: list[datetime],
        rx_pos: np.array,
        rx_vel: np.array = None,
        is_only_visible_emitters: bool = True,
    ) -> list[dict]:
        """computes satellite states and observables for list of given epochs/times and receiver states for each epoch/time

        Parameters
        ----------
        datetimes : list[datetime]
            times associated with receiver states over duration
        rx_pos : np.array
            receiver positions in ECEF reference frame over duration of time
        rx_vel : np.array, optional
            receiver velocities in ECEF reference frame over duration of time, by default None
        is_only_visible_emitters : bool, optional
            determines whether returned emitters are only those in view, by default True

        Returns
        -------
        list[dict]
            emitter states for duration of time
        """
        laika_duration_states = []
        skyfield_duration_states = []

        gps_times = [GPSTime.from_datetime(datetime=datetime) for datetime in datetimes]

        if rx_pos.size == 3:
            num_epochs = len(datetimes)
            rx_pos = np.tile(
                rx_pos, (num_epochs, 1)
            )  # needed to iterate with states over time
            rx_vel = np.zeros_like(rx_pos)

        if self._laika_constellations:
            laika_desc = f"extracting {self._laika_string} emitter states"
            laika_duration_states = [
                self._dog.get_all_sat_info(time=gps_time)
                for gps_time in tqdm(gps_times, desc=laika_desc)
            ]

        if self._skyfield_constellations:
            self._skyfield_satellites = self._get_skyfield_satellites(
                first_datetime=datetimes[0]
            )
            utc_datetimes = (
                datetime.replace(tzinfo=timezone.utc) for datetime in datetimes
            )
            times = self._ts.from_datetimes(datetime_list=utc_datetimes)
            skyfield_duration_states = self._get_multiple_epoch_skyfield_states(
                times=times
            )

        if laika_duration_states and skyfield_duration_states:
            emitter_duration_states = [
                {**laika_epoch, **skyfield_epoch}
                for (laika_epoch, skyfield_epoch) in zip(
                    laika_duration_states, skyfield_duration_states
                )
            ]
        elif laika_duration_states:
            emitter_duration_states = laika_duration_states
        elif skyfield_duration_states:
            emitter_duration_states = skyfield_duration_states

        self._emitter_states = []
        for datetime, states, pos, vel in tqdm(
            zip(datetimes, emitter_duration_states, rx_pos, rx_vel),
            desc="computing line-of-sight states",
            total=len(datetimes),
        ):
            self._time = datetime
            self._gps_time = GPSTime.from_datetime(datetime=datetime)
            self._rx_pos = pos
            self._rx_vel = vel
            self._emitter_states.append(
                self._compute_los_states(
                    emitter_states=states,
                    is_only_visible_emitters=is_only_visible_emitters,
                )
            )

        return self._emitter_states

    # TODO: add from_gps_times method

    def savemat(self, file_name: str, **kwargs):
        print(f"saving emitter states to {file_name}...")

        if isinstance(self._emitter_states, list):
            for idx, epoch in enumerate(self._emitter_states):
                self._emitter_states[idx] = self._format_mat_data(epoch)

        else:
            self._emitter_states = self._format_mat_data(self._emitter_states)

        savemat(
            file_name=file_name,
            mdict={"emitters": self._emitter_states},
            do_compression=True,
            **kwargs,
        )

        print(f"{file_name} has been saved!")

    @staticmethod
    def _format_mat_data(epoch):
        new_epoch = {}
        for emitter_name, emitter_state in epoch.items():
            if not emitter_name.isalnum():
                new_key = "".join(filter(str.isalnum, emitter_name))
            else:
                new_key = emitter_name

            new_epoch[new_key] = epoch[emitter_name]
            new_epoch[new_key].datetime = emitter_state.datetime.strftime(
                format="%Y-%m-%d %H:%M:%S"
            )

        return new_epoch

    def _compute_los_states(self, emitter_states: dict, is_only_visible_emitters: bool):
        emitters = defaultdict()

        for emitter_id, emitter_state in emitter_states.items():
            emitter_pos = emitter_state[0]
            emitter_vel = emitter_state[1]
            emitter_clock_bias = emitter_state[2]
            emitter_clock_drift = emitter_state[3]

            is_visible, emitter_az, emitter_el = compute_visibility_status(
                rx_pos=self._rx_pos,
                emitter_pos=emitter_pos,
                mask_angle=self._mask_angle,
            )

            if is_only_visible_emitters and not is_visible:
                continue

            symbol = "".join([i for i in emitter_id if i.isalpha()])
            try:  # removes specialty satellites in constellation
                if symbol in self._laika_constellations.values():
                    symbol_index = list(self._laika_constellations.values()).index(
                        symbol
                    )
                    constellation = list(self._laika_constellations.keys())[
                        symbol_index
                    ]
                else:
                    symbol_index = list(self._skyfield_constellations.values()).index(
                        symbol
                    )
                    constellation = list(self._skyfield_constellations.keys())[
                        symbol_index
                    ]
            except:
                continue

            range, unit_vector = compute_range_and_unit_vector(
                rx_pos=self._rx_pos, emitter_pos=emitter_pos
            )
            range_rate = compute_range_rate(
                rx_vel=self._rx_vel, emitter_vel=emitter_vel, unit_vector=unit_vector
            )

            emitter_state = SatelliteEmitterState(
                id=emitter_id,
                gps_time=self._gps_time,
                datetime=self._time,
                constellation=constellation,
                pos=emitter_pos,
                vel=emitter_vel,
                clock_bias=emitter_clock_bias,
                clock_drift=emitter_clock_drift,
                range=range,
                range_rate=range_rate,
                az=emitter_az,
                el=emitter_el,
            )
            self.emitter_ids.add(emitter_id)
            emitters[emitter_id] = emitter_state

        return emitters

    def _get_ephemerides(self):
        # assumes signal simulation isn't longer than 4 hours, therefore time input is last time of simulation
        ephemerides = defaultdict()

        for emitter in self.emitter_ids:
            emitter_constellation = "".join([i for i in emitter if i.isalpha()])
            if emitter_constellation in SatelliteEmitters.GNSS.values():
                eph = self._dog.get_nav(prn=emitter, time=self._gps_time)
            else:
                for satellite in self._skyfield_satellites:
                    if satellite.name == emitter:
                        eph = satellite

            ephemerides[emitter] = eph

        return ephemerides

    def _filter_constellations(self, constellations: list):
        if isinstance(constellations, str):
            constellations = constellations.split()

        self._laika_constellations = {
            gnss: symbol
            for gnss, symbol in SatelliteEmitters.GNSS.items()
            for constellation in constellations
            if constellation.casefold() == gnss.casefold()
        }
        self._skyfield_constellations = {
            leo: symbol
            for leo, symbol in SatelliteEmitters.LEO.items()
            for constellation in constellations
            if constellation.casefold() == leo.casefold()
        }

        if self._laika_constellations:
            self._laika_string = ", ".join(
                [const for const in self._laika_constellations]
            )

        if self._skyfield_constellations:
            self._skyfield_string = ", ".join(
                [const for const in self._skyfield_constellations]
            )

    def _get_laika_literals(self):
        constellations = self._laika_constellations
        literals = []

        module = importlib.import_module("laika.helpers")
        obj = getattr(module, "ConstellationId")

        if isinstance(constellations, str):
            constellations = constellations.split()

        for constellation in constellations:
            literal = getattr(obj, constellation.upper())
            literals.append(literal)

        return literals

    def _get_skyfield_satellites(self, first_datetime):
        FIRST_CELESTRAK_REPO_DATETIME = datetime(2023, 8, 11)

        constellations = self._skyfield_constellations

        if first_datetime >= FIRST_CELESTRAK_REPO_DATETIME:
            year = first_datetime.timetuple().tm_year
            day = first_datetime.timetuple().tm_yday
            urls = [
                f"https://raw.githubusercontent.com/tannerkoza/celestrak-orbital-data/main/{constellation}/{year}/{day}/{constellation}.tle"
                for constellation in constellations
            ]
        else:
            warnings.warn(
                "datetimes preceed earliest TLE in database, therfore, orbits may be invalid. Using current TLE from celestrak. This will be addressed in the future."
            )
            urls = [
                f"https://celestrak.org/NORAD/elements/gp.php?GROUP={constellation}&FORMAT=tle"
                for constellation in constellations
            ]

        satellites = [
            self._load.tle_file(
                url=url,
                reload=True,
            )
            for url in urls
        ]

        satellites = list(itertools.chain(*satellites))  # flatten list

        return satellites

    def _get_single_epoch_skyfield_states(self, time):
        emitters = defaultdict()

        for emitter in self._skyfield_satellites:
            emitter_state = emitter.at(time)
            state = [emitter_state.xyz.m, emitter_state.velocity.m_per_s, 0, 0]
            emitters[emitter.name] = state

        return emitters

    def _get_multiple_epoch_skyfield_states(self, times):
        emitters = []
        skyfield_ex_desc = f"extracting {self._skyfield_string} emitter states"

        ecef_emitters = [
            (emitter.name, emitter.at(times).frame_xyz_and_velocity(itrs))
            for emitter in self._skyfield_satellites
        ]

        n_times = range(len(times))
        emitters = [
            self._extract_skyfield_states(
                ecef_emitters=ecef_emitters,
                epoch=epoch,
            )
            for epoch in tqdm(n_times, desc=skyfield_ex_desc)
        ]

        return emitters

    @staticmethod
    def _extract_skyfield_states(ecef_emitters: list, epoch: int):
        emitters_epoch = defaultdict()

        for emitter_name, emitter_state in ecef_emitters:
            pos = np.array(emitter_state[0].m[:, epoch])
            vel = np.array(emitter_state[1].m_per_s[:, epoch])
            state = [pos, vel, 0.0, 0.0]
            emitters_epoch[emitter_name] = state

        return emitters_epoch
