import pyvista as pv
import numpy as np
import pymap3d as pm
import tkinter

from astropy.time import Time
from datetime import datetime
from collections import defaultdict
from pyvista.examples import load_globe_texture, download_cubemap_space_16k
from pyvista.examples.planets import load_earth

from navtools.constants import WGS84_RADIUS


class SatelliteEmitterVisualizer:
    def __init__(
        self, is_point_light=True, off_screen=False, point_size=7.5, window_scale=1
    ) -> None:
        self._off_screen = off_screen
        self._point_size = point_size

        self._light = pv.Light(intensity=1)

        if is_point_light:
            self._pl = pv.Plotter(lighting="none", off_screen=self._off_screen)
            self._pl.add_light(self._light)
        else:
            self._pl = pv.Plotter(off_screen=self._off_screen)

        root = tkinter.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        self._pl.window_size = int(window_scale * float(width)), int(
            window_scale * float(height)
        )

        self._setup_earth_object()
        self._setup_space_cubemap()

        self._constellations = defaultdict()
        self._receivers = defaultdict()

    def _setup_earth_object(self):
        self._earth = load_earth(radius=WGS84_RADIUS)
        earth_texture = load_globe_texture()
        self._axes = pv.Axes(show_actor=True)

        self._earth.translate((0.0, 0.0, 0.0), inplace=True)

        self._pl.add_mesh(self._earth, texture=earth_texture, smooth_shading=True)

    def _setup_space_cubemap(self):
        cubemap = download_cubemap_space_16k()
        self._pl.add_actor(cubemap.to_skybox())
        self._pl.set_environment_texture(cubemap, True)

    def add_constellation_at_epoch(
        self,
        datetime: datetime,
        x: np.array,
        y: np.array,
        z: np.array,
        name: str,
        color: str,
    ):
        eci_pos = np.asarray(pm.ecef2eci(x=x, y=y, z=z, time=datetime)).T
        constellation = pv.PolyData(eci_pos)
        self._pl.add_mesh(
            constellation,
            label=name,
            color=color,
            point_size=self._point_size,
            render_points_as_spheres=True,
        )
        self.add_legend()

        self._constellations[name.lower()] = constellation

    def update_constellation(
        self,
        datetime: datetime,
        x: np.array,
        y: np.array,
        z: np.array,
        name: str,
        color: str = None,
    ):
        if self._constellations.get(name.lower()) is None:
            self.add_constellation_at_epoch(
                datetime=datetime,
                x=x,
                y=y,
                z=z,
                name=name,
                color=color,
            )
            return

        eci_pos = np.asarray(pm.ecef2eci(x=x, y=y, z=z, time=datetime)).T
        self._pl.update_coordinates(
            eci_pos, mesh=self._constellations.get(name.lower()), render=False
        )

    def update_receiver_position(
        self,
        datetime: datetime,
        x: np.array,
        y: np.array,
        z: np.array,
        name: str,
        color: str = None,
    ):
        if self._receivers.get(name.lower()) is None:
            self.add_receiver_position_at_epoch(
                datetime=datetime,
                x=x,
                y=y,
                z=z,
                name=name,
                color=color,
            )
            return

        eci_pos = np.asarray(pm.ecef2eci(x=x, y=y, z=z, time=datetime)).T
        self._pl.update_coordinates(eci_pos, mesh=self._receivers.get(name.lower()))
        new_gst = Time(datetime).sidereal_time("apparent", "greenwich").degree
        dgst = np.abs(self._gst - new_gst)
        self._gst = new_gst
        self._earth.rotate_z(dgst, point=self._axes.origin, inplace=True)

    def add_orbits(
        self,
        x: np.array,
        y: np.array,
        z: np.array,
        datetimes: list,
        color: str,
        name: str = None,
    ):
        orbits = pv.MultiBlock()

        _, n_emitters = x.shape
        for emitter in range(n_emitters):
            eci_positions = np.asarray(
                [
                    pm.ecef2eci(
                        x=x[time_idx, emitter],
                        y=y[time_idx, emitter],
                        z=z[time_idx, emitter],
                        time=time,
                    )
                    for time_idx, time in enumerate(datetimes)
                ]
            )
            orbit = pv.Spline(eci_positions, n_points=10)
            orbits.append(orbit)

        self._pl.add_mesh(orbits.combine(), color=color, label=name)

    def add_receiver_position_at_epoch(
        self,
        datetime: datetime,
        x: np.array,
        y: np.array,
        z: np.array,
        name: str,
        color: str,
    ):
        eci_pos = np.asarray(pm.ecef2eci(x=x, y=y, z=z, time=datetime)).T
        rx = pv.PolyData(eci_pos)
        self._pl.add_mesh(
            rx,
            label=name,
            color=color,
            point_size=self._point_size,
            render_points_as_spheres=True,
        )
        self._gst = Time(datetime).sidereal_time("apparent", "greenwich").degree
        self._earth.rotate_z(180 + self._gst, point=self._axes.origin, inplace=True)

        az = np.degrees(np.arctan2(eci_pos[1], eci_pos[0])) - 45.0
        self._pl.camera.azimuth = az
        self._light.set_direction_angle(az, -20)

        self.add_legend()

        self._receivers[name.lower()] = rx

    def add_legend(self):
        self._pl.add_legend(bcolor=None, face=None, size=(0.15, 0.15))

    def add_text(self, **kwargs):
        self._text = self._pl.add_text(**kwargs)

    def show(self, **kwargs):
        self._pl.show(**kwargs)

    def save_graphic(self, **kwargs):
        self._pl.save_graphic(**kwargs)

    def open_gif(self, **kwargs):
        self._pl.open_gif(**kwargs)

    def write_frame(self, **kwargs):
        self._pl.write_frame(**kwargs)

    def render(self, **kwargs):
        self._pl.render(**kwargs)

    def close(self, **kwargs):
        self._pl.close(**kwargs)
