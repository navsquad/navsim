
import numpy as np
from dataclasses import dataclass
from navtools.constants import GRAVITY, G2T, FT2M

D2R = np.pi / 180
z = np.zeros(3)

@dataclass(frozen=True)
class IMU:
  """dataclass of typical IMU parameters"""
  
  freq: float
  vrw: np.ndarray
  arw: np.ndarray
  vrrw: np.ndarray
  arrw: np.ndarray
  ab_sta: np.ndarray
  gb_sta: np.ndarray
  ab_dyn: np.ndarray
  gb_dyn: np.ndarray
  ab_corr: np.ndarray
  gb_corr: np.ndarray
  mag_psd: np.ndarray
  mag_std: np.ndarray
  ab_std: np.ndarray
  gb_std: np.ndarray
  ab_psd: np.ndarray
  gb_psd: np.ndarray


def fix_imu_si_errors(imu: IMU) -> IMU:
  """Corrects IMU errors from spec-sheets into SI units

  Parameters
  ----------
  imu : IMU
      IMU object with the following fields
        vrw       3x1     velocity random walks [m/s/root(hour)]
        arw       3x1     angle random walks [deg/root(hour)]
        vrrw      3x1     velocity rate random walks [m/s/root(hour)/s]
        arrw      3x1     angle rate random walks [deg/root(hour)/s]
        ab_sta    3x1     accel static biases [mg]
        gb_sta    3x1     gyro static biases [deg/s]
        ab_dyn    3x1     accel dynamic biases [mg]
        gb_dyn    3x1     gyro dynamic biases[deg/s]
        ab_corr   3x1     accel correlation times [s]
        gb_corr   3x1     gyro correlation times [s]
        mag_psd   3x1     magnetometer noise density [mgauss/root(Hz)]
        freq      float   IMU frequency [Hz]

  Returns
  -------
  IMU
      IMU object with the following fields
        vrw       3x1     velocity random walks [m/s^2/root(Hz)]
        arw       3x1     angle random walks [rad/s/root(Hz)]
        vrrw      3x1     velocity rate random walks [m/s^3/root(Hz)]
        arrw      3x1     angle rate random walks [rad/s^2/root(Hz)]
        ab_sta    3x1     accel static biases [m/s^2]
        gb_sta    3x1     gyro static biases [rad/s]
        ab_dyn    3x1     accel dynamic biases [m/s^2]
        gb_dyn    3x1     gyro dynamic biases[rad/s]
        ab_corr   3x1     accel correlation times [s]
        gb_corr   3x1     gyro correlation times [s]
        ab_psd    3x1     acc dynamic bias root-PSD [m/s^2/root(Hz)]
        gb_psd    3x1     gyro dynamic bias root-PSD [rad/s/root(Hz)]
        mag_psd   3x1     magnetometer noise density [tesla]
        freq      double  IMU frequency [Hz]
  """
  # root-PSD noise
  vrw = (imu.vrw / 60)                    # m/s/root(hour) -> m/s^2/root(Hz)
  arw = (imu.arw / 60) * D2R;             # deg/root(hour) -> rad/s/root(Hz)

  # root-PSD rate noise
  vrrw = (imu.vrrw / 60) 
  arrw = (imu.arrw / 60) * D2R;           # deg/root(hour) -> rad/s/root(Hz)

  # Dynamic bias
  ab_dyn = imu.ab_dyn * 0.001 * GRAVITY;  # mg -> m/s^2
  gb_dyn = imu.gb_dyn * D2R;              # deg/s -> rad/s;

  # Correlation time
  ab_corr = imu.ab_corr
  gb_corr = imu.gb_corr

  # Dynamic bias root-PSD
  if (np.any(np.isinf(imu.ab_corr))):
      ab_psd = ab_dyn                         # m/s^2 (approximation)
  else:
      ab_psd = ab_dyn / np.sqrt(imu.ab_corr)  # m/s^2/root(Hz)

  if (np.any(np.isinf(imu.gb_corr))):
      gb_psd = gb_dyn                         # rad/s (approximation)
  else:
      gb_psd = gb_dyn / np.sqrt(imu.gb_corr)  # rad/s/root(Hz)

  # time 
  dt = 1.0 / imu.freq

  # Static bias
  ab_sta = imu.ab_sta * 0.001 * GRAVITY;  # mg -> m/s^2
  gb_sta = imu.gb_sta * D2R;              # deg/s -> rad/s

  # Standard deviation
  ab_std = vrw / np.sqrt(dt);             # m/s^2/root(Hz) -> m/s^2
  gb_std = arw / np.sqrt(dt);             # rad/s/root(Hz) -> rad/s

  # MAG
  mag_std = (imu.mag_psd * 1e-3) / np.sqrt(dt) * G2T # mGauss/root(Hz) -> Tesla
  
  return IMU(
          freq = imu.freq,
          vrw = vrw,
          arw = arw, 
          vrrw = vrrw,
          arrw = arrw, 
          ab_sta = ab_sta, 
          gb_sta = gb_sta,
          ab_dyn = ab_dyn,
          gb_dyn = gb_dyn,
          ab_corr = ab_corr, 
          gb_corr = gb_corr, 
          mag_psd = imu.mag_psd, 
          mag_std = mag_std, 
          ab_std = ab_std,
          gb_std = gb_std,
          ab_psd = ab_psd, 
          gb_psd = gb_psd, 
         )
  
def get_imu_allan_variance_values(imu_name: str) -> IMU:
    """factory function that retrieves requested imu Allan variance values

    Parameters
    ----------
    imu_name : str
        name of clock

    Returns
    -------
    IMU
        imu Allan variance values
    """
    IMUS = {
        "tactical": TACTICAL,
        "industrial": INDUSTRIAL,
    }

    imu_name = "".join([i for i in imu_name if i.isalnum()]).casefold()
    return IMUS.get(imu_name.casefold(), INDUSTRIAL)  # defaults to industrial
  
  
#* === Default IMUs ===
# Tactical Grade (Honeywell HG1700)
# https://aerospace.honeywell.com/content/dam/aerobt/en/documents/landing-pages/brochures/N61-1619-000-001-HG1700InertialMeasurementUnit-bro.pdf
TACTICAL = fix_imu_si_errors(IMU(
            freq = 100.0,
            vrw = np.array([0.65, 0.65, 0.65]) * FT2M,   # m/s/root(hr)
            arw = np.array([0.125, 0.125, 0.125]),       # deg/root(hr)
            vrrw = np.array([0.0, 0.0, 0.0]),            # m/s/root(hr)/s
            arrw = np.array([0.0, 0.0, 0.0]),            # deg/root(hr)/s
            ab_sta = np.array([0.0, 0.0, 0.0]),          # mg
            gb_sta = np.array([0.0, 0.0, 0.0]),          # deg/hr
            ab_dyn = np.array([0.58, 0.58, 0.58]),       # mg
            gb_dyn = np.array([0.017, 0.017, 0.017]),    # deg/hr
            ab_corr = np.array([100.0, 100.0, 100.0]),   # s
            gb_corr = np.array([100.0, 100.0, 100.0]),   # s
            mag_psd = np.array([0.0, 0.0, 0.0]),         # mGauss/root(Hz)
            mag_std = z, 
            ab_std = z, 
            gb_std = z, 
            ab_psd = z, 
            gb_psd = z,
          ))

# Industrial Grade (VectorNav VN100)
# https://www.vectornav.com/docs/default-source/datasheets/vn-100-datasheet-rev2.pdf?sfvrsn=8e35fd12_10
# not sure why the accelerometer seems better for the vectornav?
INDUSTRIAL = fix_imu_si_errors(IMU(
              freq = 100.0,
              vrw = np.array([0.14, 0.14, 0.14]) * 0.001 * GRAVITY * 60,  # m/s/root(hr)
              arw = np.array([3.5e-3, 3.5e-3, 3.5e-3]) * 60,              # deg/root(hr)
              vrrw = np.array([0.0, 0.0, 0.0]),                           # m/s/root(hr)/s
              arrw = np.array([0.0, 0.0, 0.0]),                           # deg/root(hr)/s
              ab_sta = np.array([0.0, 0.0, 0.0]),                         # mg
              gb_sta = np.array([0.0, 0.0, 0.0]),                         # deg/hr
              ab_dyn = np.array([0.04, 0.04, 0.04]),                      # mg
              gb_dyn = np.array([5.0, 5.0, 5.0]),                         # deg/hr
              ab_corr = np.array([100.0, 100.0, 100.0]),                  # s
              gb_corr = np.array([100.0, 100.0, 100.0]),                  # s
              mag_psd = np.array([140.0, 140.0, 140.0]),                  # mGauss/root(Hz)
              mag_std = z, 
              ab_std = z, 
              gb_std = z, 
              ab_psd = z, 
              gb_psd = z,
            ))