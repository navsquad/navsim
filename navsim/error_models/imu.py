
import numpy as np
from dataclasses import dataclass, field
from navtools.constants import GRAVITY, G2T, FT2M

D2R = np.pi / 180
z = np.zeros(3)


@dataclass
class IMU:
  """dataclass of typical IMU allan variance parameters"""
  
  f: float                                      # sampling frequency Hz
  B_acc: np.ndarray = field(default_factory=z)  # accelerometer bias instability coefficients [(m/s)/s]
  B_gyr: np.ndarray = field(default_factory=z)  # gyroscope bias instability coefficients [rad/s]
  K_acc: np.ndarray = field(default_factory=z)  # accelerometer acceleration random walk coefficients [(m/s)/(s*sqrt(s))]
  K_gyr: np.ndarray = field(default_factory=z)  # gyroscope rate random walk coefficients [rad/(s*sqrt(s))]
  N_acc: np.ndarray = field(default_factory=z)  # accelerometer velocity random coefficients [(m/s)/sqrt(s)]
  N_gyr: np.ndarray = field(default_factory=z)  # gyroscope angle random walk coefficients [rad/sqrt(s)]
  Tc_acc: np.ndarray = field(default_factory=z) # accelerometer correlation times [s]
  Tc_gyr: np.ndarray = field(default_factory=z) # gyroscope correlation times [s]


# TODO: common for B_acc to be in units of [m/s/hr] -> should I account for this?
# TODO: also for N_acc, common units are [m/s/sqrt(hr)]
def fix_imu_si_errors(imu: IMU) -> IMU:
  """Corrects IMU errors from spec-sheets into SI units

  Parameters
  ----------
  imu : IMU
      IMU object with the following fields:
        f       float   sampling frequency Hz
        B_acc   3x1     accelerometer bias instability coefficients [mg] 
        B_gyr   3x1     gyroscope bias instability coefficients [deg/hr]
        K_acc   3x1     accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
        K_gyr   3x1     gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
        N_acc   3x1     accelerometer velocity random walk coefficients [m/s/sqrt(hr)]
        N_gyr   3x1     gyroscope angle random walk coefficients [deg/sqrt(hr)]
        Tc_acc  3x1     accelerometer correlation times [s]
        Tc_gyr  3x1     gyroscope correlation times [s]

  Returns
  -------
  IMU
      IMU object with the following fields:
        f       float   sampling frequency Hz
        B_acc   3x1     accelerometer bias instability coefficients [(m/s)/s]
        B_gyr   3x1     gyroscope bias instability coefficients [rad/s]
        K_acc   3x1     accelerometer acceleration random walk coefficients [(m/s)/(s*sqrt(s))]
        K_gyr   3x1     gyroscope rate random walk coefficients [rad/(s*sqrt(s))]
        N_acc   3x1     accelerometer velocity random walk coefficients [(m/s)/sqrt(s)]
        N_gyr   3x1     gyroscope angle random walk coefficients [rad/sqrt(s)]
        Tc_acc  3x1     accelerometer correlation times [s]
        Tc_gyr  3x1     gyroscope correlation times [s]
  """
  # Random Walk root-PSD noise
  imu.N_acc = (imu.N_acc / 60)              # [(m/s)/sqrt(hr)] -> [(m/s)/sqrt(s)]
  # imu.N_acc = imu.N_acc * GRAVITY * 1e-6    #  [ug/sqrt(Hz)]  -> [(m/s)/sqrt(s)]
  imu.N_gyr = (imu.N_gyr / 60) * D2R;       # [deg/sqrt(hr)]  ->  [rad/sqrt(s)]

  # Rate Random Walk root-PSD rate noise
  imu.K_acc = (imu.K_acc / (3600*60))       # [(m/s)/(hr*sqrt(hr)] -> [(m/s)/(s*sqrt(s))]
  imu.K_gyr = (imu.K_gyr / 60) * D2R;       #  [deg/(hr*sqrt(hr))] ->  [rad/(s*sqrt(s))]

  # Dynamic bias instability
  imu.B_acc = imu.B_acc * 0.001 * GRAVITY;  #   [mg]   -> [(m/s)/s]
  imu.B_gyr = imu.B_gyr / 3600 * D2R;       # [deg/hr] ->  [rad/s];
  
  return imu
  
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
        "perfect": PERFECT,
        "navigation": NAVIGATION,
        "tactical": TACTICAL,
        "industrial": INDUSTRIAL,
        "consumer": CONSUMER,
        "hg1700": HG1700,
        "vn100": VN100,
    }

    imu_name = "".join([i for i in imu_name if i.isalnum()]).casefold()
    return IMUS.get(imu_name.casefold(), TACTICAL)  # defaults to industrial
  
  
#* === Default IMUs ===
#* Derived from VectorNav -> https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-inserrorbudget 
PERFECT = fix_imu_si_errors(IMU(
  f=150,                                # sampling frequency Hz
  B_acc=z,                              # accelerometer bias instability coefficients [mg]
  B_gyr=z,                              # gyroscope bias instability coefficients [deg/hr]
  K_acc=z,                              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=z,                              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=z,                              # accelerometer velocity random coefficients [m/s/sqrt(Hz)]
  N_gyr=z,                              # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=z,                             # accelerometer correlation times [s]
  Tc_gyr=z,                             # gyroscope correlation times [s]
))

NAVIGATION = fix_imu_si_errors(IMU(
  f=150,                                # sampling frequency Hz
  B_acc=np.array([0.01,0.01,0.01]),     # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([0.01,0.01,0.01]),     # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.01,0.01,0.01]),     # accelerometer velocity random coefficients [m/s/sqrt(hr)]
  N_gyr=np.array([0.01,0.01,0.01]),     # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([1000,1000,1000]),    # accelerometer correlation times [s]
  Tc_gyr=np.array([2000,2000,2000]),    # gyroscope correlation times [s]
))

TACTICAL = fix_imu_si_errors(IMU(
  f=150,                                # sampling frequency Hz
  B_acc=np.array([0.1,0.1,0.1]),        # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([1,1,1]),              # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.03,0.03,0.03]),     # accelerometer velocity random coefficients [m/s/sqrt(hr)]
  N_gyr=np.array([0.05,0.05,0.05]),     # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([500,500,500]),       # accelerometer correlation times [s]
  Tc_gyr=np.array([600,600,600]),       # gyroscope correlation times [s]
))

INDUSTRIAL = fix_imu_si_errors(IMU(
  f=150,                                # sampling frequency Hz
  B_acc=np.array([1,1,1]),              # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([10,10,10]),           # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.1,0.1,0.1]),        # accelerometer velocity random coefficients [m/s/sqrt(hr)]
  N_gyr=np.array([0.2,0.2,0.2]),        # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([100,100,100]),       # accelerometer correlation times [s]
  Tc_gyr=np.array([100,100,100]),       # gyroscope correlation times [s]
))

CONSUMER = fix_imu_si_errors(IMU(
  f=150,                                # sampling frequency Hz
  B_acc=np.array([10,10,10]),           # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([100,100,100]),        # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([1,1,1]),              # accelerometer velocity random coefficients [m/s/sqrt(Hz)]
  N_gyr=np.array([2,2,2]),              # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([60,60,60]),          # accelerometer correlation times [s]
  Tc_gyr=np.array([60,60,60]),          # gyroscope correlation times [s]
))


#* === Commonly Used IMUs ===
# Tactical Grade (Honeywell HG1700)
# https://aerospace.honeywell.com/content/dam/aerobt/en/documents/landing-pages/brochures/N61-1619-000-001-HG1700InertialMeasurementUnit-bro.pdf
HG1700 = fix_imu_si_errors(IMU(
  f=500,                                # sampling frequency Hz
  B_acc=np.array([1,1,1]),              # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([1,1,1]),              # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.65,0.65,0.65])*FT2M,# accelerometer velocity random coefficients [m/s/sqrt(Hz)]
  N_gyr=np.array([0.125,0.125,0.125]),  # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([500,500,500]),       # accelerometer correlation times [s]
  Tc_gyr=np.array([600,600,600]),       # gyroscope correlation times [s]
))

# Industrial Grade (VectorNav VN100)
# https://www.vectornav.com/docs/default-source/datasheets/vn-100-datasheet-rev2.pdf?sfvrsn=8e35fd12_10
# not sure why the accelerometer seems better for the vectornav?
VN100 = fix_imu_si_errors(IMU(
  f=100,                                            # sampling frequency Hz
  B_acc=np.array([0.04,0.04,0.04]),                 # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([5,5,5]),                          # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),                          # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),                          # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.14,0.14,0.14])*GRAVITY*1e-3*60, # accelerometer velocity random coefficients [m/s/sqrt(Hz)]
  N_gyr=np.array([0.125,0.125,0.125]),              # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([260,260,260]),                   # accelerometer correlation times [s]
  Tc_gyr=np.array([256,256,256]),                   # gyroscope correlation times [s]
))

# Tactical Grade (e UTC Aerospace Systems SiIMU02)
# https://datasheet.datasheetarchive.com/originals/crawler/utcaerospacesystems.com/95f6002ae75d6cb1da1af934207206b6.pdf
SILMU02 = fix_imu_si_errors(IMU(
  f=250,                                # sampling frequency Hz
  B_acc=np.array([1,1,1]),              # accelerometer bias instability coefficients [mg]
  B_gyr=np.array([2.5,2.5,2.5]),        # gyroscope bias instability coefficients [deg/hr]
  K_acc=np.array([0,0,0]),              # accelerometer acceleration random walk coefficients [(m/s)/(hr*sqrt(hr)]
  K_gyr=np.array([0,0,0]),              # gyroscope rate random walk coefficients [deg/(hr*sqrt(hr))]
  N_acc=np.array([0.5,0.5,0.5]),        # accelerometer velocity random coefficients [m/s/sqrt(Hz)]
  N_gyr=np.array([0.25,0.25,0.25]),     # gyroscope angle random walk coefficients [deg/sqrt(hr)]
  Tc_acc=np.array([300,300,300]),       # accelerometer correlation times [s]
  Tc_gyr=np.array([300,300,300]),       # gyroscope correlation times [s]
))


