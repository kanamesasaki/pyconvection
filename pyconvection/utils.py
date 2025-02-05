import numpy as np
import warnings

GRAVITY = 9.80665
MOLAR_MASS = 0.0289644
GAS_CONSTANT = 8.31447

def atmosphere_pressure(altitude: np.ndarray) -> np.ndarray:
    """
    Calculate the pressure at a given altitude.
    U.S. Standard Atmosphere 1976, Eq (33a, 33b)

    :param altitude: Altitude [m] (numpy array)
    :return: Pressure [Pa] (numpy array)
    """

    altitude = np.array(altitude, dtype=float)
    
    conditions = [
        (altitude >= 0) & (altitude <= 11000),
        (altitude > 11000) & (altitude <= 20000),
        (altitude > 20000) & (altitude <= 32000),
        (altitude > 32000) & (altitude <= 47000),
        (altitude > 47000) & (altitude <= 51000),
        (altitude > 51000) & (altitude <= 71000),
        (altitude > 71000) & (altitude <= 84852),
        (altitude < 0) | (altitude > 84852)
    ]

    functions = [
    lambda h: _p0(h),
    lambda h: _p1(h),
    lambda h: _p2(h),
    lambda h: _p3(h),
    lambda h: _p4(h),
    lambda h: _p5(h),
    lambda h: _p6(h),
    lambda h: np.nan * np.ones_like(h)
    ]

    # Validate range
    if np.any((altitude < 0) | (altitude > 84852)):
        warnings.warn("Some of the altitude values are out of range!")

    return np.piecewise(altitude, conditions, functions)

def _p0(altitude: float) -> float:
    # b = int(0)
    L_Mb = -0.0065
    T_Mb = 15.0 + 273.15
    P_Mb = 101325.0
    H_b = 0.0
    pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(GRAVITY * MOLAR_MASS / (GAS_CONSTANT * L_Mb))
    return pressure

def _p1(altitude: float) -> float:
    # b = int(1)
    L_Mb = 0.0
    T_Mb = -56.5 + 273.15
    H_b = 11000.0
    P_Mb = 22632.67601142928 # _p0(H_b)
    pressure = P_Mb * np.exp(-GRAVITY * MOLAR_MASS * (altitude - H_b) / (GAS_CONSTANT * T_Mb))
    return pressure

def _p2(altitude: float) -> float:
    # b = int(2)
    L_Mb = 0.001
    T_Mb = -56.5 + 273.15
    H_b = 20000.0
    P_Mb = 5475.1769086471 # _p1(H_b)
    pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(GRAVITY * MOLAR_MASS / (GAS_CONSTANT * L_Mb))
    return pressure

def _p3(altitude: float) -> float:
    # b = int(3)
    L_Mb = 0.0028
    T_Mb = -44.5 + 273.15
    H_b = 32000.0
    P_Mb = 868.0932265724903 # _p2(H_b)
    pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(GRAVITY * MOLAR_MASS / (GAS_CONSTANT * L_Mb))
    return pressure

def _p4(altitude: float) -> float:
    # b = int(4)
    L_Mb = 0.0
    T_Mb = -2.5 + 273.15
    H_b = 47000.0
    P_Mb = 110.91994694363213 # _p3(H_b)
    pressure = P_Mb * np.exp(-GRAVITY * MOLAR_MASS * (altitude - H_b) / (GAS_CONSTANT * T_Mb))
    return pressure

def _p5(altitude: float) -> float:
    # b = int(5)
    L_Mb = -0.0028
    T_Mb = -2.5 + 273.15
    H_b = 51000.0
    P_Mb = 66.94771636656894 # _p4(H_b)
    pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(GRAVITY * MOLAR_MASS / (GAS_CONSTANT * L_Mb))
    return pressure

def _p6(altitude: float) -> float:
    # b = int(6)
    L_Mb = -0.0020
    T_Mb = -58.5 + 273.15
    H_b = 71000.0
    P_Mb = 3.957145025771112 # _p5(H_b)
    pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(GRAVITY * MOLAR_MASS / (GAS_CONSTANT * L_Mb))
    return pressure

def atmosphere_temperature(altitude: np.ndarray) -> np.ndarray:
    """
    Calculate the atmosphere temperature at given altitudes.
    U.S. Standard Atmosphere 1976, Eq (23)

    :param altitude: Altitude [m] (numpy array)
    :return: Temperature [℃] (numpy array)
    """

    altitude = np.array(altitude, dtype=float)

    conditions = [
        (altitude >= 0.0) & (altitude <= 11000.0),
        (altitude > 11000.0) & (altitude <= 20000.0),
        (altitude > 20000.0) & (altitude <= 32000.0),
        (altitude > 32000.0) & (altitude <= 47000.0),
        (altitude > 47000.0) & (altitude <= 51000.0),
        (altitude > 51000.0) & (altitude <= 71000.0),
        (altitude > 71000.0) & (altitude <= 84852.0),
        (altitude < 0.0) | (altitude > 84852.0)
    ]
    
    functions = [
        lambda h: 15.0 - 0.0065 * h,
        lambda h: -56.5 * np.ones_like(h),
        lambda h: -56.5 + 0.001 * (h - 20000),
        lambda h: -44.5 + 0.0028 * (h - 32000),
        lambda h: -2.5 * np.ones_like(h),
        lambda h: -2.5 - 0.0028 * (h - 51000),
        lambda h: -58.5 * np.ones_like(h),
        lambda h: np.nan * np.ones_like(h)
    ]
    
    # Validate range
    if np.any((altitude < 0.0) | (altitude > 84852.0)):
        warnings.warn("Some of the altitude values are out of range!")
        
    return np.piecewise(altitude, conditions, functions)