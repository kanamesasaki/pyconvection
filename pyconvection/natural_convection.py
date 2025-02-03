from pyfluids import HumidAir, InputHumidAir
import math
import numpy as np
import pandas as pd

def vertical_plate(l: float, t_s: float, t_inf: float, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on a vertical plate.

    :param l: Length of the plate [m]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :param pressure: Pressure [Pa]
    :return: Heat transfer coefficient [W/m^2K]
    """

    t_f = (t_s + t_inf) / 2
    humid_air = HumidAir().with_state(
        InputHumidAir.pressure(pressure),
        InputHumidAir.temperature(t_f),
        InputHumidAir.relative_humidity(50)
    )
    beta = 1 / (t_f + 273.15)
    nu = humid_air.kinematic_viscosity
    k = humid_air.conductivity
    pr = humid_air.prandtl
    ra = (g * beta * (t_s - t_inf) * l**3) / nu**2 * pr
    nusselt = (0.825 + 0.387 * ra**(1/6) / (1 + (0.492 / pr)**(9/16))**(8/27))**2
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l

def horizontal_plate_up(area: float, perimeter: float, t_s: float, t_inf: float, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on the top of a horizontal plate.

    :param l: Length of the plate [m]
    :param pressure: Pressure [Pa]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :return: Heat transfer coefficient [W/m^2K]
    """

    l = area / perimeter 
    t_f = (t_s + t_inf) / 2
    humid_air = HumidAir().with_state(
        InputHumidAir.pressure(pressure),
        InputHumidAir.temperature(t_f),
        InputHumidAir.relative_humidity(50)
    )
    beta = 1 / (t_f + 273.15)
    nu = humid_air.kinematic_viscosity
    k = humid_air.conductivity
    pr = humid_air.prandtl
    ra = (g * beta * (t_s - t_inf) * l**3) / nu**2 * pr
    if 1e4 <= ra <= 1e7:
        nusselt = 0.54 * ra**(1/4)
    elif 1e7 < ra <= 1e11:
        nusselt = 0.15 * ra**(1/3)
    else:
        print("Rayleigh number out of range:", ra)
        nusselt = math.nan
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l

def horizontal_plate_down(area: float, perimeter: float, t_s: float, t_inf: float, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on the bottom of a horizontal plate.

    :param l: Length of the plate [m]
    :param pressure: Pressure [Pa]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :return: Heat transfer coefficient [W/m^2K]
    """

    l = area / perimeter 
    t_f = (t_s + t_inf) / 2
    humid_air = HumidAir().with_state(
        InputHumidAir.pressure(pressure),
        InputHumidAir.temperature(t_f),
        InputHumidAir.relative_humidity(50)
    )
    beta = 1 / (t_f + 273.15)
    nu = humid_air.kinematic_viscosity
    k = humid_air.conductivity
    pr = humid_air.prandtl
    ra = (g * beta * (t_s - t_inf) * l**3) / nu**2 * pr
    if 1e5 <= ra <= 1e11:
        nusselt = 0.27 * ra**(1/4)
    else:
        print("Rayleigh number out of range:", ra)
        nusselt = math.nan
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l

def altitude_to_pressure(altitude: float) -> float:
    """
    Calculate the pressure at a given altitude.

    :param altitude: Altitude [m]
    :return: Pressure [Pa]
    """
    g = 9.80665
    M = 0.0289644
    R = 8.31447

    if 0 <= altitude <= 11000:
        # b = int(0)
        L_Mb = -0.0065
        T_Mb = 15.0 + 273.15
        P_Mb = 101325.0
        H_b = 0.0
        pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(g * M / (R * L_Mb))
    elif 11000 < altitude <= 20000:
        # b = int(1)
        L_Mb = 0.0
        T_Mb = -56.5 + 273.15
        P_Mb = 22632.67601142928
        H_b = 11000.0
        pressure = P_Mb * math.exp(-g * M * (altitude - H_b) / (R * T_Mb))
    elif 20000 < altitude <= 32000:
        # b = int(2)
        L_Mb = 0.001
        T_Mb = -56.5 + 273.15
        P_Mb = 5475.1769086471
        H_b = 20000.0
        pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(g * M / (R * L_Mb))
    elif 32000 < altitude <= 47000:
        # b = int(3)
        L_Mb = 0.0028
        T_Mb = -44.5 + 273.15
        P_Mb = 868.0932265724903
        H_b = 32000.0
        pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(g * M / (R * L_Mb))
    elif 47000 < altitude <= 51000:
        # b = int(4)
        L_Mb = 0.0
        T_Mb = -2.5 + 273.15
        P_Mb = 110.91994694363213
        H_b = 47000.0
        pressure = P_Mb * math.exp(-g * M * (altitude - H_b) / (R * T_Mb))
    elif 51000 < altitude <= 71000:
        # b = int(5)
        L_Mb = -0.0028
        T_Mb = -2.5 + 273.15
        P_Mb = 66.94771636656894
        H_b = 51000.0
        pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(g * M / (R * L_Mb))
    elif 71000 < altitude <= 84852:
        # b = int(6)
        L_Mb = -0.0020
        T_Mb = -58.5 + 273.15
        P_Mb = 3.957145025771112
        H_b = 71000.0
        pressure = P_Mb * (T_Mb / (T_Mb + L_Mb * (altitude - H_b)))**(g * M / (R * L_Mb))
    else:
        raise ValueError("Altitude out of range")
    # print("Pressure: ", pressure)
    return pressure

def altitude_to_temperature(altitude: float) -> float:
    """
    Calculate the temperature at a given altitude.

    :param altitude: Altitude [m]
    :return: Temperature [℃]
    """
    if 0 <= altitude <= 11000:
        temperature = 15.0 - 0.0065 * altitude
    elif 11000 < altitude <= 20000:
        temperature = -56.5
    elif 20000 < altitude <= 32000:
        temperature = -56.5 + 0.001 * (altitude - 20000)
    elif 32000 < altitude <= 47000:
        temperature = -44.5 + 0.0028 * (altitude - 32000)
    elif 47000 < altitude <= 51000:
        temperature = -2.5
    elif 51000 < altitude <= 71000:
        temperature = -2.5 - 0.0028 * (altitude - 51000)
    elif 71000 < altitude <= 84852:
        temperature = -58.5
    else:
        raise ValueError("Altitude out of range")
    # print("Temperature: ", temperature)
    return temperature


# to execute this script: python -m pyconvection.natural_convection
if __name__ == '__main__':
    print("PyFluid has been imported successfully!")
    
    l = 0.6
    t_s = 60
    t_inf = 30
    h = vertical_plate(l, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")


    side = 0.6
    area = side**2
    perimeter = 4 * side
    t_s = 60
    t_inf = 30
    h = horizontal_plate_up(area, perimeter, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")

    h = horizontal_plate_down(area, perimeter, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")


    # plate
    l = 0.2
    t_s = 60
    area = l * l
    perimeter = 4 * l
    altitude = np.linspace(0, 22000, 100)
    pressure = np.array([altitude_to_pressure(a) for a in altitude])
    t_inf = np.ones(len(altitude)) * 15.0
    # t_inf = np.array([altitude_to_temperature(a) for a in altitude])

    h_vertical = np.zeros(len(altitude))
    for i in range(len(altitude)):
        h_vertical[i] = vertical_plate(l, t_s, t_inf[i], pressure[i])

    h_horizontal_up = np.zeros(len(altitude))
    for i in range(len(altitude)):
        h_horizontal_up[i] = horizontal_plate_up(area, perimeter, t_s, t_inf[i], pressure[i])
    h_horizontal_down = np.zeros(len(altitude))
    for i in range(len(altitude)):
        h_horizontal_down[i] = horizontal_plate_down(area, perimeter, t_s, t_inf[i], pressure[i])


    # Save results to CSV
    data = {
        'Altitude [m]': altitude,
        'Pressure [Pa]': pressure,
        'T_infinity [°C]': t_inf,
        'h_vertical [W/m²K]': h_vertical,
        'h_horizontal_up [W/m²K]': h_horizontal_up,
        'h_horizontal_down [W/m²K]': h_horizontal_down
    }
    df = pd.DataFrame(data)
    df.to_csv('plate_results_constant.csv', index=False)

    
    
