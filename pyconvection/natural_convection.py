from pyfluids import Fluid, FluidsList, Input
import math
import numpy as np
import pandas as pd

def vertical_plate(l: float, t_s: float, t_inf: float, fluid: FluidsList = FluidsList.Air, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on a vertical plate.

    :param l: Length of the plate [m]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :param fluid: Fluid type (pyfluids.FluidsList)
    :param pressure: Pressure [Pa]
    :param g: Gravitational acceleration [m/s^2]
    :return: Heat transfer coefficient [W/m^2K]
    """

    t_f = (t_s + t_inf) / 2
    gas = Fluid(fluid).with_state(
        Input.temperature(t_f), 
        Input.pressure(pressure)
        )
    beta = 1 / (t_f + 273.15)
    nu = gas.kinematic_viscosity
    k = gas.conductivity
    pr = gas.prandtl
    ra = (g * beta * (t_s - t_inf) * l**3) / nu**2 * pr
    nusselt = (0.825 + 0.387 * ra**(1/6) / (1 + (0.492 / pr)**(9/16))**(8/27))**2
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l

def horizontal_plate_up(area: float, perimeter: float, t_s: float, t_inf: float, fluid: FluidsList = FluidsList.Air, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on the top of a horizontal plate.

    :param area: Area of the plate [m^2]
    :param perimeter: Perimeter of the plate [m]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :param fluid: Fluid type (pyfluids.FluidsList)
    :param pressure: Pressure [Pa]
    :param g: Gravitational acceleration [m/s^2]
    :return: Heat transfer coefficient [W/m^2K]
    """

    l = area / perimeter 
    t_f = (t_s + t_inf) / 2
    gas = Fluid(fluid).with_state(
        Input.temperature(t_f), 
        Input.pressure(pressure)
        )
    beta = 1 / (t_f + 273.15)
    nu = gas.kinematic_viscosity
    k = gas.conductivity
    pr = gas.prandtl
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

def horizontal_plate_down(area: float, perimeter: float, t_s: float, t_inf: float, fluid: FluidsList = FluidsList.Air, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on the bottom of a horizontal plate.

    :param area: Area of the plate [m^2]
    :param perimeter: Perimeter of the plate [m]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :param fluid: Fluid type (pyfluids.FluidsList)
    :param pressure: Pressure [Pa]
    :param g: Gravitational acceleration [m/s^2]
    :return: Heat transfer coefficient [W/m^2K]
    """

    l = area / perimeter 
    t_f = (t_s + t_inf) / 2
    gas = Fluid(fluid).with_state(
        Input.temperature(t_f), 
        Input.pressure(pressure)
        )
    beta = 1 / (t_f + 273.15)
    nu = gas.kinematic_viscosity
    k = gas.conductivity
    pr = gas.prandtl
    ra = (g * beta * (t_s - t_inf) * l**3) / nu**2 * pr
    if 1e5 <= ra <= 1e11:
        nusselt = 0.27 * ra**(1/4)
    else:
        print("Rayleigh number out of range:", ra)
        nusselt = math.nan
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l

def sphere(d: float, t_s: float, t_inf: float, fluid: FluidsList = FluidsList.Air, pressure: float = 101325.0, g: float = 9.81) -> float:
    """
    Calculate the heat transfer coefficient for natural convection on a sphere.

    :param d: Diameter of the sphere [m]
    :param t_s: Surface temperature [℃]
    :param t_inf: Fluid temperature [℃]
    :param fluid: Fluid type (pyfluids.FluidsList)
    :param pressure: Pressure [Pa]
    :param g: Gravitational acceleration [m/s^2]
    :return: Heat transfer coefficient [W/m^2K]
    """

    t_f = (t_s + t_inf) / 2
    gas = Fluid(fluid).with_state(
        Input.temperature(t_f), 
        Input.pressure(pressure)
        )
    beta = 1 / (t_f + 273.15)
    nu = gas.kinematic_viscosity
    k = gas.conductivity
    pr = gas.prandtl
    ra = (g * beta * (t_s - t_inf) * d**3) / nu**2 * pr
    if 0 < ra  <= 1e11:
        nusselt = 2 + 0.589 * ra**(1/4) / (1 + (0.469 / pr)**(9/16))**(4/9)
    else:
        print("Rayleigh number out of range:", ra)
        nusselt = math.nan
    # print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / d


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
    

