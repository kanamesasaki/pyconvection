from pyfluids import HumidAir, InputHumidAir

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
    print("Ra:", ra, "Nu:", nusselt)
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
    if ra >= 1e4 and ra <= 1e7:
        nusselt = 0.54 * ra**(1/4)
    elif ra > 1e7 and ra <= 1e11:
        nusselt = 0.15 * ra**(1/3)
    else:
        raise ValueError("Rayleigh number out of range")
    print("Ra:", ra, "Nu:", nusselt)
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
    if ra >= 1e5 and ra <= 1e11:
        nusselt = 0.27 * ra**(1/4)
    else:
        raise ValueError("Rayleigh number out of range")
    print("Ra:", ra, "Nu:", nusselt)
    return nusselt * k / l


# to execute this script: python -m pyconvection.natural_convection
if __name__ == '__main__':
    print("PyFluid has been imported successfully!")
    
    l = 0.6
    t_s = 90
    t_inf = 30
    h = vertical_plate(l, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")


    area = 0.6 * 0.6
    perimeter = 4 * 0.6
    t_s = 90
    t_inf = 30
    h = horizontal_plate_up(area, perimeter, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")

    h = horizontal_plate_down(area, perimeter, t_s, t_inf)
    print(f"Heat transfer coefficient: {h:.2f} W/m^2K")
