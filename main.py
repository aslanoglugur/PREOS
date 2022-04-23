import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from scipy import integrate

"""Water"""
Tc = 647.0  # Kelvin
Pc = 220.6  # Bar
w = 0.344  # Acentric factor
cp = [33.763, -0.006*(10**(-1)), 0.224*(10**(-4)), -0.100*(10**(-7)), 0.110*(10**(-11))]  # Cp constants: a, b, c, d, e
R = 83.14  # gas constant cm^3*bar/mol*K

# T = float(input("Insert Temperature (K): "))  # Kelvin
# P = float("{:.2f}".format((float(input("Insert Pressure  (bar): ")))))  # Bar
# x = float(input("Insert quality(1 for saturated or superheated vapor, 0 for saturated or sub-cooled liquid): "))
m = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
# Tr = T / Tc
# alpha = (1 + m * (1 - np.sqrt(Tr))) ** 2
# Gamma = m * np.sqrt(Tr / alpha)
# a = 0.45724 * (((R * Tc) ** 2) / Pc) * alpha
b = 0.07780 * ((R * Tc) / Pc)


def peng_robinson(volume, temperature):
    global Tr, a, b
    Tr = temperature / Tc
    a = 0.45724 * (R * Tc) ** 2 / Pc * (1 + m * (1 - np.sqrt(Tr))) ** 2
    b = 0.07780 * ((R * Tc) / Pc)
    sol = (R * temperature) / (volume - b)
    sag = a / (volume * (volume + b) + b * (volume - b))
    pressure = sol - sag
    return pressure


def Z_coefficients(pressure, temperature):
    tr = temperature / Tc
    alpha = (1 + m * (1 - np.sqrt(tr))) ** 2
    Pr = pressure / Pc
    A = 0.45724 * (Pr / (tr ** 2)) * alpha
    B = 0.0778 * (Pr / tr)
    p = - 1 + B
    q = A - 2 * B - 3 * B ** 2
    r = -A * B + B ** 2 + B ** 3
    z = [1, p, q, r]
    roots = np.roots(z)
    Z_roots = np.array(roots)
    Z_real = Z_roots.real
    return Z_real


def p_reduced(volume, temperature, degree_reduced):
    p = peng_robinson(volume, temperature) - degree_reduced
    return p


def p0_values(temperature, degree_reduced):
    roots = []
    for v in np.arange(b + 1, 25000, 0.1):
        if p_reduced(v, temperature, degree_reduced) * p_reduced(v + 0.1, temperature, degree_reduced) <= 0:
            roots.append(v)
            if len(roots) == 3:
                return roots
    return roots


def saturation_components(temperature):
    p = 0
    root = p0_values(temperature, p)
    while len(root) < 3:
        p += 1
        root = p0_values(temperature, p)
        if len(root) >= 3:
            p -= 1
            root = p0_values(temperature, p)
            while len(root) < 3:
                p += 0.01
                root = p0_values(temperature, p)

    while True:
        x1 = root[0]
        x2 = root[1]
        x3 = root[2]

        area1 = trapz(p_reduced(np.arange(x1, x2, 0.01), temperature, p), dx=0.01)
        area2 = trapz(p_reduced(np.arange(x2, x3, 0.01), temperature, p), dx=0.01)

        if abs(area1) <= abs(area2):
            p += 1
            root = p0_values(temperature, p)
        else:
            p -= 1
            root = p0_values(temperature, p)
            while True:
                x1 = root[0]
                x2 = root[1]
                x3 = root[2]
                area1 = trapz(p_reduced(np.arange(x1, x2, 0.01), temperature, p), dx=0.01)
                area2 = trapz(p_reduced(np.arange(x2, x3, 0.01), temperature, p), dx=0.01)
                if abs(area1) < abs(area2):
                    p += 0.01
                    root = p0_values(temperature, p)
                else:
                    saturation_component = {"pressure": float("{:.2f}".format((float(p)))),
                                            "roots": [x1, x3]}
                    return saturation_component


def compressibility_factor(pressure, temperature):
    p = pressure
    t = temperature
    z_coeff = Z_coefficients(p, t)
    z = []
    if t >= Tc:
        z.append(max(z_coeff))
    else:
        P_sat = float("{:.2f}".format(saturation_components(t)["pressure"]))
        if p == P_sat:  # Vapor-Liquid mixture
            z.append(max(z_coeff))
            z.append(min(z_coeff))
        elif p < P_sat:  # Super heated vapor
            z.append(max(z_coeff))
        else:  # Compressed Liquid
            z.append(min(z_coeff))
    return z


# Z = compressibility_factor(15, 343)
# print(Z)
# Ztrue = min(Z)*(1-x)+max(Z)*x
# vol = Ztrue * (R * T / P)
# print(saturation_components(247.74))
#

volu = np.arange(b + 1, 25000, 1)  # molar volume m^3/mol from 20 cm3 to 400 cm3

pres = peng_robinson(volu, 247.74)

plt.plot(volu, pres)
plt.ylabel('Pressure (bar)')
plt.xlabel('Molar Volume (cm^3/mol)')
plt.grid(True)
plt.xlim([0, 1000])
plt.ylim([0, 360])
# plt.show()

# TODO: take x values
"""T P V x değerlerinden 3'ünü inputla, inputlanmayanı bul, Z hesapla. Toplam 4 stream için girilmesi gerekli"""

# TODO: calculate enthalpy


def in_out(pressure, temperature, quality):
    x = quality
    m = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
    Tred = temperature / Tc
    alpha = (1 + m * (1 - np.sqrt(Tred))) ** 2
    Gamma = m * np.sqrt(Tr / alpha)
    Z = compressibility_factor(pressure, temperature)
    Ztrue = min(Z) * (1 - x) + max(Z) * x
    Pr = pressure / Pc
    A = 0.45724 * (Pr / (Tr ** 2)) * alpha
    B = 0.0778 * (Pr / Tr)
    output = 8.314*temperature*(Ztrue - 1 - ((A*(1+Gamma)) / (B*np.sqrt(8))) * np.log((Ztrue + (1 + np.sqrt(2))*B) / (Ztrue + (1 - np.sqrt(2))*B)))
    return output


def Cp(temperature):
    t = temperature
    out = cp[0] + cp[1] * t + cp[2] * t**2 + cp[3] * t**3 + cp[4] * t**4
    return out


def Enthalpy_Change():
    print("Insert INLET STREAM values\n")
    T_in = float(input("Insert INLET Temperature (K): "))  # Kelvin
    x1 = float(input("Insert quality(1 for saturated or superheated vapor, 0 for saturated or sub-cooled liquid): "))
    print("Insert OUTLET STREAM values\n")
    T_out = float(input("Insert OUTLET Temperature (K): "))  # Kelvin
    x2 = float(input("Insert quality(1 for saturated or superheated vapor, 0 for saturated or sub-cooled liquid): "))
    P = float("{:.2f}".format((float(input("Insert Pressure  (bar): ")))))  # Bar
    int_Cp = integrate.quad(Cp, T_in, T_out)
    value = in_out(P, T_out, x2) - in_out(P, T_in, x1) + int_Cp[0]
    return value


# print(in_out(2, 247.74, 0))
# print(integrate.quad(Cp, 329.18, 247.74))

print(saturation_components(349.06))
#print(f"The change in molar enthalpy is: {Enthalpy_Change()} J/mol")

# TODO: calculate COPr and COPhp
