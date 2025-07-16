import numpy as np

d2r, r2d = np.pi / 180.0, 180.0 / np.pi


# The following functions take a dictionary of astronomical values (in degrees)
# and return dimensionless scale factors for constituent amplitudes.

def f_unity(a):
    return 1.0


# Schureman equations 73, 65
def f_Mm(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선: np.clip 사용
    sin_omega_sq = np.clip(np.sin(omega) ** 2, 0.0, 1.0)
    sin_i_sq = np.clip(np.sin(i) ** 2, 0.0, 1.0)
    sin_I_sq = np.clip(np.sin(I) ** 2, 0.0, 1.0)
    mean = (2 / 3.0 - sin_omega_sq) * (1 - 3 / 2.0 * sin_i_sq)
    return (2 / 3.0 - sin_I_sq) / mean


# Schureman equations 74, 66
def f_Mf(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    sin_omega_sq = np.clip(np.sin(omega) ** 2, 0.0, 1.0)
    sin_I_sq = np.clip(np.sin(I) ** 2, 0.0, 1.0)
    mean = sin_omega_sq * np.cos(0.5 * i) ** 4
    return sin_I_sq / mean


# Schureman equations 75, 67
def f_O1(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    sin_omega = np.sin(omega)
    sin_I = np.sin(I)
    mean = sin_omega * np.cos(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return (sin_I * np.cos(0.5 * I) ** 2) / mean


# Schureman equations 76, 68
def f_J1(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    sin_2omega = np.sin(2 * omega)
    sin_2I = np.sin(2 * I)
    sin_i_sq = np.clip(np.sin(i) ** 2, 0.0, 1.0)
    mean = sin_2omega * (1 - 3 / 2.0 * sin_i_sq)
    return sin_2I / mean


# Schureman equations 77, 69
def f_OO1(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    sin_omega = np.sin(omega)
    sin_I = np.sin(I)
    mean = sin_omega * np.sin(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return sin_I * np.sin(0.5 * I) ** 2 / mean


# Schureman equations 78, 70
def f_M2(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    mean = np.cos(0.5 * omega) ** 4 * np.cos(0.5 * i) ** 4
    return np.cos(0.5 * I) ** 4 / mean


# Schureman equations 227, 226, 68
# Should probably eventually include the derivations of the magic numbers (0.5023 etc).
def f_K1(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    nu = d2r * a['nu'].value
    # 수치 안정성 개선
    sin_2omega = np.sin(2 * omega)
    sin_2I = np.sin(2 * I)
    sin_i_sq = np.clip(np.sin(i) ** 2, 0.0, 1.0)
    sin2Icosnu_mean = sin_2omega * (1 - 3 / 2.0 * sin_i_sq)
    mean = 0.5023 * sin2Icosnu_mean + 0.1681
    # 수치 안정성을 위해 np.clip 사용
    result = (0.2523 * sin_2I ** 2 + 0.1689 * sin_2I * np.cos(nu) + 0.0283) ** (0.5) / mean
    return np.clip(result, 0.0, np.inf)


# Schureman equations 215, 213, 204
# It can be (and has been) confirmed that the exponent for R_a reads 1/2 via Schureman Table 7
def f_L2(a):
    P = d2r * a['P'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    tan_half_I = np.tan(0.5 * I)
    cos_2P = np.cos(2 * P)
    R_a_inv = (1 - 12 * tan_half_I ** 2 * cos_2P + 36 * tan_half_I ** 4) ** (0.5)
    return f_M2(a) * R_a_inv


# Schureman equations 235, 234, 71
# Again, magic numbers
def f_K2(a):
    omega = d2r * a['omega'].value
    i = d2r * a['i'].value
    I = d2r * a['I'].value
    nu = d2r * a['nu'].value
    # 수치 안정성 개선
    sin_omega_sq = np.clip(np.sin(omega) ** 2, 0.0, 1.0)
    sin_I_sq = np.clip(np.sin(I) ** 2, 0.0, 1.0)
    sin_i_sq = np.clip(np.sin(i) ** 2, 0.0, 1.0)
    sinsqIcos2nu_mean = sin_omega_sq * (1 - 3 / 2.0 * sin_i_sq)
    mean = 0.5023 * sinsqIcos2nu_mean + 0.0365
    # 수치 안정성을 위해 np.clip 사용
    result = (0.2533 * np.sin(I) ** 4 + 0.0367 * sin_I_sq * np.cos(2 * nu) + 0.0013) ** (0.5) / mean
    return np.clip(result, 0.0, np.inf)


# Schureman equations 206, 207, 195
def f_M1(a):
    P = d2r * a['P'].value
    I = d2r * a['I'].value
    # 수치 안정성 개선
    cos_I = np.cos(I)
    cos_2P = np.cos(2 * P)
    cos_half_I = np.cos(0.5 * I)
    Q_a_inv = (0.25 + 1.5 * cos_I * cos_2P * cos_half_I ** (-0.5) + 2.25 * cos_I ** 2 * cos_half_I ** (-4)) ** (0.5)
    return f_O1(a) * Q_a_inv


# See e.g. Schureman equation 149
def f_Modd(a, n):
    return f_M2(a) ** (n / 2.0)


# Node factors u, see Table 2 of Schureman.

def u_zero(a):
    return 0.0


def u_Mf(a):
    result = -2.0 * a['xi'].value
    return np.clip(result, -180.0, 180.0)


def u_O1(a):
    result = 2.0 * a['xi'].value - a['nu'].value
    return np.clip(result, -180.0, 180.0)


def u_J1(a):
    result = -a['nu'].value
    return np.clip(result, -180.0, 180.0)


def u_OO1(a):
    result = -2.0 * a['xi'].value - a['nu'].value
    return np.clip(result, -180.0, 180.0)


def u_M2(a):
    result = 2.0 * a['xi'].value - 2.0 * a['nu'].value
    return np.clip(result, -180.0, 180.0)


def u_K1(a):
    result = -a['nup'].value
    return np.clip(result, -180.0, 180.0)


# Schureman 214
def u_L2(a):
    I = d2r * a['I'].value
    P = d2r * a['P'].value
    # 수치 안정성 개선
    tan_half_I = np.tan(0.5 * I)
    if tan_half_I == 0:
        result = 2.0 * a['xi'].value - 2.0 * a['nu'].value
        return np.clip(result, -180.0, 180.0)
    R = r2d * np.arctan(np.sin(2 * P) / (1 / 6.0 * tan_half_I ** (-2) - np.cos(2 * P)))
    result = 2.0 * a['xi'].value - 2.0 * a['nu'].value - R
    return np.clip(result, -180.0, 180.0)


def u_K2(a):
    result = -2.0 * a['nupp'].value
    return np.clip(result, -180.0, 180.0)


# Schureman 202
def u_M1(a):
    I = d2r * a['I'].value
    P = d2r * a['P'].value
    # 수치 안정성 개선
    cos_I = np.cos(I)
    if cos_I == 0:
        result = a['xi'].value - a['nu'].value
        return np.clip(result, -180.0, 180.0)
    Q = r2d * np.arctan((5 * cos_I - 1) / (7 * cos_I + 1) * np.tan(P))
    result = a['xi'].value - a['nu'].value + Q
    return np.clip(result, -180.0, 180.0)


def u_Modd(a, n):
    return n / 2.0 * u_M2(a)
