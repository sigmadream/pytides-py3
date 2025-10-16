import numpy as np
from collections.abc import Iterable
from datetime import datetime

# Define AstronomicalParameter class for compatibility with existing tests
class AstronomicalParameter:
    """Class to represent astronomical parameters with value (scalar or array) and speed."""
    def __init__(self, value, speed):
        self.value = value
        self.speed = speed


class AstronomicalParameters(dict):
    """
    Mapping of astronomical parameter names to AstronomicalParameter instances.

    The `times` attribute stores the datetime sequence used to generate the parameters.
    Use `for_time(index)` to retrieve a single-time view when multiple times were supplied.
    """

    def __init__(self, parameters, times):
        super().__init__(parameters)
        self.times = tuple(times)

    def for_time(self, index):
        """
        Return an AstronomicalParameters instance for a specific time index.

        Args:
            index: Integer index into the stored times sequence.

        Returns:
            AstronomicalParameters populated with scalar values for the requested time.
        """
        if not self.times:
            raise IndexError("No time information stored.")
        if not (0 <= index < len(self.times)):
            raise IndexError(f"Time index {index} out of range for {len(self.times)} stored times.")

        single_parameters = {
            name: AstronomicalParameter(
                np.asarray(param.value, dtype=np.float64)[index],
                np.asarray(param.speed, dtype=np.float64)[index],
            )
            for name, param in self.items()
        }
        return AstronomicalParameters(single_parameters, [self.times[index]])

# Use NumPy mathematical constants for better precision and consistency
d2r = np.float64(np.pi / 180.0)
r2d = np.float64(180.0 / np.pi)

# Additional mathematical constants for astronomical calculations
DEGREES_PER_HOUR = np.float64(15.0)  # Earth's rotation rate in degrees per hour
SECONDS_PER_DAY = np.float64(24.0 * 60.0 * 60.0)
MINUTES_PER_DAY = np.float64(24.0 * 60.0)
HOURS_PER_DAY = np.float64(24.0)
DAYS_PER_YEAR = np.float64(365.25)
CENTURIES_PER_YEAR = np.float64(100.0)
JULIAN_EPOCH = np.float64(2451545.0)  # Julian Day Number for J2000.0
JULIAN_CENTURY = np.float64(36525.0)  # Days in a Julian century

# Constants for sexagesimal conversions
ARCMIN_PER_DEGREE = np.float64(60.0)
ARCSEC_PER_DEGREE = np.float64(3600.0)
MAS_PER_DEGREE = np.float64(3.6e6)  # milliarcseconds per degree
MUAS_PER_DEGREE = np.float64(3.6e9)  # microarcseconds per degree

# Define parameter names for structured arrays
PARAMETER_NAMES = [
    's', 'h', 'p', 'N', 'pp', '90', 'omega', 'i',
    'I', 'xi', 'nu', 'nup', 'nupp', 'T+h-s', 'P'
]

# Define structured array dtype for astronomical parameters
# This replaces the namedtuple for better memory efficiency
astro_dtype = np.dtype([
    ('s', [('value', np.float64), ('speed', np.float64)]),
    ('h', [('value', np.float64), ('speed', np.float64)]),
    ('p', [('value', np.float64), ('speed', np.float64)]),
    ('N', [('value', np.float64), ('speed', np.float64)]),
    ('pp', [('value', np.float64), ('speed', np.float64)]),
    ('90', [('value', np.float64), ('speed', np.float64)]),
    ('omega', [('value', np.float64), ('speed', np.float64)]),
    ('i', [('value', np.float64), ('speed', np.float64)]),
    ('I', [('value', np.float64), ('speed', np.float64)]),
    ('xi', [('value', np.float64), ('speed', np.float64)]),
    ('nu', [('value', np.float64), ('speed', np.float64)]),
    ('nup', [('value', np.float64), ('speed', np.float64)]),
    ('nupp', [('value', np.float64), ('speed', np.float64)]),
    ('T+h-s', [('value', np.float64), ('speed', np.float64)]),
    ('P', [('value', np.float64), ('speed', np.float64)])
])

# Most of this is based around Meeus's Astronomical Algorithms, since it
# presents reasonably good approximations of all the quantities we require in a
# clear fashion.  Reluctant to go all out and use VSOP87 unless it can be shown
# to make a significant difference to the resulting accuracy of harmonic
# analysis.


def s2d(degrees, arcmins=0.0, arcsecs=0.0, mas=0.0, muas=0.0):
    """
    Convert a sexagesimal angle into decimal degrees.
    
    Args:
        degrees: Degrees component
        arcmins: Arcminutes component (default: 0.0)
        arcsecs: Arcseconds component (default: 0.0)
        mas: Milliarcseconds component (default: 0.0)
        muas: Microarcseconds component (default: 0.0)
        
    Returns:
        Decimal degrees
    """
    return np.float64(
        degrees
        + (arcmins / ARCMIN_PER_DEGREE)
        + (arcsecs / ARCSEC_PER_DEGREE)
        + (mas / MAS_PER_DEGREE)
        + (muas / MUAS_PER_DEGREE)
    )


def polynomial(coefficients, argument):
    """
    Evaluate a polynomial at argument using NumPy's optimized polyval.
    
    Args:
        coefficients: Polynomial coefficients
        argument: Argument to evaluate polynomial at
        
    Returns:
        Polynomial value
    """
    result = np.polyval(coefficients[::-1], argument)
    if np.ndim(result) == 0:
        return result
    elif np.size(result) == 1:
        return result.item()
    return result


def d_polynomial(coefficients, argument):
    """
    Evaluate the first derivative of a polynomial at argument using NumPy's optimized polyder.
    
    Args:
        coefficients: Polynomial coefficients
        argument: Argument to evaluate derivative at
        
    Returns:
        Derivative value
    """
    if len(coefficients) == 0:
        return 0.0 if np.ndim(argument) == 0 else np.zeros_like(argument, dtype=np.float64)
    # Use NumPy's polyder for derivative calculation
    derivative_coeffs = np.polyder(coefficients[::-1])
    result = np.polyval(derivative_coeffs, argument)
    if np.ndim(result) == 0:
        return result
    elif np.size(result) == 1:
        return result.item()
    return result


def T(t):
    """
    Meeus formula 11.1 - Calculate Julian centuries since J2000.0.
    
    Args:
        t: Datetime object
        
    Returns:
        Julian centuries since J2000.0
    """
    return np.float64((JD(t) - JULIAN_EPOCH) / JULIAN_CENTURY)


def JD(t):
    """
    Meeus formula 7.1 - Calculate Julian Day Number.
    
    Args:
        t: Datetime object
        
    Returns:
        Julian Day Number
    """
    Y, M = t.year, t.month
    D = (
        t.day
        + t.hour / HOURS_PER_DAY
        + t.minute / MINUTES_PER_DAY
        + t.second / SECONDS_PER_DAY
        + t.microsecond / (SECONDS_PER_DAY * 1e6)
    )
    if M <= 2:
        Y = Y - 1
        M = M + 12
    A = np.floor(Y / 100.0)
    B = 2 - A + np.floor(A / 4.0)
    return np.float64(np.floor(DAYS_PER_YEAR * (Y + 4716)) + np.floor(30.6001 * (M + 1)) + D + B - 1524.5)


# Meeus formula 21.3
terrestrial_obliquity_coefficients = np.array([
    s2d(23, 26, 21.448),
    -s2d(0, 0, 4680.93),
    -s2d(0, 0, 1.55),
    s2d(0, 0, 1999.25),
    -s2d(0, 0, 51.38),
    -s2d(0, 0, 249.67),
    -s2d(0, 0, 39.05),
    s2d(0, 0, 7.12),
    s2d(0, 0, 27.87),
    s2d(0, 0, 5.79),
    s2d(0, 0, 2.45),
], dtype=np.float64)

# Adjust these coefficients for parameter T rather than U
terrestrial_obliquity_coefficients_adjusted = np.array([
    c * (1e-2) ** i for i, c in enumerate(terrestrial_obliquity_coefficients)
], dtype=np.float64)

# Not entirely sure about this interpretation, but this is the difference
# between Meeus formulae 24.2 and 24.3 and seems to work
solar_perigee_coefficients = np.array((
    280.46645 - 357.52910,
    36000.76932 - 35999.05030,
    0.0003032 + 0.0001559,
    0.00000048,
), dtype=np.float64)

# Meeus formula 24.2
solar_longitude_coefficients = np.array((280.46645, 36000.76983, 0.0003032), dtype=np.float64)

# This value is taken from JPL Horizon and is essentially constant
lunar_inclination_coefficients = np.array((5.145,), dtype=np.float64)

# Meeus formula 45.1
lunar_longitude_coefficients = np.array((
    218.3164591,
    481267.88134236,
    -0.0013268,
    1 / 538841.0 - 1 / 65194000.0,
), dtype=np.float64)

# Meeus formula 45.7
lunar_node_coefficients = np.array((
    125.0445550,
    -1934.1361849,
    0.0020762,
    1 / 467410.0,
    -1 / 60616000.0,
), dtype=np.float64)

# Meeus, unnumbered formula directly preceded by 45.7
lunar_perigee_coefficients = np.array((
    83.3532430,
    4069.0137111,
    -0.0103238,
    -1 / 80053.0,
    1 / 18999000.0,
), dtype=np.float64)


# Now follow some useful auxiliary values, we won't need their speed.
# See notes on Table 6 in Schureman for I, nu, xi, nu', 2nu''
def _I(N, i, omega):
    """
    Calculate I parameter with vectorized operations.
    
    Args:
        N: Lunar node longitude
        i: Lunar inclination
        omega: Terrestrial obliquity
        
    Returns:
        I parameter value
    """
    N, i, omega = d2r * N, d2r * i, d2r * omega
    # Use optimized NumPy trigonometric functions with vectorization
    cosI = np.cos(i) * np.cos(omega) - np.sin(i) * np.sin(omega) * np.cos(N)
    # Improve numerical stability: use np.clip
    cosI = np.clip(cosI, -1.0, 1.0)
    return r2d * np.arccos(cosI)


def _xi(N, i, omega):
    """
    Calculate xi parameter with vectorized operations.
    
    Args:
        N: Lunar node longitude
        i: Lunar inclination
        omega: Terrestrial obliquity
        
    Returns:
        xi parameter value
    """
    N, i, omega = d2r * N, d2r * i, d2r * omega
    # Use optimized NumPy trigonometric functions with better numerical stability
    omega_minus_i = omega - i
    omega_plus_i = omega + i
    half_N = 0.5 * N
    
    cos_half_omega_minus_i = np.cos(0.5 * omega_minus_i)
    cos_half_omega_plus_i = np.cos(0.5 * omega_plus_i)
    sin_half_omega_minus_i = np.sin(0.5 * omega_minus_i)
    sin_half_omega_plus_i = np.sin(0.5 * omega_plus_i)
    tan_half_N = np.tan(half_N)
    
    # Improve numerical stability: prevent division by zero
    cos_half_omega_plus_i = np.where(np.abs(cos_half_omega_plus_i) < 1e-10, 1e-10, cos_half_omega_plus_i)
    sin_half_omega_plus_i = np.where(np.abs(sin_half_omega_plus_i) < 1e-10, 1e-10, sin_half_omega_plus_i)
    
    e1 = (cos_half_omega_minus_i / cos_half_omega_plus_i) * tan_half_N
    e2 = (sin_half_omega_minus_i / sin_half_omega_plus_i) * tan_half_N
    
    # Improve numerical stability
    e1 = np.clip(e1, -1e10, 1e10)
    e2 = np.clip(e2, -1e10, 1e10)
    
    result = r2d * 0.5 * (np.arctan(e1) + np.arctan(e2))
    return np.clip(result, -180.0, 180.0)


def _nu(N, i, omega):
    """
    Calculate nu parameter with vectorized operations.
    
    Args:
        N: Lunar node longitude
        i: Lunar inclination
        omega: Terrestrial obliquity
        
    Returns:
        nu parameter value
    """
    N, i, omega = d2r * N, d2r * i, d2r * omega
    # Use optimized NumPy trigonometric functions with better numerical stability
    omega_minus_i = omega - i
    omega_plus_i = omega + i
    half_N = 0.5 * N
    
    cos_half_omega_minus_i = np.cos(0.5 * omega_minus_i)
    cos_half_omega_plus_i = np.cos(0.5 * omega_plus_i)
    sin_half_omega_minus_i = np.sin(0.5 * omega_minus_i)
    sin_half_omega_plus_i = np.sin(0.5 * omega_plus_i)
    tan_half_N = np.tan(half_N)
    
    # Improve numerical stability: prevent division by zero
    cos_half_omega_plus_i = np.where(np.abs(cos_half_omega_plus_i) < 1e-10, 1e-10, cos_half_omega_plus_i)
    sin_half_omega_plus_i = np.where(np.abs(sin_half_omega_plus_i) < 1e-10, 1e-10, sin_half_omega_plus_i)
    
    e1 = (cos_half_omega_minus_i / cos_half_omega_plus_i) * tan_half_N
    e2 = (sin_half_omega_minus_i / sin_half_omega_plus_i) * tan_half_N
    
    # Improve numerical stability
    e1 = np.clip(e1, -1e10, 1e10)
    e2 = np.clip(e2, -1e10, 1e10)
    
    result = r2d * 0.5 * (np.arctan(e1) - np.arctan(e2))
    return np.clip(result, -180.0, 180.0)


def _nup(N, i, omega):
    """
    Calculate nup parameter with vectorized operations.
    
    Args:
        N: Lunar node longitude
        i: Lunar inclination
        omega: Terrestrial obliquity
        
    Returns:
        nup parameter value
    """
    N, i, omega = d2r * N, d2r * i, d2r * omega
    # Use optimized NumPy trigonometric functions
    result = r2d * (np.arctan(np.sin(omega) * np.tan(N)))
    return np.clip(result, -180.0, 180.0)


def _nupp(N, i, omega):
    """
    Calculate nupp parameter with vectorized operations.
    
    Args:
        N: Lunar node longitude
        i: Lunar inclination
        omega: Terrestrial obliquity
        
    Returns:
        nupp parameter value
    """
    N, i, omega = d2r * N, d2r * i, d2r * omega
    # Use optimized NumPy trigonometric functions
    result = r2d * (np.arctan(np.sin(omega) * np.tan(2 * N)))
    return np.clip(result, -180.0, 180.0)


def create_astro_array():
    """
    Create a structured array for astronomical parameters.
    
    Returns:
        Zero-initialized structured array
    """
    return np.zeros(1, dtype=astro_dtype)


def astro(t):
    """
    Return the astronomical parameters at time t.
    
    Optimized version with better memory efficiency and numerical stability.
    
    Args:
        t: Datetime object or iterable of datetime objects
        
    Returns:
        AstronomicalParameters mapping names to AstronomicalParameter instances. When
        multiple times are supplied, each parameter stores NumPy arrays and the returned
        object retains the originating `times`.
    """
    if isinstance(t, datetime):
        times = [t]
    elif isinstance(t, Iterable) and not isinstance(t, (str, bytes)):
        times = list(t)
        if not times:
            raise ValueError("astro() requires at least one datetime when an iterable is provided.")
        for idx, each in enumerate(times):
            if not isinstance(each, datetime):
                raise TypeError(f"astro() expects datetime instances; element {idx} has type {type(each)!r}.")
    else:
        times = [t]
    
    # Convert times to NumPy arrays
    t_array = np.asarray([T(t_i) for t_i in times], dtype=np.float64)
    jd_array = np.asarray([JD(t_i) for t_i in times], dtype=np.float64)
    
    # Vectorized calculations
    s = polynomial(solar_longitude_coefficients, t_array)
    h = polynomial(lunar_longitude_coefficients, t_array)
    p = polynomial(lunar_perigee_coefficients, t_array)
    N = polynomial(lunar_node_coefficients, t_array)
    pp = polynomial(solar_perigee_coefficients, t_array)
    
    # Improve numerical stability
    s = np.mod(s, 360.0)
    h = np.mod(h, 360.0)
    p = np.mod(p, 360.0)
    N = np.mod(N, 360.0)
    pp = np.mod(pp, 360.0)
    
    # Calculate astronomical parameters
    omega = polynomial(terrestrial_obliquity_coefficients_adjusted, t_array)
    i = polynomial(lunar_inclination_coefficients, t_array)
    
    # Improve numerical stability
    omega = np.clip(omega, 0.0, 360.0)
    i = np.clip(i, 0.0, 360.0)
    
    # Calculate auxiliary parameters (vectorized)
    I = _I(N, i, omega)
    xi = _xi(N, i, omega)
    nu = _nu(N, i, omega)
    nup = _nup(N, i, omega)
    nupp = _nupp(N, i, omega)
    
    # Calculate speeds (vectorized) - convert Julian Century angles to hourly angles
    s_speed = d_polynomial(solar_longitude_coefficients, t_array) / (JULIAN_CENTURY * 24.0)
    h_speed = d_polynomial(lunar_longitude_coefficients, t_array) / (JULIAN_CENTURY * 24.0)
    p_speed = d_polynomial(lunar_perigee_coefficients, t_array) / (JULIAN_CENTURY * 24.0)
    N_speed = d_polynomial(lunar_node_coefficients, t_array) / (JULIAN_CENTURY * 24.0)
    pp_speed = d_polynomial(solar_perigee_coefficients, t_array) / (JULIAN_CENTURY * 24.0)
    
    # Improve numerical stability - astronomical speeds are generally small values
    s_speed = np.clip(s_speed, -10.0, 10.0)
    h_speed = np.clip(h_speed, -10.0, 10.0)
    p_speed = np.clip(p_speed, -10.0, 10.0)
    N_speed = np.clip(N_speed, -10.0, 10.0)
    pp_speed = np.clip(pp_speed, -10.0, 10.0)
    
    # Generate results
    t_plus_h_minus_s = np.mod(((jd_array - np.floor(jd_array)) * 360.0) + h - s, 360.0)
    t_plus_h_minus_s_speed = 15.0 + h_speed - s_speed

    parameters = {
        's': AstronomicalParameter(s, s_speed),
        'h': AstronomicalParameter(h, h_speed),
        'p': AstronomicalParameter(p, p_speed),
        'N': AstronomicalParameter(N, N_speed),
        'pp': AstronomicalParameter(pp, pp_speed),
        '90': AstronomicalParameter(np.full_like(s, 90.0), np.zeros_like(s_speed)),
        'omega': AstronomicalParameter(omega, np.zeros_like(s_speed)),
        'i': AstronomicalParameter(i, np.zeros_like(s_speed)),
        'I': AstronomicalParameter(I, np.zeros_like(s_speed)),
        'xi': AstronomicalParameter(xi, np.zeros_like(s_speed)),
        'nu': AstronomicalParameter(nu, np.zeros_like(s_speed)),
        'nup': AstronomicalParameter(nup, np.zeros_like(s_speed)),
        'nupp': AstronomicalParameter(nupp, np.zeros_like(s_speed)),
        'T+h-s': AstronomicalParameter(t_plus_h_minus_s, t_plus_h_minus_s_speed),
        'P': AstronomicalParameter(p, p_speed),
    }

    result = AstronomicalParameters(parameters, times)

    if len(times) == 1:
        single_parameters = {
            name: AstronomicalParameter(
                float(np.asarray(param.value).item()),
                float(np.asarray(param.speed).item()),
            )
            for name, param in result.items()
        }
        return AstronomicalParameters(single_parameters, times)

    return result


def vectorized_astro(times):
    """
    Vectorized astronomical parameter calculation.
    
    Optimized version for large-scale data processing.
    
    Args:
        times: Array of datetime objects
        
    Returns:
        Structured array of astronomical parameters
    """
    if not isinstance(times, np.ndarray):
        times = np.asarray(times)
    
    # Convert times to Julian Day
    T_array = np.asarray([T(t) for t in times], dtype=np.float64)
    
    # Vectorized calculations
    s = polynomial(solar_longitude_coefficients, T_array)
    h = polynomial(lunar_longitude_coefficients, T_array)
    p = polynomial(lunar_perigee_coefficients, T_array)
    N = polynomial(lunar_node_coefficients, T_array)
    pp = polynomial(solar_perigee_coefficients, T_array)
    
    # Improve numerical stability
    s = np.mod(s, 360.0)
    h = np.mod(h, 360.0)
    p = np.mod(p, 360.0)
    N = np.mod(N, 360.0)
    pp = np.mod(pp, 360.0)
    
    # Calculate astronomical parameters
    omega = polynomial(terrestrial_obliquity_coefficients_adjusted, T_array)
    i = polynomial(lunar_inclination_coefficients, T_array)
    
    # Improve numerical stability
    omega = np.clip(omega, 0.0, 360.0)
    i = np.clip(i, 0.0, 360.0)
    
    # Calculate auxiliary parameters (vectorized)
    I = _I(N, i, omega)
    xi = _xi(N, i, omega)
    nu = _nu(N, i, omega)
    nup = _nup(N, i, omega)
    nupp = _nupp(N, i, omega)
    
    # Calculate speeds (vectorized) - convert Julian Century angles to hourly angles
    s_speed = d_polynomial(solar_longitude_coefficients, T_array) / (JULIAN_CENTURY * 24.0)
    h_speed = d_polynomial(lunar_longitude_coefficients, T_array) / (JULIAN_CENTURY * 24.0)
    p_speed = d_polynomial(lunar_perigee_coefficients, T_array) / (JULIAN_CENTURY * 24.0)
    N_speed = d_polynomial(lunar_node_coefficients, T_array) / (JULIAN_CENTURY * 24.0)
    pp_speed = d_polynomial(solar_perigee_coefficients, T_array) / (JULIAN_CENTURY * 24.0)
    
    # Improve numerical stability - astronomical speeds are generally small values
    s_speed = np.clip(s_speed, -10.0, 10.0)
    h_speed = np.clip(h_speed, -10.0, 10.0)
    p_speed = np.clip(p_speed, -10.0, 10.0)
    N_speed = np.clip(N_speed, -10.0, 10.0)
    pp_speed = np.clip(pp_speed, -10.0, 10.0)
    
    # Return results as structured array
    result = np.zeros(len(times), dtype=astro_dtype)
    
    result['s']['value'] = s
    result['s']['speed'] = s_speed
    result['h']['value'] = h
    result['h']['speed'] = h_speed
    result['p']['value'] = p
    result['p']['speed'] = p_speed
    result['N']['value'] = N
    result['N']['speed'] = N_speed
    result['pp']['value'] = pp
    result['pp']['speed'] = pp_speed
    result['90']['value'] = 90.0
    result['90']['speed'] = 0.0
    result['omega']['value'] = omega
    result['omega']['speed'] = 0.0
    result['i']['value'] = i
    result['i']['speed'] = 0.0
    result['I']['value'] = I
    result['I']['speed'] = 0.0
    result['xi']['value'] = xi
    result['xi']['speed'] = 0.0
    result['nu']['value'] = nu
    result['nu']['speed'] = 0.0
    result['nup']['value'] = nup
    result['nup']['speed'] = 0.0
    result['nupp']['value'] = nupp
    result['nupp']['speed'] = 0.0
    # T+h-s calculation: T is the angle of time
    T_values = np.array([((JD(t) - int(JD(t))) * 360.0) % 360.0 for t in times])
    result['T+h-s']['value'] = (T_values + h - s) % 360.0
    result['T+h-s']['speed'] = 15.0 + h_speed - s_speed
    result['P']['value'] = p
    result['P']['speed'] = p_speed
    
    return result


def vectorized_polynomial(coefficients, arguments):
    """
    Vectorized polynomial calculation.
    
    Optimized for large-scale data processing.
    
    Args:
        coefficients: Polynomial coefficients
        arguments: Arguments to evaluate polynomial at
        
    Returns:
        Polynomial values
    """
    result = np.polyval(coefficients[::-1], arguments)
    if np.ndim(result) == 0:
        return result
    elif np.size(result) == 1:
        return result.item()
    return result


def vectorized_d_polynomial(coefficients, arguments):
    """
    Vectorized polynomial derivative calculation.
    
    Optimized for large-scale data processing.
    
    Args:
        coefficients: Polynomial coefficients
        arguments: Arguments to evaluate derivative at
        
    Returns:
        Derivative values
    """
    if len(coefficients) == 0:
        return np.zeros_like(arguments, dtype=np.float64)
    derivative_coeffs = np.polyder(coefficients[::-1])
    result = np.polyval(derivative_coeffs, arguments)
    if np.ndim(result) == 0:
        return result
    elif np.size(result) == 1:
        return result.item()
    return result
