import numpy as np
import pytidespy3.constituent as constituent

from collections import OrderedDict
from collections.abc import Iterable
from itertools import takewhile, count
from datetime import datetime, timedelta
from scipy.optimize import least_squares
from pytidespy3.astro import astro

d2r, r2d = np.pi / 180.0, 180.0 / np.pi


class Tide(object):
    """
    A class for tidal analysis and prediction.
    
    This class provides methods for analyzing tidal data and predicting
    tidal heights using harmonic analysis.
    """
    dtype = np.dtype([("constituent", object), ("amplitude", float), ("phase", float)])

    def __init__(
        self, constituents=None, amplitudes=None, phases=None, model=None, radians=False
    ):
        """
        Initialize a tidal model.
        
        Provide constituents, amplitudes and phases OR a model.
        
        Args:
            constituents: List of constituents used in the model.
            amplitudes: List of amplitudes corresponding to constituents
            phases: List of phases corresponding to constituents
            model: An ndarray of type Tide.dtype representing the constituents, amplitudes and phases.
            radians: Boolean representing whether phases are in radians (default False)
        """
        if None not in [constituents, amplitudes, phases]:
            if (constituents is not None and amplitudes is not None and phases is not None and
                len(constituents) == len(amplitudes) == len(phases)):
                model = np.zeros(len(phases), dtype=Tide.dtype)
                model["constituent"] = np.array(constituents)
                model["amplitude"] = np.asarray(amplitudes, dtype=np.float64)
                model["phase"] = np.asarray(phases, dtype=np.float64)
            else:
                raise ValueError(
                    "Constituents, amplitudes and phases should all be arrays of equal length."
                )
        elif model is not None:
            if not model.dtype == Tide.dtype:
                raise ValueError("Model must be a numpy array with dtype == Tide.dtype")
        else:
            raise ValueError(
                "Must be initialized with constituents, amplitudes and phases; or a model."
            )
        if radians:
            model["phase"] = r2d * model["phase"]
        self.model = model[:]
        self.normalize()

    def prepare(self, *args, **kwargs):
        """
        Prepare constituent speed and equilibrium argument at a given time.
        
        Returns:
            Prepared tidal parameters for analysis.
        """
        return Tide._prepare(self.model["constituent"], *args, **kwargs)

    @staticmethod
    def _prepare(constituents, t0, t=None, radians=True):
        """
        Return constituent speed and equilibrium argument at a given time, and constituent node factors at given times.
        
        The equilibrium argument is constant and taken at the beginning of the
        time series (t0). The speed of the equilibrium argument changes very
        slowly, so again we take it to be constant over any length of data. The
        node factors change more rapidly.
        
        Args:
            constituents: List of constituents to prepare
            t0: Time at which to evaluate speed and equilibrium argument for each constituent
            t: List of times at which to evaluate node factors for each constituent (default: t0)
            radians: Whether to return the angular arguments in radians or degrees (default: True)
            
        Returns:
            Tuple of (speed, u, f, V0) where:
                speed: Constituent speeds
                u: Node factors u
                f: Node factors f  
                V0: Equilibrium arguments
        """
        if isinstance(t0, Iterable):
            t0 = t0[0]
        if t is None:
            t = [t0]
        if not isinstance(t, Iterable):
            t = [t]
        a0 = astro(t0)
        a = [astro(t_i) for t_i in t]

        # For convenience give u, V0 (but not speed!) in [0, 360)
        V0 = np.asarray([c.V(a0) for c in constituents], dtype=np.float64)[:, np.newaxis]
        speed = np.asarray([c.speed(a0) for c in constituents], dtype=np.float64)[:, np.newaxis]
        u = [
            np.mod(np.asarray([c.u(a_i) for c in constituents], dtype=np.float64)[:, np.newaxis], 360.0)
            for a_i in a
        ]
        f = [
            np.mod(np.asarray([c.f(a_i) for c in constituents], dtype=np.float64)[:, np.newaxis], 360.0)
            for a_i in a
        ]

        if radians:
            speed = d2r * speed
            V0 = d2r * V0
            u = [d2r * each for each in u]
        return speed, u, f, V0

    def at(self, t):
        """
        Return the modelled tidal height at given times.

        Args:
            t: Array of times at which to evaluate the tidal height

        Returns:
            Array of tidal heights at the specified times

        Raises:
            ValueError: 빈 배열이 입력된 경우
        """
        if not isinstance(t, Iterable) or len(t) == 0:
            raise ValueError("t는 비어있지 않은 시간 배열이어야 합니다.")
        t0 = t[0]
        hours = self._hours(t0, t)
        # Prepare using t0 as reference
        speed, u, f, V0 = self.prepare(t0, radians=True)
        H = self.model["amplitude"][:, np.newaxis]
        p = d2r * self.model["phase"][:, np.newaxis]
        # Use only first element from u, f lists (t0 reference)
        u_single = u[0] if isinstance(u, list) else u
        f_single = f[0] if isinstance(f, list) else f
        hours_array = np.array(hours).reshape(1, -1)
        return Tide._tidal_series(hours_array, H, p, speed, u_single, f_single, V0)

    def highs(self, *args):
        """
        Generator yielding only the high tides.
        
        Args:
            *args: Arguments passed to Tide.extrema()
            
        Yields:
            High tide extrema
        """
        for t in filter(lambda e: e[2] == "H", self.extrema(*args)):
            yield t

    def lows(self, *args):
        """
        Generator yielding only the low tides.
        
        Args:
            *args: Arguments passed to Tide.extrema()
            
        Yields:
            Low tide extrema
        """
        for t in filter(lambda e: e[2] == "L", self.extrema(*args)):
            yield t

    def form_number(self):
        """
        Returns the model's form number, a helpful heuristic for classifying tides.
        
        Returns:
            Form number for tide classification
        """
        k1 = np.extract(
            self.model["constituent"] == constituent._K1, self.model["amplitude"]
        )
        o1 = np.extract(
            self.model["constituent"] == constituent._O1, self.model["amplitude"]
        )
        m2 = np.extract(
            self.model["constituent"] == constituent._M2, self.model["amplitude"]
        )
        s2 = np.extract(
            self.model["constituent"] == constituent._S2, self.model["amplitude"]
        )

        # Handle case where constituents are not present
        k1_val = k1[0] if len(k1) > 0 else 0.0
        o1_val = o1[0] if len(o1) > 0 else 0.0
        m2_val = m2[0] if len(m2) > 0 else 0.0
        s2_val = s2[0] if len(s2) > 0 else 0.0

        denominator = m2_val + s2_val
        if denominator == 0:
            return 0.0
        return (k1_val + o1_val) / denominator

    def classify(self):
        """
        Classify the tide according to its form number.
        
        Returns:
            String classification of the tide type
        """
        form = self.form_number()
        if 0 <= form <= 0.25:
            return "semidiurnal"
        elif 0.25 < form <= 1.5:
            return "mixed (semidiurnal)"
        elif 1.5 < form <= 3.0:
            return "mixed (diurnal)"
        else:
            return "diurnal"

    def extrema(self, t0, t1=None, partition=2400.0):
        """
        A generator for high and low tides.
        
        Args:
            t0: Time after which extrema are sought
            t1: Optional time before which extrema are sought (if not given, the generator is infinite)
            partition: Number of hours for which we consider the node factors to be constant (default: 2400.0)
            
        Yields:
            Tuples of (time, height, type) where type is 'H' for high tide or 'L' for low tide
        """
        if t1:
            # yield from in python 3.4
            for e in takewhile(lambda t: t[0] < t1, self.extrema(t0)):
                yield e
        else:
            # We assume that extrema are separated by at least delta hours
            delta = np.amin(
                [
                    90.0 / c.speed(astro(t0))
                    for c in self.model["constituent"]
                    if not c.speed(astro(t0)) == 0
                ]
            )
            # We search for stationary points from offset hours before t0 to
            # ensure we find any which might occur very soon after t0.
            offset = 24.0
            partitions = (Tide._times(t0, i * partition) for i in count()), (
                Tide._times(t0, i * partition) for i in count(1)
            )

            # We'll overestimate to be on the safe side;
            # values outside (start,end) won't get yielded.
            interval_count = int(np.ceil((partition + offset) / delta)) + 1
            amplitude = self.model["amplitude"][:, np.newaxis]
            phase = d2r * self.model["phase"][:, np.newaxis]

            for start, end in zip(*partitions):
                speed, [u], [f], V0 = self.prepare(
                    start, Tide._times(start, 0.5 * partition)
                )

                # These derivatives don't include the time dependence of u or f,
                # but these change slowly.
                def d(t):
                    return np.sum(
                        -speed * amplitude * f * np.sin(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                def d2(t):
                    return np.sum(
                        -(speed**2.0)
                        * amplitude
                        * f
                        * np.cos(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                # We'll overestimate to be on the safe side;
                # values outside (start,end) won't get yielded.
                intervals = (delta * i - offset for i in range(interval_count)), (
                    delta * (i + 1) - offset for i in range(interval_count)
                )
                for a, b in zip(*intervals):
                    if d(a) * d(b) < 0:
                        extrema = least_squares(d, (a + b) / 2.0, jac='2-point').x[0]
                        time = Tide._times(start, extrema)
                        [height] = self.at([time])
                        hilo = "H" if d2(extrema) < 0 else "L"
                        if start < time < end:
                            yield (time, height, hilo)

    @staticmethod
    def _hours(t0, t):
        """
        Return the hourly offset(s) of a (list of) time from a given time.
        
        Args:
            t0: Time from which offsets are sought
            t: Times to find hourly offsets from t0.
            
        Returns:
            Array of hourly offsets
        """
        if not isinstance(t, Iterable):
            return Tide._hours(t0, [t])[0]
        elif isinstance(t[0], datetime):
            return np.array([(ti - t0).total_seconds() / 3600.0 for ti in t])
        else:
            return t

    @staticmethod
    def _partition(hours, partition=3600.0):
        """
        Partition a sorted list of numbers (or in this case hours).
        
        Args:
            hours: Sorted ndarray of hours.
            partition: Maximum partition length (default: 3600.0)
            
        Returns:
            List of partitioned hour arrays
        """
        partition = float(partition)
        relative = hours - hours[0]
        total_partitions = np.ceil(
            relative[-1] / partition + 10 * np.finfo(np.float64).eps
        ).astype("int")
        return [
            hours[np.floor(np.divide(relative, partition)) == i]
            for i in range(total_partitions)
        ]

    @staticmethod
    def _times(t0, hours):
        """
        Return a (list of) datetime(s) given an initial time and an (list of) hourly offset(s).
        
        Args:
            t0: Initial time
            hours: Hourly offsets from t0
            
        Returns:
            List of datetime objects
        """
        if not isinstance(hours, Iterable):
            return Tide._times(t0, [hours])[0]
        elif not isinstance(hours[0], datetime):
            return [t0 + timedelta(hours=h) for h in hours]
        else:
            return list(hours)

    @staticmethod
    def _tidal_series(t, amplitude, phase, speed, u, f, V0):
        """
        Vectorized tidal calculation.
        
        Args:
            t: (1, n_times) - Time array
            amplitude: (n_constituents, 1) - Amplitude
            phase: (n_constituents, 1) - Phase
            speed: (n_constituents, 1) - Speed
            u, f, V0: (n_constituents, 1) - Node factors
            
        Returns:
            Array of tidal heights
        """
        # Calculate contribution of each constituent
        amplitude_f = amplitude * f  # (n_constituents, 1)
        
        # speed * t: (n_constituents, n_times)
        speed_t = speed * t  # Broadcasting
        
        # V0 + u - phase: (n_constituents, 1)
        angle_offset = V0 + u - phase
        
        # Calculate cosine term - broadcasting for each time
        cos_term = np.cos(speed_t + angle_offset)
        
        # Sum contributions from each constituent
        result = np.sum(amplitude_f * cos_term, axis=0)
        
        return result

    @staticmethod
    def _tidal_series_single(t, amplitude, phase, speed, u, f, V0):
        """
        Single time tidal calculation.
        
        Args:
            t: Scalar - Time
            amplitude: (n_constituents, 1) - Amplitude
            phase: (n_constituents, 1) - Phase
            speed: (n_constituents, 1) - Speed
            u, f, V0: (n_constituents, 1) - Node factors
            
        Returns:
            Tidal height at the specified time
        """
        # Calculate contribution of each constituent
        amplitude_f = amplitude * f  # (n_constituents, 1)
        
        # speed * t: (n_constituents, 1)
        speed_t = speed * t
        
        # V0 + u - phase: (n_constituents, 1)
        angle_offset = V0 + u - phase
        
        # Final angle: speed * t + angle_offset
        final_angle = speed_t + angle_offset
        
        # Calculate cosine term
        cos_term = np.cos(final_angle)
        
        # Sum contributions from each constituent
        result = np.sum(amplitude_f * cos_term)
        
        return result

    def normalize(self):
        """
        Adapt self.model so that amplitudes are positive and phases are in [0,360) as per convention.
        """
        for i, (_, amplitude, phase) in enumerate(self.model):
            if amplitude < 0:
                self.model["amplitude"][i] = -amplitude
                self.model["phase"][i] = phase + 180.0
            self.model["phase"][i] = np.mod(self.model["phase"][i], 360.0)

    @classmethod
    def decompose(
        cls,
        heights,
        t=None,
        t0=None,
        interval=None,
        constituents=constituent.noaa,
        initial=None,
        n_period=2,
        callback=None,
        full_output=False,
        weights=None,
        loss='linear',
    ):
        """
        Return an instance of Tide which has been fitted to a series of tidal observations.

        Args:
            heights: Numpy array of tidal observation heights (NaN/inf는 자동 제거)
            t: Numpy array of tidal observation times
            t0: Datetime representing the time at which heights[0] was recorded
            interval: Hourly interval between readings
            constituents: List of constituents to use in the fit (default: constituent.noaa)
            initial: Optional Tide instance to use as first guess for least squares solver
            n_period: Only include constituents which complete at least this many periods (default: 2)
            callback: Optional function to be called at each iteration of the solver
            full_output: Whether to return the output of scipy's leastsq solver (default: False)
            weights: 각 관측값의 가중치 배열 (높을수록 중요, default: None — 균등 가중치)
            loss: scipy least_squares 손실함수 ('linear', 'huber', 'soft_l1', 'cauchy', 'arctan')

        Returns:
            Fitted Tide instance

        Raises:
            ValueError: 유효한 관측값이 없는 경우
        """
        if t is not None:
            if isinstance(t[0], datetime):
                hours = Tide._hours(t[0], t)
                t0 = t[0]
            elif t0 is not None:
                hours = t
            else:
                raise ValueError(
                    "t can be an array of datetimes, or an array "
                    "of hours since t0 in which case t0 must be "
                    "specified."
                )
        elif None not in [t0, interval]:
            hours = np.arange(len(heights)) * interval
        else:
            raise ValueError(
                "Must provide t(datetimes), or t(hours) and "
                "t0(datetime), or interval(hours) and t0(datetime) "
                "so that each height can be identified with an "
                "instant in time."
            )

        # NaN/inf 제거: heights에서 유효하지 않은 값과 대응하는 시간을 함께 제거
        heights = np.asarray(heights, dtype=np.float64)
        hours = np.asarray(hours, dtype=np.float64)
        valid = np.isfinite(heights)
        if not np.all(valid):
            heights = heights[valid]
            hours = hours[valid]
            if weights is not None:
                weights = np.asarray(weights, dtype=np.float64)[valid]
        if len(heights) == 0:
            raise ValueError("유효한 관측값이 없습니다 (모든 값이 NaN 또는 inf).")

        # 가중치 전처리: sqrt(정규화된 weights)로 변환
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if len(weights) != len(heights):
                raise ValueError(
                    f"weights 길이({len(weights)})가 유효한 heights 길이({len(heights)})와 일치하지 않습니다."
                )
            w = np.sqrt(weights / np.mean(weights))
        else:
            w = None

        # Remove duplicate constituents (those which travel at exactly the same
        # speed, irrespective of phase)
        constituents = list(OrderedDict.fromkeys(constituents))

        # No need for least squares to find the mean water level constituent z0,
        # work relative to mean
        constituents = [c for c in constituents if not c == constituent._Z0]
        z0 = np.mean(heights)
        heights = heights - z0

        # Only analyse frequencies which complete at least n_period cycles over
        # the data period.
        if n_period > 0:
            constituents = [
                c for c in constituents if 360.0 * n_period < hours[-1] * c.speed(astro(t0))
            ]
        n = len(constituents)
        
        # Handle case where no constituents are present
        if n == 0:
            if callback:
                callback(heights)
            model = np.zeros(1, dtype=cls.dtype)
            model[0] = (constituent._Z0, z0, 0)
            if full_output:
                return cls(model=model, radians=True), {}
            return cls(model=model, radians=True)

        hours = np.array(hours)
        sort = np.argsort(hours)
        hours = hours[sort]
        heights = heights[sort]

        # We partition our time/height data into intervals over which we consider
        # the values of u and f to assume a constant value (that is, their true
        # value at the midpoint of the interval).  Constituent
        # speeds change much more slowly than the node factors, so we will
        # consider these constant and equal to their speed at t0, regardless of
        # the length of the time series.

        partition = 240.0

        t = Tide._partition(hours, partition)
        times = Tide._times(t0, [(i + 0.5) * partition for i in range(len(t))])

        speed, u, f, V0 = Tide._prepare(constituents, t0, times, radians=True)

        # Residual to be minimised by variation of parameters (amplitudes, phases)
        def residual(hp):
            amplitudes = hp[:n]
            phases = hp[n:]
            H, p = amplitudes[:, np.newaxis], phases[:, np.newaxis]
            s = np.concatenate(
                [
                    Tide._tidal_series(t_i, H, p, speed, u_i, f_i, V0)
                    for t_i, u_i, f_i in zip(t, u, f)
                ]
            )
            res = heights - s
            if w is not None:
                res = res * w
            if callback:
                callback(res)
            return res

        # Analytic Jacobian of the residual - this makes solving significantly
        # faster than just using gradient approximation, especially with many
        # measurements / constituents.
        def D_residual(hp):
            amplitudes = hp[:n]
            phases = hp[n:]
            H, p = amplitudes[:, np.newaxis], phases[:, np.newaxis]
            ds_dH = np.concatenate(
                [
                    f_i * np.cos(speed * t_i + u_i + V0 - p)
                    for t_i, u_i, f_i in zip(t, u, f)
                ],
                axis=1,
            )

            ds_dp = np.concatenate(
                [
                    H * f_i * np.sin(speed * t_i + u_i + V0 - p)
                    for t_i, u_i, f_i in zip(t, u, f)
                ],
                axis=1,
            )

            return np.append(-ds_dH, -ds_dp, axis=0)

        # Initial guess for solver, haven't done any analysis on this since the
        # solver seems to converge well regardless of the initial guess We do
        # however scale the initial amplitude guess with some measure of the
        # variation
        amplitudes = np.ones(n) * (np.sqrt(np.dot(heights, heights)) / len(heights))
        phases = np.ones(n)

        if initial is not None:
            initial_data = getattr(initial, 'model', initial)
            lookup = {
                c0: (amplitude, phase)
                for c0, amplitude, phase in initial_data
            }
            initial_guess = []
            for idx, c in enumerate(constituents):
                if c in lookup:
                    amp, phs = lookup[c]
                else:
                    amp = amplitudes[idx]
                    phs = phases[idx]
                initial_guess.extend([amp, phs])
            initial = np.array(initial_guess, dtype=float)
        else:
            # Default values
            initial = np.concatenate([amplitudes, phases])

        lsq = least_squares(residual, initial, jac='2-point', ftol=1e-7, loss=loss)

        model = np.zeros(1 + n, dtype=cls.dtype)
        model[0] = (constituent._Z0, z0, 0)
        model[1:]["constituent"] = constituents[:]
        model[1:]["amplitude"] = lsq.x[:n]
        model[1:]["phase"] = lsq.x[n:]

        if full_output:
            return cls(model=model, radians=True), lsq
        return cls(model=model, radians=True)
