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
    dtype = np.dtype([("constituent", object), ("amplitude", float), ("phase", float)])

    def __init__(
        self, constituents=None, amplitudes=None, phases=None, model=None, radians=False
    ):
        """
        Initialise a tidal model. Provide constituents, amplitudes and phases OR a model.
        Arguments:
        constituents -- list of constituents used in the model.
        amplitudes -- list of amplitudes corresponding to constituents
        phases -- list of phases corresponding to constituents
        model -- an ndarray of type Tide.dtype representing the constituents, amplitudes and phases.
        radians -- boolean representing whether phases are in radians (default False)
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
                "Must be initialised with constituents, amplitudes and phases; or a model."
            )
        if radians:
            model["phase"] = r2d * model["phase"]
        self.model = model[:]
        self.normalize()

    def prepare(self, *args, **kwargs):
        return Tide._prepare(self.model["constituent"], *args, **kwargs)

    @staticmethod
    def _prepare(constituents, t0, t=None, radians=True):
        """
        Return constituent speed and equilibrium argument at a given time, and constituent node factors at given times.
        Arguments:
        constituents -- list of constituents to prepare
        t0 -- time at which to evaluate speed and equilibrium argument for each constituent
        t -- list of times at which to evaluate node factors for each constituent (default: t0)
        radians -- whether to return the angular arguments in radians or degrees (default: True)
        """
        # The equilibrium argument is constant and taken at the beginning of the
        # time series (t0).  The speed of the equilibrium argument changes very
        # slowly, so again we take it to be constant over any length of data. The
        # node factors change more rapidly.
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
        Arguments:
        t -- array of times at which to evaluate the tidal height
        """
        t0 = t[0]
        hours = self._hours(t0, t)
        # t0 기준으로만 prepare
        speed, u, f, V0 = self.prepare(t0, radians=True)
        H = self.model["amplitude"][:, np.newaxis]
        p = d2r * self.model["phase"][:, np.newaxis]
        # u, f는 리스트에서 첫 번째 요소만 사용 (t0 기준)
        u_single = u[0] if isinstance(u, list) else u
        f_single = f[0] if isinstance(f, list) else f
        hours_array = np.array(hours).reshape(1, -1)
        return Tide._tidal_series(hours_array, H, p, speed, u_single, f_single, V0)

    def highs(self, *args):
        """
        Generator yielding only the high tides.
        Arguments:
        see Tide.extrema()
        """
        for t in filter(lambda e: e[2] == "H", self.extrema(*args)):
            yield t

    def lows(self, *args):
        """
        Generator yielding only the low tides.
        Arguments:
        see Tide.extrema()
        """
        for t in filter(lambda e: e[2] == "L", self.extrema(*args)):
            yield t

    def form_number(self):
        """
        Returns the model's form number, a helpful heuristic for classifying tides.
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

        # 각 조화분조가 없을 경우 0으로 처리
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
        Classify the tide according to its form number
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
        Arguments:
        t0 -- time after which extrema are sought
        t1 -- optional time before which extrema are sought (if not given, the generator is infinite)
        partition -- number of hours for which we consider the node factors to be constant (default: 2400.0)
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
        Arguments:
        t0 -- time from which offsets are sought
        t -- times to find hourly offsets from t0.
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
        Arguments:
        hours -- sorted ndarray of hours.
        partition -- maximum partition length (default: 3600.0)
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
        Arguments:
        t0 -- initial time
        hours -- hourly offsets from t0
        """
        if not isinstance(hours, Iterable):
            return Tide._times(t0, [hours])[0]
        elif not isinstance(hours[0], datetime):
            return [t0 + timedelta(hours=h) for h in hours]
        else:
            return list(hours)

    @staticmethod
    def _tidal_series(t, amplitude, phase, speed, u, f, V0):
        # 벡터화된 조석 계산
        # t: (1, n_times) - 시간 배열
        # amplitude: (n_constituents, 1) - 진폭
        # phase: (n_constituents, 1) - 위상
        # speed: (n_constituents, 1) - 속도
        # u, f, V0: (n_constituents, 1) - 노드 팩터들
        
        # 각 조화분조의 기여도 계산
        amplitude_f = amplitude * f  # (n_constituents, 1)
        
        # speed * t: (n_constituents, n_times)
        speed_t = speed * t  # 브로드캐스팅
        
        # V0 + u - phase: (n_constituents, 1)
        angle_offset = V0 + u - phase
        
        # cos 계산 - 브로드캐스팅으로 각 시간에 대해 계산
        cos_term = np.cos(speed_t + angle_offset)
        
        # 각 조화분조의 기여도 합산
        result = np.sum(amplitude_f * cos_term, axis=0)
        
        return result

    @staticmethod
    def _tidal_series_single(t, amplitude, phase, speed, u, f, V0):
        # 단일 시간에 대한 조석 계산
        # t: 스칼라 - 시간
        # amplitude: (n_constituents, 1) - 진폭
        # phase: (n_constituents, 1) - 위상
        # speed: (n_constituents, 1) - 속도
        # u, f, V0: (n_constituents, 1) - 노드 팩터들
        
        # 각 조화분조의 기여도 계산
        amplitude_f = amplitude * f  # (n_constituents, 1)
        
        # speed * t: (n_constituents, 1)
        speed_t = speed * t
        
        # V0 + u - phase: (n_constituents, 1)
        angle_offset = V0 + u - phase
        
        # 최종 각도: speed * t + angle_offset
        final_angle = speed_t + angle_offset
        
        # cos 계산
        cos_term = np.cos(final_angle)
        
        # 각 조화분조의 기여도 합산
        result = np.sum(amplitude_f * cos_term)
        
        return result

    def normalize(self):
        """
        Adapt self.model so that amplitudes are positive and phases are in [0,360) as per convention
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
    ):
        """
        Return an instance of Tide which has been fitted to a series of tidal observations.
        Arguments:
        It is not necessary to provide t0 or interval if t is provided.
        heights -- ndarray of tidal observation heights
        t -- ndarray of tidal observation times
        t0 -- datetime representing the time at which heights[0] was recorded
        interval -- hourly interval between readings
        constituents -- list of constituents to use in the fit (default: constituent.noaa)
        initial -- optional Tide instance to use as first guess for least squares solver
        n_period -- only include constituents which complete at least this many periods (default: 2)
        callback -- optional function to be called at each iteration of the solver
        full_output -- whether to return the output of scipy's leastsq solver (default: False)
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
        
        # 분조가 없는 경우 처리
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
            # 초기 추정치에서 해당 분조들만 추출
            initial_guess = []
            for c in constituents:
                found = False
                for c0, amplitude, phase in initial_data:
                    if c0 == c:
                        initial_guess.append(amplitude)
                        initial_guess.append(phase)
                        found = True
                        break
                if not found:
                    # 해당 분조가 없으면 기본값 사용
                    initial_guess.append(amplitudes[len(initial_guess)//2])
                    initial_guess.append(phases[len(initial_guess)//2])
            initial = np.array(initial_guess, dtype=float)
        else:
            # 기본값
            initial = np.concatenate([amplitudes, phases])

        lsq = least_squares(residual, initial, jac='2-point', ftol=1e-7)

        model = np.zeros(1 + n, dtype=cls.dtype)
        model[0] = (constituent._Z0, z0, 0)
        model[1:]["constituent"] = constituents[:]
        model[1:]["amplitude"] = lsq.x[:n]
        model[1:]["phase"] = lsq.x[n:]

        if full_output:
            return cls(model=model, radians=True), lsq
        return cls(model=model, radians=True)
