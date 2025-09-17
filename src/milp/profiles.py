from __future__ import annotations

import numpy as np


def _hours_vector(num_periods: int, dt_hours: float) -> np.ndarray:
    return np.arange(num_periods, dtype=float) * dt_hours


def build_demand_profile(name: str, num_periods: int, dt_hours: float, rng: np.random.Generator) -> np.ndarray:
    hours = _hours_vector(num_periods, dt_hours) % 24.0

    key = (name or "baseline").lower()
    if key in {"flat", "baseline"}:
        base = 0.85 + 0.10 * np.sin(2 * np.pi * hours / 24.0)
    elif key in {"evening_peak", "evening"}:
        peak = np.exp(-((hours - 19.0) / 3.0) ** 2)
        shoulder = 0.5 * np.exp(-((hours - 8.0) / 4.0) ** 2)
        base = 0.55 + 0.35 * peak + 0.1 * shoulder
    elif key in {"double_peak", "commuter"}:
        morning = np.exp(-((hours - 8.0) / 2.5) ** 2)
        evening = np.exp(-((hours - 19.0) / 3.0) ** 2)
        base = 0.35 + 0.3 * morning + 0.35 * evening
    elif key in {"office_peak", "office", "business"}:
        midday = np.exp(-((hours - 13.0) / 2.5) ** 2)
        shoulder = 0.3 * np.exp(-((hours - 9.0) / 3.0) ** 2)
        base = 0.4 + 0.45 * midday + shoulder
    elif key in {"industrial", "factory"}:
        base = 0.75 + 0.12 * np.sin(2 * np.pi * hours / 24.0) + 0.05 * np.sin(2 * np.pi * hours / 8.0)
    elif key in {"night_shift", "overnight"}:
        night = np.exp(-((hours - 2.0) / 2.5) ** 2)
        late = np.exp(-((hours - 22.0) / 2.5) ** 2)
        base = 0.45 + 0.3 * night + 0.25 * late
    elif key in {"residential_morning", "morning_peak"}:
        dawn = np.exp(-((hours - 7.0) / 2.0) ** 2)
        evening = 0.6 * np.exp(-((hours - 19.0) / 3.0) ** 2)
        base = 0.4 + 0.4 * dawn + evening
    else:
        base = 0.7 + 0.25 * np.sin(2 * np.pi * hours / 24.0)

    base += 0.05 * rng.standard_normal(num_periods)
    base = np.clip(base, 0.05, None)
    return base / base.max()


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    values = np.clip(values, 0.0, None)
    peak = float(values.max())
    if peak < 1e-6:
        return np.zeros_like(values)
    return values / peak


def _autocorrelated_noise(rng: np.random.Generator, length: int, window: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=float)
    window = max(1, int(window))
    if window == 1:
        return rng.standard_normal(length)
    raw = rng.standard_normal(length + window - 1)
    kernel = np.ones(window, dtype=float)
    kernel /= kernel.sum()
    smoothed = np.convolve(raw, kernel, mode="valid")
    return smoothed[:length]


def build_solar_profile(
    name: str,
    num_periods: int,
    dt_hours: float,
    rng: np.random.Generator,
    intensity: float,
) -> np.ndarray:
    if num_periods <= 0:
        return np.zeros(0, dtype=float)

    hours = _hours_vector(num_periods, dt_hours) % 24.0
    key = (name or "temperate").lower()

    base_day = np.maximum(0.0, np.sin(np.pi * hours / 24.0))
    midday = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
    shoulder = np.maximum(0.0, np.sin(np.pi * (hours - 4.0) / 16.0))

    if key in {"clear", "sunny", "continental", "desert"}:
        base = 0.55 * base_day ** 1.05 + 0.45 * (base_day * midday ** 1.4)
    elif key in {"temperate", "balanced", "default"}:
        base = 0.5 * base_day ** 1.2 + 0.35 * (base_day * midday ** 1.5) + 0.15 * (base_day * shoulder)
    elif key in {"coastal", "humid", "partly_cloudy"}:
        base = 0.45 * base_day ** 1.35 + 0.4 * (base_day * shoulder ** 1.6) + 0.15 * (base_day * 0.6)
    elif key in {"storm", "monsoon", "cloudy"}:
        base = 0.35 * base_day ** 1.5 + 0.25 * (base_day * shoulder ** 1.8)
    elif key in {"winter", "short_day", "northern"}:
        daylight = np.clip(1.0 - np.abs((hours - 12.0) / 6.0), 0.0, None)
        base = (base_day ** 1.6) * daylight
    else:
        base = 0.48 * base_day ** 1.25 + 0.32 * (base_day * midday ** 1.4) + 0.2 * (base_day * shoulder)

    intensity = float(max(0.0, intensity))
    capped_intensity = min(intensity, 4.0)
    noise_scale = 0.04 + 0.12 * capped_intensity
    window_hours = max(1.5, 2.0 + 3.0 * capped_intensity)
    step = max(dt_hours, 1e-3)
    window = max(1, int(round(window_hours / step)))
    cloud_noise = noise_scale * _autocorrelated_noise(rng, num_periods, window)
    base += cloud_noise

    if capped_intensity > 0.3:
        shading_window = max(1, window // 2)
        shading = 0.06 * capped_intensity * np.maximum(0.0, _autocorrelated_noise(rng, num_periods, shading_window))
        base *= np.clip(1.0 - shading, 0.3, 1.0)

    return _normalize_profile(base)


def build_wind_profile(
    name: str,
    num_periods: int,
    dt_hours: float,
    rng: np.random.Generator,
    intensity: float,
) -> np.ndarray:
    if num_periods <= 0:
        return np.zeros(0, dtype=float)

    hours = _hours_vector(num_periods, dt_hours) % 24.0
    key = (name or "temperate").lower()

    diurnal = 0.5 + 0.35 * np.sin(2 * np.pi * (hours - 14.0) / 24.0)
    mesoscale = 0.45 + 0.25 * np.sin(2 * np.pi * hours / 18.0)
    gust = 0.15 * np.sin(2 * np.pi * hours / 6.0)

    if key in {"windy", "storm", "cyclonic"}:
        base = 0.55 + 0.35 * diurnal + 0.25 * mesoscale + 0.2 * gust
    elif key in {"coastal", "sea_breeze"}:
        base = 0.45 + 0.32 * diurnal + 0.2 * np.sin(2 * np.pi * hours / 9.0)
    elif key in {"continental", "stable", "calm"}:
        base = 0.3 + 0.25 * diurnal + 0.1 * np.sin(2 * np.pi * hours / 16.0)
    elif key in {"mountain", "valley"}:
        base = 0.35 + 0.3 * np.sin(2 * np.pi * (hours - 8.0) / 24.0) + 0.18 * np.sin(2 * np.pi * hours / 8.0)
    else:
        base = 0.4 + 0.28 * diurnal + 0.18 * np.sin(2 * np.pi * hours / 16.0)

    base = np.clip(base, 0.05, None)

    intensity = float(max(0.0, intensity))
    capped_intensity = min(intensity, 4.0)
    long_window_hours = max(2.0, 6.0 - 0.8 * capped_intensity)
    step = max(dt_hours, 1e-3)
    long_window = max(1, int(round(long_window_hours / step)))
    long_noise = (0.08 + 0.05 * capped_intensity) * _autocorrelated_noise(rng, num_periods, long_window)
    short_noise = (0.12 + 0.08 * capped_intensity) * rng.standard_normal(num_periods)
    base = base + long_noise + short_noise

    if capped_intensity > 0.5:
        lull_window = max(1, long_window // 3)
        lull = 0.05 * capped_intensity * np.maximum(0.0, _autocorrelated_noise(rng, num_periods, lull_window))
        base *= np.clip(1.0 - lull, 0.2, 1.0)

    return _normalize_profile(base)


def build_renewable_profile(
    name: str,
    num_periods: int,
    dt_hours: float,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    solar = build_solar_profile(name, num_periods, dt_hours, rng, intensity)
    wind = build_wind_profile(name, num_periods, dt_hours, rng, intensity)
    mix = 0.5 * solar + 0.5 * wind
    return _normalize_profile(mix)


def build_runofriver_profile(num_periods: int, dt_hours: float, intensity: float,
                             rng: np.random.Generator) -> np.ndarray:
    hours = _hours_vector(num_periods, dt_hours) % 24.0
    base = 0.6 + 0.2 * np.sin(2 * np.pi * hours / 24.0)
    base += 0.15 * intensity * rng.standard_normal(num_periods)
    base = np.clip(base, 0.1, None)
    return base / base.max()


def build_inflow_profile(num_periods: int, dt_hours: float, inflow_factor: float,
                         rng: np.random.Generator) -> np.ndarray:
    trend = 0.5 + 0.5 * np.sin(2 * np.pi * _hours_vector(num_periods, dt_hours) / (24.0 * 7.0))
    trend += 0.1 * inflow_factor * rng.standard_normal(num_periods)
    trend = np.clip(trend, 0.05, None)
    return trend / trend.max()
