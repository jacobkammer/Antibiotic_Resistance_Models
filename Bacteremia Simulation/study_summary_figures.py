"""Generate cross-parameter figures for the manuscript Results section.

The script reruns the three calibrated 1,000-patient cohorts so that the
stacked outcomes and relapse-time curves use the same trajectory-based
clearance/reseeding definition as the individual Monte Carlo analyses.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm

import model_Bacteremia as model
from parameters import (
    BASE_PARAMS,
    EMAX_L_BOUNDS,
    LOD,
    NUM_ITERATIONS,
    RNG_SEEDS,
    SAMPLING,
    TOTAL_H,
    VANCO_START,
    Y0,
    bacteremia_outcome_masks,
    canonical_time_grid,
    save_figure,
)
from solver import integrate_model


FIGURES_DIR = "figures"
COLORS = {
    "resolved": "#4c78a8",
    "relapse": "#e45756",
    "persistent": "#8f63b8",
    "density": "#263f6a",
    "tail": "#e45756",
    "mean": "#2a9d8f",
    "switch": "#202020",
}

_worker_times = None
_worker_van = None
_worker_lzd = None


def _lognormal_parameters(mean, cv):
    sigma = np.sqrt(np.log(1.0 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma


def _samples():
    ec_mean, ec_cv = SAMPLING["EC50_L"]
    ec_mu, ec_sigma = _lognormal_parameters(ec_mean, ec_cv)
    ec50 = np.random.default_rng(RNG_SEEDS["MC_ec50_lzd"]).lognormal(
        ec_mu, ec_sigma, NUM_ITERATIONS
    )

    em_mean, em_cv = SAMPLING["Emax_l"]
    em_mu, em_sigma = _lognormal_parameters(em_mean, em_cv)
    emax = np.random.default_rng(RNG_SEEDS["MC_emax_lzd"]).lognormal(
        em_mu, em_sigma, NUM_ITERATIONS
    )

    im_mean, im_cv = SAMPLING["k_immune"]
    im_mu, im_sigma = _lognormal_parameters(im_mean, im_cv)
    immune = np.random.default_rng(RNG_SEEDS["MC_immune_response"]).lognormal(
        im_mu, im_sigma, NUM_ITERATIONS
    )
    return {"EC50_L": ec50, "Emax_l": emax, "k_immune": immune}


def _simulation_components():
    times = canonical_time_grid()
    pk = model.PharmacokineticModel()
    van = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
    lzd = pk.concentration_function(
        "linezolid", TOTAL_H, VANCO_START + pk.van_duration
    )
    return times, van, lzd


def _terminal_reservoir(parameter, value, times, van, lzd):
    params = dict(BASE_PARAMS)
    immune = model.ImmuneResponse()
    if parameter == "k_immune":
        immune = model.ImmuneResponse(k_immune=value)
    else:
        params[parameter] = value
    solution = integrate_model(
        model.dual_reservoir_model,
        Y0,
        times,
        args=(params, van, lzd, immune),
    )
    return solution[-1, 3]


def _classification_boundary(parameter, lower, upper, times, van, lzd):
    """Bisect the hard-LOD clearance/persistence classification change."""
    lower_persists = _terminal_reservoir(parameter, lower, times, van, lzd) >= LOD
    upper_persists = _terminal_reservoir(parameter, upper, times, van, lzd) >= LOD
    if lower_persists == upper_persists:
        raise RuntimeError(f"No persistence classification change for {parameter}")
    for _ in range(30):
        midpoint = 0.5 * (lower + upper)
        midpoint_persists = (
            _terminal_reservoir(parameter, midpoint, times, van, lzd) >= LOD
        )
        if midpoint_persists == lower_persists:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _init_worker():
    global _worker_times, _worker_van, _worker_lzd
    _worker_times, _worker_van, _worker_lzd = _simulation_components()


def _run_cohort(parameter, values):
    outcomes = np.empty(len(values), dtype=np.int8)
    relapse_lags = np.full(len(values), np.nan)
    lzd_end = VANCO_START + model.PharmacokineticModel().van_duration + \
        model.PharmacokineticModel().lzd_duration
    end_index = int(np.searchsorted(_worker_times, lzd_end))

    for index, value in enumerate(values):
        params = dict(BASE_PARAMS)
        immune = model.ImmuneResponse()
        if parameter == "k_immune":
            immune = model.ImmuneResponse(k_immune=value)
        else:
            params[parameter] = value
        solution = integrate_model(
            model.dual_reservoir_model,
            Y0,
            _worker_times,
            args=(params, _worker_van, _worker_lzd, immune),
        )
        resistant_blood = solution[:, 1]
        relapse, persistent, _ = bacteremia_outcome_masks(
            resistant_blood[np.newaxis, :], _worker_times
        )
        if persistent[0]:
            outcomes[index] = 2
        elif relapse[0]:
            outcomes[index] = 1
            returned = np.flatnonzero(resistant_blood[end_index + 1:] >= LOD)
            first_index = end_index + 1 + returned[0]
            relapse_lags[index] = (_worker_times[first_index] - lzd_end) / 24.0
        else:
            outcomes[index] = 0
    return parameter, outcomes, relapse_lags


def _plot_distributions(samples, switches):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    specifications = (
        ("EC50_L", r"$EC_{50,L}$ (mg/L)", "upper"),
        ("Emax_l", r"$E_{max,L}$ (h$^{-1}$)", "lower"),
        ("k_immune", r"$k_{immune}$ (h$^{-1}$)", "lower"),
    )

    for ax, (parameter, xlabel, tail_direction) in zip(axes, specifications):
        values = samples[parameter]
        mean, cv = SAMPLING[parameter]
        switch = switches[parameter]
        mu, sigma = _lognormal_parameters(mean, cv)
        x_upper = max(np.quantile(values, 0.999), switch * 1.15, mean * 1.15)
        x = np.linspace(max(np.quantile(values, 0.0005), 1e-8), x_upper, 600)
        density = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        z_value = (np.log(switch) - mu) / sigma
        theoretical = norm.sf(z_value) if tail_direction == "upper" else norm.cdf(z_value)
        distance_text = fr"log-space $z={z_value:.2f}$"

        tail_mask = x >= switch if tail_direction == "upper" else x <= switch
        realized = np.mean(values >= switch) if tail_direction == "upper" else np.mean(values <= switch)
        ax.hist(values, bins=45, density=True, color="#9fbad0", alpha=0.52,
                edgecolor="white", linewidth=0.4, label="Monte Carlo sample")
        ax.plot(x, density, color=COLORS["density"], lw=2.2,
                label="Calibrated density")
        ax.fill_between(x, 0, density, where=tail_mask, color=COLORS["tail"],
                        alpha=0.30, label="Persistence-side tail")
        ax.axvline(mean, color=COLORS["mean"], lw=2, ls="-.",
                   label=f"Mean = {mean:.3g}")
        ax.axvline(switch, color=COLORS["switch"], lw=2, ls="--",
                   # Same precision as the manuscript: two significant figures, except
                   # three for Emax_l, whose narrow distribution makes 1.85 vs 1.9 material.
                   label=f"Transition = {switch:.3g}" if parameter == "Emax_l"
                   else f"Transition = {switch:.2g}")
        ax.text(0.97, 0.95,
                f"{distance_text}\nTheoretical tail = {100*theoretical:.1f}%\n"
                f"Realized tail = {100*realized:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9})
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(True, ls=":", alpha=0.35)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Calibrated Parameter Distributions Relative to Persistence Transitions",
                 fontsize=18, fontweight="bold")
    fig.tight_layout()
    output = os.path.join(FIGURES_DIR, "distribution_to_switch.png")
    save_figure(fig, output, __file__)
    plt.close(fig)
    print(f"Saved: {output}")


def _plot_outcomes(results):
    labels = [r"$EC_{50,L}$", r"$E_{max,L}$", r"$k_{immune}$"]
    keys = ["EC50_L", "Emax_l", "k_immune"]
    resolved = np.array([np.mean(results[key][0] == 0) for key in keys]) * 100
    relapse = np.array([np.mean(results[key][0] == 1) for key in keys]) * 100
    persistent = np.array([np.mean(results[key][0] == 2) for key in keys]) * 100

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x = np.arange(len(keys))
    layers = ((resolved, "Resolution", COLORS["resolved"]),
              (relapse, "Relapse", COLORS["relapse"]),
              (persistent, "Persistent treatment failure", COLORS["persistent"]))
    bottom = np.zeros(len(keys))
    for values, name, color in layers:
        bars = ax.bar(x, values, bottom=bottom, color=color, width=0.62,
                      edgecolor="white", label=name)
        for bar, value, base in zip(bars, values, bottom):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, base + value/2,
                        f"{value:.1f}%", ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")
        bottom += values

    ax.set_xticks(x, labels)
    ax.set_ylabel("Virtual patients (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Clinical Outcome Classification Across Monte Carlo Analyses",
                 fontsize=17, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False)
    fig.tight_layout()
    output = os.path.join(FIGURES_DIR, "stacked_outcomes.png")
    save_figure(fig, output, __file__)
    plt.close(fig)
    print(f"Saved: {output}")


def _plot_relapse_times(results):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    specifications = (
        ("EC50_L", r"$EC_{50,L}$", "#4c78a8"),
        ("Emax_l", r"$E_{max,L}$", "#e45756"),
        ("k_immune", r"$k_{immune}$", "#59a14f"),
    )
    follow_up_days = (TOTAL_H - (VANCO_START + model.PharmacokineticModel().van_duration
                                 + model.PharmacokineticModel().lzd_duration)) / 24.0
    for parameter, label, color in specifications:
        lags = np.sort(results[parameter][1][np.isfinite(results[parameter][1])])
        cumulative = 100.0 * np.arange(1, len(lags) + 1) / NUM_ITERATIONS
        x = np.r_[0.0, lags, follow_up_days]
        y = np.r_[0.0, cumulative, cumulative[-1] if len(cumulative) else 0.0]
        ax.step(x, y, where="post", lw=2.5, color=color,
                label=f"{label}: {y[-1]:.1f}%")

    ax.set_xlim(0, follow_up_days)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Days after linezolid completion")
    ax.set_ylabel("Cumulative bacteremia relapse (%)")
    ax.set_title("Timing of Post-Linezolid Bacteremia Relapse",
                 fontsize=17, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    output = os.path.join(FIGURES_DIR, "relapse_time_curve.png")
    save_figure(fig, output, __file__)
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    samples = _samples()
    times, van, lzd = _simulation_components()
    switches = {
        "EC50_L": _classification_boundary("EC50_L", 0.05, 3.3, times, van, lzd),
        "Emax_l": _classification_boundary("Emax_l", EMAX_L_BOUNDS[0],
                                             EMAX_L_BOUNDS[1], times, van, lzd),
        "k_immune": _classification_boundary("k_immune", 0.02, 0.65,
                                               times, van, lzd),
    }
    print("Classification switches:", switches, flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=3, initializer=_init_worker) as executor:
        futures = [executor.submit(_run_cohort, key, values)
                   for key, values in samples.items()]
        for future in as_completed(futures):
            parameter, outcomes, relapse_lags = future.result()
            results[parameter] = (outcomes, relapse_lags)
            counts = np.bincount(outcomes, minlength=3)
            print(f"{parameter}: resolved={counts[0]}, relapse={counts[1]}, "
                  f"persistent={counts[2]}", flush=True)

    _plot_distributions(samples, switches)
    _plot_outcomes(results)
    _plot_relapse_times(results)


if __name__ == "__main__":
    main()
