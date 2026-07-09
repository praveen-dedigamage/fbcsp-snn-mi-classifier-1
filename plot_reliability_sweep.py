"""Figure generator for the reliability sweep results (main.tex Fig. reliability).

Data hardcoded from `aggregate_reliability.py --results-dir Results_verify`'s
output (2026-07-09, job 35412677, 9 subjects x 5 folds each, all bugs in
PIPELINE_REFERENCE.md S11a/S11c already fixed by this point). Re-run
aggregate_reliability.py and paste fresh numbers in here if Results_verify
changes.

Output copied to <paper repo>/figures/reliability_sweep.pdf for
\\includegraphics in main.tex, Section "Hardware-Realism Reliability Sweep".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLEAN_MEAN = 0.659799
CHANCE = 0.25
SEVERITIES = [0.0, 0.05, 0.1, 0.2, 0.3]

# sweep -> (title, [acc_mean per severity], [acc_std per severity])
DATA = {
    "csp_weight_noise": ("CSP Weight Noise",
        [0.659799, 0.651073, 0.621200, 0.455093, 0.329171],
        [0.135601, 0.133405, 0.125892, 0.107988, 0.056945]),
    "snn_weight_noise": ("SNN Weight Noise",
        [0.659799, 0.648966, 0.624660, 0.560390, 0.502319],
        [0.135601, 0.133075, 0.127759, 0.113909, 0.097862]),
    "beta_noise": ("LIF Beta Noise",
        [0.659799, 0.640444, 0.621130, 0.586292, 0.559174],
        [0.135601, 0.134792, 0.134361, 0.130963, 0.126147]),
    "filter_bank_noise": ("Filter Bank Noise",
        [0.659799, 0.624850, 0.560475, 0.489340, 0.440988],
        [0.135601, 0.125628, 0.108638, 0.093748, 0.086919]),
    "ea_whitener_noise": ("EA Whitener Noise",
        [0.659799, 0.657712, 0.654128, 0.642311, 0.621458],
        [0.135601, 0.135689, 0.135033, 0.131820, 0.126107]),
    "znorm_noise": ("Z-Norm Noise",
        [0.659799, 0.654471, 0.638156, 0.548781, 0.387118],
        [0.135601, 0.133419, 0.129170, 0.112912, 0.074806]),
    "encoder_threshold_noise": ("Encoder Threshold Noise",
        [0.659799, 0.659749, 0.659896, 0.659958, 0.660046],
        [0.135601, 0.135768, 0.135678, 0.135722, 0.135822]),
    "encoder_adaptation_noise": ("Encoder Adaptation Noise",
        [0.659799, 0.450532, 0.391130, 0.350652, 0.330795],
        [0.135601, 0.087970, 0.058129, 0.038534, 0.031018]),
    "joint_noise_all_sources": ("Joint (All Sources)",
        [0.659799, 0.417670, 0.345733, 0.289317, 0.264008],
        [0.135601, 0.072458, 0.035930, 0.014089, 0.005739]),
}

ORDER = [
    "csp_weight_noise", "snn_weight_noise", "beta_noise",
    "filter_bank_noise", "ea_whitener_noise", "znorm_noise",
    "encoder_threshold_noise", "encoder_adaptation_noise", "joint_noise_all_sources",
]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
})

fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.4), sharex=True, sharey=True)

for ax, key in zip(axes.flat, ORDER):
    title, means, stds = DATA[key]
    means = np.array(means)
    stds = np.array(stds)
    sev = np.array(SEVERITIES)

    ax.plot(sev, means, color="steelblue", lw=1.6, marker="o", ms=3.5, zorder=3,
            label="mean $\\pm$ 1 SD across subjects" if key == "csp_weight_noise" else None)
    ax.fill_between(sev, means - stds, means + stds, color="steelblue", alpha=0.18, zorder=2)
    ax.axhline(CHANCE, color="crimson", lw=0.9, ls="--", alpha=0.75, zorder=1,
               label="chance (0.25)" if key == "csp_weight_noise" else None)
    ax.set_title(title)
    ax.set_ylim(0.05, 0.90)
    ax.set_xlim(-0.01, 0.31)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.grid(alpha=0.25, lw=0.5)

for ax in axes[-1, :]:
    ax.set_xlabel(r"Noise severity ($\sigma_{\mathrm{frac}}$)")
for ax in axes[:, 0]:
    ax.set_ylabel("Test accuracy")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=8.5)

fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig("reliability_sweep.pdf", dpi=300, bbox_inches="tight")
fig.savefig("reliability_sweep.png", dpi=200, bbox_inches="tight")
print("Saved reliability_sweep.pdf / .png")
