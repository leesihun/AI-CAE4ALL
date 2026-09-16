"""Reproduce research schematics and mathematical counterexamples; no training.

Run from any directory: python <this-file>
All plotted data are explicitly synthetic. Production modules are read only.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import subprocess
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, GRAY = "#2864A0", "#CC6929", "#24846B", "#586575"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, color=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4, edgecolor=color, facecolor=color + "0E"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size, color="#172A3B")


def arrow(ax, p, q, color=GRAY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.4, color=color))


def architecture():
    fig, ax = plt.subplots(figsize=(13.0, 6.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(.02, .96, "Replace the conditional latent prior; keep the simulator fixed",
            fontsize=17, weight="bold")
    box(ax, .025, .39, .20, .31,
        "Known condition c\n\nMesh / geometry\nThickness field\nB.C. mask + values*", size=11)
    box(ax, .025, .78, .20, .10, "Observed field y\ntraining / diagnosis only", GRAY, 10)
    box(ax, .29, .76, .18, .13, "Posterior encoder\nFROZEN", GRAY)
    box(ax, .545, .76, .18, .13, "Posterior samples\nz ~ q(z | y, c)", GRAY)
    arrow(ax, (.227, .83), (.275, .83)); arrow(ax, (.225, .68), (.32, .752))
    arrow(ax, (.483, .83), (.53, .83))
    box(ax, .29, .43, .18, .20, "Prior conditioner\npooled vector first\ngraph tokens later", BLUE)
    box(ax, .545, .43, .18, .20, "Latent generator\nEDM diffusion OR\nscore-trained sampler", ORANGE, 10)
    box(ax, .545, .21, .18, .105, "Independent noise\nnew draw per sample", GREEN, 10)
    box(ax, .795, .43, .18, .20, "Existing decoder\nFROZEN\nfield = D(c, z)", GRAY)
    arrow(ax, (.24, .53), (.278, .53)); arrow(ax, (.483, .53), (.53, .53))
    arrow(ax, (.635, .328), (.635, .414), GREEN)
    arrow(ax, (.635, .744), (.635, .647), GRAY)
    ax.text(.646, .69, "fit distribution", fontsize=9, color=GRAY)
    arrow(ax, (.738, .53), (.78, .53))
    ax.text(.755, .66, "joint z", ha="center", color=ORANGE, fontsize=10)
    # c also conditions the fixed simulator; the prior path is not its only input.
    ax.plot([.13, .13, .88, .88], [.378, .36, .36, .414], color=BLUE, lw=1.2)
    ax.text(.025, .23, "Same known condition\nalso enters the fixed simulator", fontsize=9, color=BLUE)
    ax.text(.795, .17, "Repeat to obtain\na field ensemble\nand QoI distributions", fontsize=11, color=GRAY)
    ax.text(.025, .045,
        "* Include B.C. only where recorded and variable; actual SAOI fixture is unverified here.\n"
        "Design schematic. The posterior branch is unavailable during ordinary inference.",
        fontsize=9, color=GRAY)
    save(fig, "01_prior_architecture")


def joint_example(rng):
    n = 1600
    sign = np.repeat([-1., 1.], n//2)
    z = np.column_stack([2*sign + .22*rng.normal(size=n),
                         2*sign + .22*rng.normal(size=n)])
    shuffled = z.copy(); shuffled[:, 1] = rng.permutation(z[:, 1])
    # Permuting one column preserves each empirical marginal exactly.
    assert np.array_equal(np.sort(z[:, 1]), np.sort(shuffled[:, 1]))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.7))
    for ax, data, title, color in zip(axs[:2], [z, shuffled],
            ["(a) Correct joint samples", "(b) Independent slots"], [BLUE, ORANGE]):
        ax.scatter(data[:, 0], data[:, 1], s=3, alpha=.22, color=color, rasterized=True)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="$z_1$", ylabel="$z_2$", title=title)
        ax.set_aspect("equal")
        ax.text(.05, .91, f"corr = {np.corrcoef(data.T)[0,1]:.3f}", transform=ax.transAxes)
    bins = np.linspace(-3, 3, 45)
    axs[2].hist(z[:, 1], bins, density=True, histtype="step", color=BLUE, lw=2, label="Joint")
    axs[2].hist(shuffled[:, 1], bins, density=True, histtype="step", color=ORANGE, lw=1.8, ls="--", label="Independent")
    axs[2].set(title="(c) Identical marginals", xlabel="$z_2$", ylabel="Density")
    axs[2].legend(frameon=False, fontsize=9)
    bins = np.linspace(-6, 6, 70)
    for data, color, label in [(z, BLUE, "Joint"), (shuffled, ORANGE, "Independent")]:
        axs[3].hist(data[:, 0]-data[:, 1], bins, density=True,
                    histtype="step", lw=1.8, color=color, label=label)
    axs[3].set(title="(d) Wrong derived output", xlabel="$h(z)=z_1-z_2$", ylabel="Density")
    fig.suptitle("Matching every marginal does not match the joint distribution", fontsize=15, y=1.01)
    fig.text(.5, -.01, "Synthetic illustration only. No trained model, measured field, or SAOI result is shown.",
             ha="center", color=GRAY, fontsize=9)
    fig.tight_layout()
    save(fig, "02_joint_dependence")
    return {"joint_correlation": float(np.corrcoef(z.T)[0, 1]),
            "independent_correlation": float(np.corrcoef(shuffled.T)[0, 1]),
            "marginals_identical": True,
            "derived_std_joint": float(np.std(z[:,0]-z[:,1])),
            "derived_std_independent": float(np.std(shuffled[:,0]-shuffled[:,1]))}


def variance_example():
    x = np.linspace(-4, 4, 800)
    pdf = lambda m: np.exp(-.5*((x-m)/.5)**2)/(.5*np.sqrt(2*np.pi))
    wrong = .5 / np.sqrt(4 + .25)
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    for m, color in [(-2, BLUE), (2, GREEN)]:
        axs[0].plot(x, pdf(m), color=color, label=f"q(z | c), mean {m:+d}")
        axs[0].plot(x[::20], pdf(m)[::20], "o", mfc="none", color=color, ms=4)
    axs[0].plot(x, .5*(pdf(-2)+pdf(2)), color=GRAY, ls="--", label="Pooled across conditions")
    axs[0].set(xlabel="Latent coordinate z", ylabel="Density", title="(a) The prior is exactly correct for each c")
    axs[0].legend(frameon=False, fontsize=8, loc="upper center")
    axs[1].barh([1, 0], [wrong, 1.], color=[ORANGE, GREEN], height=.44)
    axs[1].set(yticks=[1, 0], yticklabels=["Within / pooled\n(incompatible)", "Within / within\n(conditional)"],
               xlim=(0, 1.16), xlabel="Standard-deviation ratio", title="(b) The denominator changes the diagnosis")
    axs[1].text(wrong+.02, 1, f"{wrong:.3f}", va="center")
    axs[1].text(1.02, 0, "1.000", va="center")
    fig.text(.5, -.015,
        "Synthetic exact calculation: Var(z | c) = 0.25; Var(E[z | c]) = 4. "
        "This confound disappears for a single fixed condition.", ha="center", fontsize=9, color=GRAY)
    fig.tight_layout()
    save(fig, "03_variance_diagnostic")
    return {"conditional_variance": .25, "between_condition_variance": 4.,
            "correct_prior_misleading_ratio": wrong, "conditional_ratio": 1.}


def protocol():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    ax.text(.025, .94, "Controlled experiment: separate conditioning, density, and decoder sensitivity",
            fontsize=15, weight="bold")
    box(ax, .025, .70, .94, .13,
        "Fixed throughout: E / D weights + normalization + graph hierarchy\n"
        "Condition c = geometry + thickness + recorded B.C.; vary only the random realization", GRAY, 11)
    box(ax, .025, .44, .20, .15, "0. Validate paths\nGT / posterior / prior\ncheckpoint parity", GRAY, 11)
    box(ax, .275, .44, .20, .15, "1. Fit one condition\nCan a prior reproduce\nits latent cloud?", BLUE, 11)
    box(ax, .525, .44, .20, .15, "2. Compare families\nFM / EDM / implicit\nsame conditioner", ORANGE, 11)
    box(ax, .775, .44, .20, .15, "3. Change one factor\ngraph-token context\nthen field score", GREEN, 11)
    for a,b in [(.238,.261),(.488,.511),(.738,.761)]: arrow(ax,(a,.515),(b,.515))
    box(ax, .025, .11, .45, .19,
        "Split A: new realizations at known conditions\nSplit B: held-out geometries / thickness ranges\n"
        "No test targets in a training latent bank", BLUE, 11)
    box(ax, .525, .11, .45, .19,
        "Measure joint latent + field distributions\nPV / tails / spatial dependence / B.C. residuals*\n"
        "Report three seeds, time and denoiser NFE", GREEN, 11)
    ax.text(.025, .035, "Proposed procedure, not completed training. * B.C. residuals require verified physical constraints.",
            fontsize=9, color=GRAY)
    save(fig, "04_experiment_protocol")


def fm_v2():
    fig, ax = plt.subplots(figsize=(13.4, 6.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(.025, .95, "FM v2: conditional moments first, residual flow matching second",
            fontsize=16, weight="bold")
    box(ax, .025, .55, .18, .24,
        "Known condition c\n\nGraph / geometry\nThickness\nRecorded B.C.", BLUE, 11)
    box(ax, .255, .58, .17, .18, "Graph conditioner\nh(c)", BLUE, 12)
    arrow(ax, (.208, .67), (.242, .67))
    box(ax, .47, .70, .19, .15, "Moment head\nμ(c), L(c)", GREEN, 12)
    ax.text(.47, .875, "STAGE A: fit moments", fontsize=9, color=GREEN, weight="bold")
    arrow(ax, (.425, .70), (.458, .765))
    box(ax, .025, .18, .18, .20, "Observed field y\n(training only)", GRAY, 11)
    box(ax, .255, .20, .17, .16, "Posterior encoder\nFROZEN", GRAY, 11)
    arrow(ax, (.208, .28), (.242, .28))
    box(ax, .47, .20, .19, .16, "Target z*\nposterior sample", GRAY, 11)
    arrow(ax, (.425, .28), (.458, .28))
    box(ax, .71, .20, .24, .16, "Residual target\nr* = L(c)⁻¹[z* − μ(c)]", ORANGE, 11)
    arrow(ax, (.665, .28), (.698, .28))
    ax.plot([.565, .565, .70], [.69, .40, .36], color=GREEN, lw=1.3)
    box(ax, .47, .46, .19, .13, "r₀ ~ N(0, I)", GREEN, 12)
    box(ax, .71, .48, .24, .19, "Residual FM\nvθ(rₜ, t, h(c))\nmodels remaining shape", ORANGE, 11)
    ax.text(.71, .695, "STAGE B: freeze moments; fit FM", fontsize=9, color=ORANGE, weight="bold")
    arrow(ax, (.665, .525), (.698, .555))
    arrow(ax, (.425, .64), (.698, .60), BLUE)
    arrow(ax, (.83, .47), (.83, .38), ORANGE)
    box(ax, .71, .025, .24, .115, "Inference: z = μ(c) + L(c)r₁\n→ existing decoder (FROZEN)", GRAY, 10)
    ax.add_patch(FancyArrowPatch((.945, .48), (.945, .145), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.4, color=ORANGE,
                 connectionstyle="arc3,rad=-0.22"))
    ax.text(.955, .31, "r₁", fontsize=10, color=ORANGE)
    ax.text(.025, .085, "A  Fit μ,L on frozen posterior samples", fontsize=9.5, color="#172A3B")
    ax.text(.025, .045, "B  Freeze μ,L; train residual FM   |   C  Optional small-LR joint fine-tune",
            fontsize=9.5, color="#172A3B")
    save(fig, "05_fm_conditional_residual")


def source_class(path, scope):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ConditionalFMPrior"]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope["ConditionalFMPrior"]


def sampler_checks():
    # Execute the actual FM class body with a tiny condition-only base. The graph
    # trunk is deliberately excluded: this probes checkpoint buffers and solver.
    class Base(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.num_z, self.z_dim, self.hidden_dim = 1, 2, 8
        def condition(self, graph):
            return graph
    def mlp(i,h,o,layer_norm=True):
        return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))
    common = {"torch": torch, "nn": nn, "math": math, "_ConditionalPriorBase": Base,
              "build_mlp": mlp, "init_weights": lambda m: None,
              "_autocast_disabled_for": lambda t: nullcontext()}
    native = ROOT / "methods/MeshGraphNets_Variational/model/conditional_prior.py"
    packaged = ROOT / "inference/cae_infer/families/meshgraphnets_v/model/conditional_prior.py"
    Native = source_class(native, common.copy())
    Packaged = source_class(packaged, common.copy())
    torch.manual_seed(916)
    a, b = Native({"prior_fm_steps": 4}), Packaged({"prior_fm_steps": 4})
    b.load_state_dict(a.state_dict(), strict=True)
    assert set(a.state_dict()) == set(b.state_dict())
    # Exact moment check: E(mu)=2, Var(mu)=1, E(sigma^2)=0.25.
    mu = torch.tensor([[[1., 1.]], [[3., 3.]]])
    lv = torch.full_like(mu, math.log(.25))
    shift, scale = a.fit_standardization(mu, lv)
    assert torch.allclose(shift, torch.tensor([2., 2.]))
    assert torch.allclose(scale, torch.full_like(scale, math.sqrt(1.25)))
    return {"scope": "Actual FM class bodies, minimal stub conditioner; no real checkpoint or mesh",
            "strict_load_native_to_package": "passed",
            "native_default_solver": a.solver,
            "packaged_solver": b.solver,
            "posterior_sample_moment_standardization": "passed"}


def main():
    rng = np.random.default_rng(916)
    architecture()
    checks = {"kind": "Mathematical and source-isolated checks, not empirical model results",
              "joint_distribution": joint_example(rng),
              "variance_comparison": variance_example(),
              "sampler_compatibility": sampler_checks()}
    protocol()
    fm_v2()
    # Exact CRPS population counterexample: Y = -1 or +1 equiprobably.
    # Correct F has E|X-Y|=1 and E|X-X'|=1; a point mass at zero has score 1.
    checks["population_crps"] = {"true_distribution": .5, "collapsed_at_mean": 1.,
        "conclusion": "One observed y per condition does not make CRPS improper in expectation"}
    # Frozen decoder parameters still allow gradients through its input.
    decoder = nn.Linear(2, 2, bias=False)
    decoder.requires_grad_(False)
    gen = nn.Linear(2, 2)
    decoder(gen(torch.ones(1, 2))).square().sum().backward()
    assert gen.weight.grad is not None and decoder.weight.grad is None
    checks["frozen_decoder_input_gradient"] = "passed"
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    paths = [
        "methods/MeshGraphNets_Variational/model/conditional_prior.py",
        "methods/MeshGraphNets_Variational/model/mlp.py",
        "methods/MeshGraphNets_Variational/model/vae.py",
        "methods/MeshGraphNets_Variational/training_profiles/training_loop.py",
        "methods/MeshGraphNets_Variational/training_profiles/single_training.py",
        "methods/MeshGraphNets_Variational/training_profiles/distributed_training.py",
        "methods/MeshGraphNets_Variational/misc/retrain_prior.py",
        "inference/cae_infer/families/meshgraphnets_v/model/conditional_prior.py",
        "inference/cae_infer/families/meshgraphnets_v/driver.py",
        "configs/MeshGraphNets_Variational/SAOI_all_input/config_train_bot.txt",
        "configs/MeshGraphNets_Variational/SAOI_sweep/config_train_8.txt",
        "configs/MeshGraphNets_Variational/b8_all_warpage_input/config_train8.txt"]
    snapshot = {"date": "2026-09-16", "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        "data_present": {p: (ROOT/p).exists() for p in [
            "dataset/SAOI/saoi_train_bot.h5", "dataset/SAOI/saoi_train_top.h5",
            "dataset/b8_main_sec_dataset_traincopy.h5"]},
        "note": "FM-v2 model, training, native/package inference parity, diagnostics and controlled sweep were implemented; real SAOI training was not run locally."}
    (OUT / "audit_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print("Saved five figures in PNG, PDF and SVG.")


if __name__ == "__main__":
    main()
