"""OpenRadioss Starter (`*_0000.rad`) / Engine (`*_0001.rad`) deck writer.

Fixed-width field convention, empirically ground-truthed against a real
working deck (OpenRadioss's own ModelExchange repo,
`Components/Materials/ElastoPlasticLaw/Law001/Shells/TEST_01_MB_0000.rad`):
the Starter reader is column-positional, not token-delimited --

  * id / integer / short-text fields : right-justified in 10-char columns
  * float fields                     : right-justified in 20-char columns
    (this includes some fields named like flags, e.g. PROP/SHELL's Ithick --
    width is per-field, not inferable from the field's apparent type)
  * `/BEGIN` unit-system lines        : three 20-char columns
  * `/GRNOD/NODE` member lists        : 10 ids per line, 10-char columns

Every line must end in a single '\\r\\n'. Lines are joined and written in
BINARY mode -- opening in text mode with newline='\\r\\n' re-translates any
'\\n' already present in a joined '\\r\\n'-separated string, doubling every
line ending to '\\r\\r\\n' and making the Starter reject the very first line.

The Engine deck is comparatively free-format (whitespace/line-separated,
not column-positional) -- confirmed by reading TEST_01_MB_0001.rad.

Boundary condition / loading pattern (also read off TEST_01_MB): each end of
the shell is slaved to a rigid body (`/RBODY`) whose ID *is* its master
node's ID (a hard Radioss requirement, not a convention to relax). The
master node is unconstrained by itself; the RBODY's `Gnod_id` field names a
`/GRNOD/NODE` group holding the *slave* ring (the shell's real edge nodes).
`/BCS` fixes/frees the master node's own 6 DOFs, referencing a one-node
`/GRNOD/NODE` group wrapping it. `/IMPVEL` drives the master node along a
`/FUNCT` ramp curve for the loaded end.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- fields
def _i10(x):
    return f"{int(x):>10d}"


def _f20(x):
    # Fixed-exponent form, not %g: %.15g's width varies with magnitude, and a
    # near-zero float64 residual (e.g. cos/sin a hair off a quarter-turn,
    # ~1e-14) renders as a 21-char signed-exponent string that overflows this
    # column and shifts every later field on the line -- silently corrupting
    # whichever coordinate follows, with no parse error on either side.
    # %+.13e is exactly 20 chars for any exponent in [-99, 99].
    return f"{float(x):+.13e}"


def _s10(x):
    return f"{str(x):>10s}"


def _s20(x):
    return f"{str(x):>20s}"


def _write_lines(path, lines):
    data = ("\r\n".join(lines) + "\r\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(data)


def _grnod_node_block(gid, title, node_ids):
    lines = [f"/GRNOD/NODE/{gid}", str(title)[:100]]
    node_ids = list(node_ids)
    for k in range(0, len(node_ids), 10):
        chunk = node_ids[k:k + 10]
        lines.append("".join(_i10(n) for n in chunk))
    return lines


# --------------------------------------------------------------------- decks
class DeckIds:
    """Node/element/part/etc ID allocation for one deck (1-based Radioss IDs)."""

    def __init__(self, mesh):
        self.n_shell_nodes = mesh.n_nodes
        self.node_bottom_master = mesh.n_nodes + 1
        self.node_top_master = mesh.n_nodes + 2
        self.mat_id = 1
        self.prop_id = 1
        self.part_id = 1
        self.rbody_bottom = self.node_bottom_master
        self.rbody_top = self.node_top_master
        self.grnod_bottom_slaves = 1     # shell nodes slaved to the bottom RBODY
        self.grnod_top_slaves = 2        # shell nodes slaved to the top RBODY
        self.grnod_bottom_master = 3     # wraps just node_bottom_master (for /BCS)
        self.grnod_top_master = 4        # wraps just node_top_master (for /BCS and /IMPVEL)
        self.bcs_bottom = 1
        self.bcs_top = 2
        self.funct_ramp = 1
        self.impvel_top = 1


def write_starter(path, mesh, coords, ids, E, nu, rho, thickness,
                   length_scale=1.0, impose_velocity=None, t_ramp=None,
                   title="shell buckling", n_integ=5):
    """Write the Starter deck (`*_0000.rad`) for one geometry realization.

    `coords`      -- (N,3) node coordinates in mesh units (mesh.coords, or
                      mesh.coords perturbed by imperfection.apply_imperfection);
                      scaled by `length_scale` on write so the dimensionless
                      geometry maps onto a physically-sized model.
    `E, nu, rho`  -- material properties in the same unit system as
                      `length_scale` (recommended: mm, ms, kg -> stress in
                      GPa-equivalent so wave speeds keep the explicit time
                      step sane; see build plan step 4).
    `thickness`   -- shell thickness, same length unit as `length_scale`.
    `impose_velocity` -- if given, write /BCS + /RBODY + /IMPVEL + /FUNCT for
                      an axial-compression test at this velocity (model
                      length units per model time unit). If None, both ends
                      are simply clamped (no loading) -- for a pure statics/
                      eigenmode-adjacent sanity check.
    """
    L = float(mesh.L) * length_scale
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append(str(title)[:100])
    lines.append(_i10(2022) + _i10(0))
    lines.append(_s20("kg") + _s20("mm") + _s20("ms"))
    lines.append(_s20("kg") + _s20("mm") + _s20("ms"))

    lines.append(f"/MAT/ELAST/{ids.mat_id}")
    lines.append("elastic")
    lines.append("#              RHO_I")
    lines.append(_f20(rho) + _f20(0))
    lines.append("#                  E                  nu")
    lines.append(_f20(E) + _f20(nu))

    lines.append("/NODE")
    scaled = coords * length_scale
    for nid, (x, y, z) in enumerate(scaled, start=1):
        lines.append(_i10(nid) + _f20(x) + _f20(y) + _f20(z))
    # master nodes on the axis, at each end
    lines.append(_i10(ids.node_bottom_master) + _f20(0) + _f20(0) + _f20(0))
    lines.append(_i10(ids.node_top_master) + _f20(0) + _f20(0) + _f20(L))

    lines.append(f"/PROP/SHELL/{ids.prop_id}")
    lines.append("shell")
    lines.append("#   Ishell    Ismstr     Ish3n    Idrill")
    lines.append(_i10(24) + _i10(2) + _i10(0) + _i10(0))
    lines.append("#                 hm                  hf                  hr                  dm                  dn")
    lines.append(_f20(0) + _f20(0) + _f20(0) + _f20(0) + _f20(0))
    lines.append("#        N   Istrain               Thick              Ashear              Ithick     Iplas")
    # N = through-thickness integration points. This must be >= 3 for a
    # buckling model: with N = 1 the only integration point sits on the
    # mid-surface, so the element carries membrane stress but no through-
    # thickness stress gradient, and the shell's bending stiffness is lost.
    # Shell buckling is sigma_cr ~ sqrt(membrane * bending), so N = 1 knocks
    # the critical load down by more than an order of magnitude (measured:
    # collapse at ~5% of sigma_cl instead of the ~0.5 expected at this R/t).
    lines.append(_i10(n_integ) + _i10(1) + _f20(thickness * length_scale) + _f20(0.8333) + _f20(1) + _i10(1))

    lines.append(f"/PART/{ids.part_id}")
    lines.append("shell_part")
    lines.append(_i10(ids.prop_id) + _i10(ids.mat_id) + _i10(0))

    lines.append(f"/SHELL/{ids.part_id}")
    for eid, (n1, n2, n3, n4) in enumerate(mesh.elements, start=1):
        lines.append(_i10(eid) + _i10(n1 + 1) + _i10(n2 + 1) + _i10(n3 + 1) + _i10(n4 + 1)
                      + _i10(0) + _i10(0) + _i10(0) + _i10(0) + _i10(0))

    bottom_ids = [int(n) + 1 for n in mesh.bottom_nodes]
    top_ids = [int(n) + 1 for n in mesh.top_nodes]

    lines += _grnod_node_block(ids.grnod_bottom_slaves, "bottom_ring", bottom_ids)
    lines += _grnod_node_block(ids.grnod_top_slaves, "top_ring", top_ids)
    lines += _grnod_node_block(ids.grnod_bottom_master, "bottom_master", [ids.node_bottom_master])
    lines += _grnod_node_block(ids.grnod_top_master, "top_master", [ids.node_top_master])

    lines.append(f"/BCS/{ids.bcs_bottom}")
    lines.append("clamp_bottom")
    lines.append("#  Tra rot   skew_ID  grnod_ID")
    lines.append(_s10("111 111") + _i10(0) + _i10(ids.grnod_bottom_master))

    top_tra = "110 111" if impose_velocity is not None else "111 111"
    lines.append(f"/BCS/{ids.bcs_top}")
    lines.append("clamp_top")
    lines.append("#  Tra rot   skew_ID  grnod_ID")
    lines.append(_s10(top_tra) + _i10(0) + _i10(ids.grnod_top_master))

    lines.append(f"/RBODY/{ids.rbody_bottom}")
    lines.append("rbody_bottom")
    lines.append("#     RBID     ISENS     NSKEW    ISPHER                MASS   Gnod_id     IKREM      ICOG   Surf_id")
    lines.append(_i10(ids.rbody_bottom) + _i10(0) + _i10(0) + _i10(0) + _f20(0)
                 + _i10(ids.grnod_bottom_slaves) + _i10(0) + _i10(1) + _i10(0))
    lines.append("#                Jxx                 Jyy                 Jzz")
    lines.append(_f20(0) + _f20(0) + _f20(0))
    lines.append("#                Jxy                 Jyz                 Jxz")
    lines.append(_f20(0) + _f20(0) + _f20(0))
    lines.append("#Ioptoff")
    lines.append(_i10(0))

    lines.append(f"/RBODY/{ids.rbody_top}")
    lines.append("rbody_top")
    lines.append("#     RBID     ISENS     NSKEW    ISPHER                MASS   Gnod_id     IKREM      ICOG   Surf_id")
    lines.append(_i10(ids.rbody_top) + _i10(0) + _i10(0) + _i10(0) + _f20(0)
                 + _i10(ids.grnod_top_slaves) + _i10(0) + _i10(1) + _i10(0))
    lines.append("#                Jxx                 Jyy                 Jzz")
    lines.append(_f20(0) + _f20(0) + _f20(0))
    lines.append("#                Jxy                 Jyz                 Jxz")
    lines.append(_f20(0) + _f20(0) + _f20(0))
    lines.append("#Ioptoff")
    lines.append(_i10(0))

    if impose_velocity is not None:
        if t_ramp is None:
            t_ramp = 0.0
        lines.append(f"/FUNCT/{ids.funct_ramp}")
        lines.append("velocity_ramp")
        lines.append("#                  X                   Y")
        lines.append(_f20(0) + _f20(0))
        lines.append(_f20(t_ramp) + _f20(1))
        lines.append(_f20(t_ramp * 1000.0 + 1.0) + _f20(1))

        lines.append(f"/IMPVEL/{ids.impvel_top}")
        lines.append("axial_compression")
        lines.append("#funct_IDT       Dir   skew_ID sensor_ID  grnod_ID  frame_ID     Icoor")
        lines.append(_i10(ids.funct_ramp) + _s10("Z") + _i10(0) + _i10(0)
                     + _i10(ids.grnod_top_master) + _i10(0) + _i10(0))
        lines.append("#           Ascale_x            Fscale_Y              Tstart               Tstop")
        lines.append(_f20(0) + _f20(-abs(impose_velocity)) + _f20(0) + _f20(1e28))

    lines.append("/END")
    _write_lines(path, lines)


def write_engine(path, run_name, run_no, tend, dt_his=None, anim_dt=None,
                  parith_on=True):
    """Write the Engine deck (`*_0001.rad`). Free-format -- no column rules.

    `/ANIM` output is deliberately just the deformed shape (node coordinates
    are inherent to every animation frame) plus one lightweight coloring
    field (element von Mises stress). A 9-field dense list at anim_dt=tend/100
    measured at ~1.35GB for a single ~91k-node/~91k-element run -- far more
    than the start/end-frame figures actually need. Default `anim_dt=tend`
    writes exactly two frames (t=0, t=tend), matching that need; pass a
    smaller value only if intermediate frames are actually wanted (e.g. QA).
    """
    if dt_his is None:
        dt_his = tend / 200.0
    if anim_dt is None:
        anim_dt = tend
    lines = ["#"]
    if parith_on:
        lines.append("/PARITH/ON")
    lines.append("/PRINT/-100")
    lines.append(f"/RUN/{run_name}/{run_no}")
    lines.append(f"{tend:.9g}")
    lines.append("/TFILE/4")
    lines.append("#            dT_HIS")
    lines.append(f"{dt_his:.9g}")
    lines.append("/TITLE")
    lines.append(run_name)
    lines.append("/ANIM/DT")
    lines.append("#   TSTART     TFREQ")
    lines.append(f"0.0 {anim_dt:.9g}")
    lines.append("/ANIM/ELEM/VONM")
    _write_lines(path, lines)
