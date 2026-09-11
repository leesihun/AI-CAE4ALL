# `buckling/` — a static one-to-many shell post-buckling benchmark

A thin cylinder under axial compression buckles into one of many near-degenerate
modes. Which one it picks is decided by an imperfection far too small to matter
mechanically. **We specify the imperfection law and then withhold the
realization**, so the model sees only the nominal geometry and must predict a
*distribution* over post-buckled fields.

That withholding is the whole point. Every existing many-draw dataset in
scientific ML puts the random realization on the **input** side — the initial
condition, the coefficient field, the crack image — which makes the map
deterministic and destroys the one-to-many structure.

```
                 nominal geometry  ─┐
                                    ├──►  p( terminal displacement field )
   imperfection draw  (WITHHELD)  ─┘
```

| | |
|---|---|
| **Design** | [docs/BENCHMARK_DESIGN.md](docs/BENCHMARK_DESIGN.md) — the full spec: imperfection law, OOD split, allocation, gates, risks |
| **Build log** | [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — what has actually been run and what it measured |
| **Code** | [src/](src/) |
| **Scratch** | `work/` (git-ignored) |

---

## Quick start

```bash
cd buckling/src

python run_g1.py --quick     # ~1 min   pipeline validation, one geometry
python run_g1.py             # ~9 min   gate G1 on all four box corners
python run_pilot.py --smoke  # ~30 min  one nonlinear solve, end to end
```

## Requirements

- **CalculiX 2.22** under WSL. There is no Windows build in use here; `src/ccx.py`
  shells out through `wsl -e bash -lc`.
- Python 3.10+, numpy, scipy, h5py.

CalculiX was installed without root as follows (apt needs sudo, which is not
available in this environment):

```bash
curl -O http://www.dhondt.de/ccx_2.22.tar.bz2        # ships a prebuilt binary
tar xjf ccx_2.22.tar.bz2 && cp CalculiX/ccx_2.22/src/ccx_2.22 ~/bin/ccx

# it links against libgfortran.so.4, which Ubuntu 24.04 no longer ships
curl -Lo lgf.deb http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-7/libgfortran4_7.5.0-6ubuntu2_amd64.deb
dpkg-deb -x lgf.deb lgf && cp lgf/usr/lib/x86_64-linux-gnu/libgfortran.so.4 ~/lib/

export LD_LIBRARY_PATH=$HOME/lib
```

Override the binary and library paths with `CCX_BIN` / `CCX_LD` if yours differ.

---

## Code map

| File | Role |
|---|---|
| [src/geometry.py](src/geometry.py) | analytic mapped S8R meshes in float64 — cylinder, cone, axial thickness variation |
| [src/imperfection.py](src/imperfection.py) | the imperfection law: deterministic process signature + band-limited Matérn random field |
| [src/ccx.py](src/ccx.py) | CalculiX `.inp` writers (buckle / static / dynamic), runner, `.dat` and `.frd` readers |
| [src/run_g1.py](src/run_g1.py) | gate G1 — perfect-shell eigenvalue census |
| [src/run_pilot.py](src/run_pilot.py) | gates G2–G5 — the nonlinear pilot |

---

## Four things that will silently ruin this dataset

Each of these produces a plausible-looking run whose imperfection signal is
buried, so **every draw returns the same mode** and the one-to-many claim
evaporates with no error message anywhere.

1. **Writing coordinates at `%.15e`.** CalculiX caps a free-format field at 20
   characters; `%.15e` is 21 and the deck is silently corrupted. Verified: the
   parser rejects `*ELASTIC` with a Poisson-ratio error that has nothing to do
   with the Poisson ratio. Use `%.13e` — 14 significant digits, 20 characters
   with the sign.
2. **A free mesh, or linear elements.** Leaks ~2.1e-4·t of spurious symmetry
   breaking, about 110× the physical signal. Mapped quadratic mesh only.
3. **CalculiX default convergence tolerances** (Rn=0.005, Cn=0.01) sit roughly
   four orders above the signal.
4. **i.i.d. random values at mesh nodes.** The effective correlation length is
   then the element size, so the "law" is not a law: halving the element size
   halves the effective imperfection (measured: exactly 2.00× per halving).
   The field must be band-limited to a *physical* correlation length, which is
   why `imperfection.py` evaluates a truncated Fourier series in closed form at
   the node coordinates and persists the **spectral coefficients** rather than
   nodal values.

## Convention notes

Units are non-dimensional: `R = 1`, `E = 1`, `ν = 0.3`, `t = R/(R/t)`. Every
result scales.

Generated data does **not** live in this directory. Raw solves go to
`work/` (git-ignored); the finished dataset is written into the repo's shared
mesh HDF5 contract under `dataset/`.
