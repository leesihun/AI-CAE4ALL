"""Replace the contributions list with one that matches the implementation.

The original list was written before the code was audited and asserts three
things the source does not do: minibatch optimal-transport coupling in the
field flow (the source draws an independent Gaussian), a bi-stride hierarchy
(production uses farthest-point sampling plus Voronoi), and a variogram score
(not implemented anywhere). It also described a several-hundred-step rollout,
while the reported task is static.
"""
import io

OLD_START = r"\begin{enumerate}"
OLD_END = r"\end{enumerate}"

NEW = r"""\begin{enumerate}
  \item \textbf{Two mesh-native probabilistic hierarchical simulators sharing
    one backbone.} MeshGraphNets-V places per-level graph latents inside the
    hierarchical processor, regularises them with an MMD-InfoVAE term, and
    learns their conditional prior by flow matching jointly with the simulator.
    cHI-MGNflow removes the latent bottleneck and flow-matches the field
    itself, conditioned hierarchically, with the flow time drawn per graph and
    the data-prediction parameterisation expressed exactly as a loss weight.
    Because they differ only in where the stochasticity enters, the comparison
    isolates that choice (\S\ref{sec:method}).
  \item \textbf{A head-to-head on industrial warpage data showing two
    \emph{orthogonal} failure modes} (\S\ref{sec:results:saoi}).
    MeshGraphNets-V wins by $2.2\times$ with no overlap across an eight-arm
    design each, but is correctly centred and roughly $2\times$ too narrow,
    while cHI-MGNflow fails on one part family through an almost pure mean
    offset whose sign flips between families. Neither model is calibrated,
    which the ranking alone does not reveal.
  \item \textbf{A proof that fitting a flow-matching prior to a jointly trained
    encoder does not supply a variational rate term}
    (\S\ref{sec:method:shrink}). The optimal loss floor is linear in the
    trainable target scale, so the objective can always be reduced by shrinking
    the target, while the corresponding KL divergence vanishes at every scale.
    We also measure the effect on this task and find it nil, which rules the
    coupling out as a remedy without claiming it explains the deficit.
  \item \textbf{A purpose-built benchmark whose withheld variable is known by
    construction} (\S\ref{sec:results:buckling}): axially compressed shells in
    which a specified band-limited imperfection of RMS $10^{-3}t$---$150\times$
    smaller than the signature the model can see---decides which post-buckled
    branch is reached, over $22$ geometries in four tiers that separate
    parameter extrapolation from structural novelty. Characterisation before
    any training establishes that the conditional law genuinely varies with
    geometry, and shows that the discrete mode label is a \emph{lossy} target
    relative to the field.
  \item \textbf{A scoring protocol that is proper by construction}
    (\S\ref{sec:protocol}): fair, ensemble-size-corrected energy score and
    CRPS, reported against both a finite-sample self floor and a zero-spread
    reference, with dispersion ratio and rank histograms alongside because a
    single proper score cannot separate under-dispersion from bias.
  \item \textbf{An audit of the available benchmark landscape}
    (\S\ref{sec:bench}), establishing which datasets can and cannot support a
    distributional claim, including a withholding-based construction with its
    information leak quantified, and two negative results on candidate slots.
\end{enumerate}"""


def main():
    p = "introduction.tex"
    s = io.open(p, encoding="utf-8").read()
    key = r"\subsection{Contributions}"
    i = s.index(key)
    a = s.index(OLD_START, i)
    b = s.index(OLD_END, a) + len(OLD_END)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s[:a] + NEW + s[b:])
    print("contributions replaced (%d -> %d chars)" % (b - a, len(NEW)))


if __name__ == "__main__":
    main()
