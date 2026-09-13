"""Rewrite the benchmark characterisation subsection against the full train tier.

The first version was written when four geometries had completed. The finished
train tier (twelve geometries, 576 draws) changes three of its numbers and adds
a convergence limitation that the easy geometries had hidden.
"""
import io

NEW = r"""\subsubsection{Characterisation}

The training tier is complete: $12$ geometries at $48$ draws each, $576$ draws,
with \emph{zero} solver failures out of $588$ attempted including the tier that
follows.

Figure~\ref{fig:buckling-fields} shows three realisations of one geometry. They
are the same shell under the same load with the same visible inputs, and they
buckle into different circumferential modes; the profile panel makes the
wavenumbers directly countable. This is the one-to-many property the benchmark
exists to supply.

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{figures/buckling_fields.pdf}
\caption{Three draws from one geometry, differing only in the withheld
Component~B realisation. Top: radial displacement $w/t$ over the unrolled
mid-surface. Bottom: circumferential profile at the most-deformed axial
station, where the different wavenumbers are directly countable.}
\label{fig:buckling-fields}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/buckling_modes.pdf}
\caption{Measured mode distributions at fixed $L/R=1.0$, $48$ draws per
geometry. The dashed line marks the classical prediction
$n_{\mathrm{cr}}=0.86\sqrt{R/t}$. The distribution both \emph{shifts} with
$R/t$ and \emph{widens}, and the classical value drifts progressively to the
right of the realised mode.}
\label{fig:buckling-modes}
\end{figure}

Figure~\ref{fig:buckling-modes} reports the mode distributions. Every geometry
supports three to five distinct modes, with a top-mode share between $38\%$ and
$67\%$, so no geometry is close to deterministic. Two trends are visible across
the tier and both are physically expected.

\paragraph{The distribution widens as the shell thins.}
The top-mode share falls from $65$--$67\%$ at $R/t=110$ to $38$--$56\%$ at
$R/t=200$, with a correlation of $-0.68$ against $R/t$. Thinner shells have a
denser cluster of near-degenerate modes near the critical load, so the withheld
field has more nearly equivalent branches to choose between. The benchmark
therefore contains a built-in difficulty axis: the conditional law is broader
exactly where the shell is thinner.

\paragraph{The classical wavenumber over-predicts, systematically.}
The mean realised mode lies below $0.86\sqrt{R/t}$ at \emph{every one} of the
twelve geometries, by $1.6\%$ to $13.0\%$, and the deficit grows with $R/t$
(correlation $+0.59$ on its magnitude). The direction is the expected one---the
closed form is derived for a perfect shell, whereas these are imperfect and
buckle early into a marginally longer wavelength---but the size of the
deviation at the thin end means the classical value should be treated as a
reference line, not as a label the data is expected to reproduce.

\paragraph{Convergence is not uniform across the box, and this is a limitation.}
The settle window was calibrated on $R/t=110$, $L/R=1.0$ and does not transfer
to the long shells. Grouping the retention-threshold exceedance by aspect ratio
makes the pattern unambiguous:

\begin{center}
\begin{tabular}{lccc}
\toprule
$L/R$ & median drift & draws over threshold \\
\midrule
$0.8$ & $0.021$--$0.034$ & $0\%$ at every $R/t$ \\
$1.0$ & $0.020$--$0.116$ & $0$--$21\%$ \\
$1.3$ & $0.038$--$0.191$ & $0$--$81\%$ \\
\bottomrule
\end{tabular}
\end{center}

The driver is $L/R$, not $R/t$: the correlation of exceedance with $R/t$ is only
$0.28$, while every $L/R=0.8$ geometry converges completely and two of the three
$L/R=1.3$ geometries at higher $R/t$ exceed the threshold on more than half
their draws. An earlier frequency extraction had found the fundamental frequency
of the \emph{undeformed} shell nearly constant across the box (spread
$1.014\times$), which is what motivated a single fixed settle window; that
measurement does not govern the \emph{post-buckled} relaxation timescale, and
the present data shows the two are not interchangeable. The $L/R=1.3$ rows
should therefore be regarded as under-converged at the current settle length,
their drift value is stored per draw so the threshold can be re-applied, and the
settle window needs to scale with $L/R$ in any regeneration. The relationship
between drift and near-degeneracy persists across the full tier (correlation
$-0.54$ with the top-mode share), so the draws that fail to settle are
preferentially the marginal ones---which is why they are retained and flagged
rather than silently dropped.
"""


def main():
    p = "experiments.tex"
    s = io.open(p, encoding="utf-8").read()
    a = s.index(r"\subsubsection{Characterisation}")
    b = s.index(r"\subsubsection{The conditional law does depend on geometry}")
    io.open(p, "w", encoding="utf-8", newline="\n").write(s[:a] + NEW + "\n" + s[b:])
    print("characterisation rewritten (%d -> %d chars)" % (b - a, len(NEW)))


if __name__ == "__main__":
    main()
