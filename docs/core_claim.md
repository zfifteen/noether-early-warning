## Early Warning from Drift Before Direct Symmetry Detection

In regimes where symmetry breaking is gradual rather than instantaneous, a drift observable can become detectable before a direct symmetry observable does.

The central idea is that these two kinds of signal do not always become visible on the same timescale. A direct symmetry observable may remain below threshold while a drift observable linked to the same breakdown process is already detectable in the time-series. In this repository, the drift channel is a rolling update-norm onset detector and the direct channel is covariance mismatch across paired units.

This makes drift an early warning signal in a precise operational sense. It is not only evidence that breakdown is occurring. It can be the first detectable sign, while the direct symmetry observable is still not yet plainly visible.

The important distinction is between process observables and direct structural detection. A drift observable can become visible through persistent deviation over time. A direct symmetry observable may require a larger instantaneous separation before it crosses threshold. Because of that difference, drift can become detectable first.

The stronger supported implication is an operational asymmetry in detectability: in the benchmarked gradual regime, the update process carries an earlier warning horizon than the direct symmetry metric.

This repository does not claim a universal law or a first-principles proof about all Noether-conserved quantities. It does not directly compare a conserved quantity with its derivative. It benchmarks a narrower claim in a controlled paired-MLP regime.

The core claim therefore breaks into four parts.

First, in a gradual-breaking regime, drift can become detectable before direct symmetry detection.

Second, this ordering is not generic. In an instant-break regime, direct symmetry detection can appear at or before drift.

Third, under a fixed practical observation budget, drift can be the more sensitive detector.

Fourth, at the time the drift alarm fires, the direct symmetry observable can still remain below its own detection threshold.

The novelty is not only that drift and symmetry are related. It is that a process observable may be the earlier and more practically useful observable under gradual symmetry breaking.

This repository benchmarks those four claims directly.
