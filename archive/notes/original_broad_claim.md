## SYMMETRY EROSION RATE AS A LEADING INDICATOR OF CONSERVATION LAW COLLAPSE

A physical or computational system will show measurable degradation in a conserved quantity before that quantity's symmetry is detectably broken, because the rate at which symmetry-violating perturbations accumulate relative to the system's capacity to absorb them determines when conservation stops being approximate and starts being structurally absent.

Standard physics treats symmetry and conservation as a binary pair: either the symmetry holds and the quantity is conserved, or the symmetry breaks and it is not. This is accurate for isolated, Hamiltonian systems, but it misses a key regime in open, noisy, or finite-time systems where the breakdown is gradual and directional.

In those systems, the conserved quantity does not suddenly fail. It decays at a rate proportional to how fast symmetry-violating interactions are being introduced relative to the system's total capacity for absorbing them without restructuring.

This means the ratio of perturbation rate to absorption capacity is a genuine early warning signal, not just a description of failure after the fact. A system near the tipping region of this ratio should exhibit measurable drift in the conserved quantity even while the symmetry, if measured statically at any single moment, still appears intact.

The non-obvious part is the decoupling of timescales: the symmetry can look whole in a snapshot while the conserved quantity is already eroding in the time-series, because symmetry is a global property of the action but conservation is a local, moment-to-moment integral that accumulates errors faster than a single-frame symmetry test would reveal.

This implies that monitoring conservation violation in real time is a strictly more sensitive probe of oncoming symmetry breaking than testing the symmetry directly, reversing the conventional explanatory arrow that runs from symmetry to conservation.

Concrete expected pattern: in neural network training under finite learning rate (where Noether-like conservation laws are known to be broken by discrete steps), the L2 norm of weight increments should drift measurably before any detectable asymmetry appears in the gradient covariance structure, and the drift amplitude should scale with the ratio of learning rate to loss-landscape curvature, not with the learning rate alone.
