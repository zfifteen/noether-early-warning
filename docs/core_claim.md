## Early Warning from Conservation Law Erosion

A physical or computational system can begin to show measurable erosion in a conserved quantity before the underlying symmetry looks detectably broken.

The central idea is that symmetry and conservation do not always become visibly unstable on the same timescale. In an idealized system they rise and fall together. In an open, noisy, finite-time, or discretized system, they can separate. A symmetry can still appear intact in a static snapshot while the associated conserved quantity is already drifting in the time-series.

This makes conservation drift an early warning signal. It is not just evidence that failure has already happened. It can be the first practical sign that the system is moving out of a regime where the conservation law is still functioning as a meaningful approximation.

The important distinction is between what is visible in a moment and what becomes visible only through accumulation. Symmetry is often tested as a structural property of the system at a given instant. Conservation violation is often seen through many small local deviations that build up over time. Because of that difference, the conserved quantity can start showing damage before a direct symmetry test has enough sensitivity to register clear asymmetry.

If this is right, then monitoring conservation erosion in real time is a more sensitive probe of oncoming failure than checking the symmetry directly. The explanatory order also changes in practice. Instead of saying that first we detect symmetry breaking and then infer conservation loss, we can detect conservation loss early and treat it as a warning that direct symmetry breaking is approaching but not yet plainly visible.

The core claim is therefore simple. In systems where symmetry breaking is gradual rather than instantaneous, the earliest reliable evidence of breakdown may appear first in the conserved quantity. The novelty is not just that conservation and symmetry are related, but that conservation erosion may be the earlier and more useful observable.

In neural network training, this means a quantity like the norm of weight increments may begin to show a stable directional drift before any direct probe of symmetry in the gradient structure becomes clearly detectable. The broader implications can be explored later. The core result is the early warning effect itself.
