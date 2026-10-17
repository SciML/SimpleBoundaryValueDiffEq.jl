# Developer API

`AbstractSimpleMIRK` is the package's documented extension interface. It is intended for
solver developers, not ordinary problem-solving workflows. Subtypes must implement all required
tableau methods below; the package's generic `solve` method handles both in-place and
out-of-place `BVProblem`s.

```@docs
AbstractSimpleMIRK
alg_order
alg_stage
constructSimpleMIRK
```
