# SimpleBoundaryValueDiffEq.jl

SimpleBoundaryValueDiffEq.jl provides lightweight boundary value problem algorithms for the
SciML common solve interface. Construct a `BVProblem` or `TwoPointBVProblem` with SciMLBase,
then select a MIRK collocation method or single-shooting method below.

## Algorithms

```@docs
SimpleMIRK4
SimpleMIRK5
SimpleMIRK6
SimpleShooting
```
