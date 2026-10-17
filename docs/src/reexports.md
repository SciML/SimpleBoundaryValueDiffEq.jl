# Reexported SciML common interface

`using SimpleBoundaryValueDiffEq` also brings in the parts of the SciML common interface
needed to construct and solve the boundary value problems supported by this package. These
names are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/):

  - Problems: `BVProblem` and `TwoPointBVProblem`
  - Solutions: `ODESolution`
  - Solving and problem updates: `solve` and `remake`
  - Return status: `ReturnCode` and `successful_retcode`

Anything else from SciMLBase must be imported from SciMLBase directly.

## Solver methods

```@docs
solve(::BVProblem, ::AbstractSimpleMIRK)
solve(::BVProblem, ::SimpleShooting)
```
