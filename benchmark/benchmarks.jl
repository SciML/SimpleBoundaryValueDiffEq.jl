using SimpleBoundaryValueDiffEq, BenchmarkTools
using SciMLBase: TwoPointBVProblem

const SUITE = BenchmarkGroup()

# Linear BVP: y'' + y = 0, y(0)=0, y(1)=1
function f_bvp!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end
bca!(res, u, p) = (res[1] = u[1])
bcb!(res, u, p) = (res[1] = u[1] - 1.0)

tspan = (0.0, 1.0)
u0 = [0.0, 1.0]
u0_guess = [[0.0, 1.0], [1.0, 0.0]]
bcp = (zeros(1), zeros(1))

prob_mirk = TwoPointBVProblem(
    f_bvp!, (bca!, bcb!), u0, tspan; bcresid_prototype = bcp
)
prob_shoot = TwoPointBVProblem(
    f_bvp!, (bca!, bcb!), u0, tspan; bcresid_prototype = bcp
)

# =============================================================================
# MIRK solves
# =============================================================================

SUITE["mirk"] = BenchmarkGroup()

SUITE["mirk"]["SimpleMIRK4"] = @benchmarkable solve($prob_mirk, SimpleMIRK4(); dt = 0.05)
SUITE["mirk"]["SimpleMIRK5"] = @benchmarkable solve($prob_mirk, SimpleMIRK5(); dt = 0.05)
SUITE["mirk"]["SimpleMIRK6"] = @benchmarkable solve($prob_mirk, SimpleMIRK6(); dt = 0.05)

# =============================================================================
# Single shooting
# =============================================================================

SUITE["shooting"] = BenchmarkGroup()

SUITE["shooting"]["SimpleShooting"] = @benchmarkable solve(
    $prob_shoot, SimpleShooting()
)

# =============================================================================
# Construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()
SUITE["construct"]["twopoint"] = @benchmarkable TwoPointBVProblem(
    $f_bvp!, ($bca!, $bcb!), $u0, $tspan; bcresid_prototype = $bcp
)
