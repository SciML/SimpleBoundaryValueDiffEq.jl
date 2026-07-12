recursive_length(x::Vector{<:AbstractArray}) = sum(length, x)
function recursive_flatten(x::Vector{<:AbstractArray})
    y = zero(first(x), recursive_length(x))
    recursive_flatten!(y, x)
    return y
end

@views function recursive_flatten!(y::AbstractVector, x::Vector{<:AbstractArray})
    i = 0
    for xᵢ in x
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    return y
end
@views function recursive_flatten_twopoint!(y::AbstractVector, x::Vector{<:AbstractArray}, sizes)
    x_, xiter = Iterators.peel(x)
    copyto!(y[1:prod(sizes[1])], x_[1:prod(sizes[1])])
    i = prod(sizes[1])
    for xᵢ in xiter
        copyto!(y[(i + 1):(i + length(xᵢ))], xᵢ)
        i += length(xᵢ)
    end
    copyto!(y[(i + 1):(i + prod(sizes[2]))], x_[(end - prod(sizes[2]) + 1):end])
    return y
end

@views function recursive_unflatten!(y::Vector{<:AbstractArray}, x::AbstractVector)
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
    return y
end

function __finite_difference_jacobian(f, u)
    fu = f(u)
    J = similar(fu, length(fu), length(u))
    tmp = similar(fu)
    x = copy(u)
    __finite_difference_jacobian_columns!(J, fu, tmp, f, x)
    return J
end

function __finite_difference_jacobian!(J, f!, u)
    x = copy(u)
    fu = similar(u, size(J, 1))
    tmp = similar(fu)
    f!(fu, x)
    __finite_difference_jacobian_columns!(J, fu, tmp, (y) -> f!(tmp, y), x)
    return J
end

function __finite_difference_jacobian_columns!(J, fu, tmp, f, x)
    for j in eachindex(x)
        xj = x[j]
        h = sqrt(eps(Float64)) * max(abs(xj), one(abs(xj)))
        x[j] = xj + h
        fx = f(x)
        for i in eachindex(fu)
            J[i, j] = (fx[i] - fu[i]) / h
        end
        x[j] = xj
    end
    return J
end

function eval_bc_residual!(residual, y, mesh, prob, pt::StandardBVProblem)
    return prob.f.bc(residual, y, prob.p, mesh)
end

function eval_bc_residual(y, mesh, prob, pt::StandardBVProblem)
    return prob.f.bc(y, prob.p, mesh)
end

function eval_bc_residual!(residual, y, t, prob, pt::TwoPointBVProblem)
    length_a = length(prob.f.bcresid_prototype[1])
    length_b = length(prob.f.bcresid_prototype[2])
    ua = y isa ODESolution ? y(first(t)) : y[1]
    ub = y isa ODESolution ? y(last(t)) : y[end]
    @views first(prob.f.bc)(residual[1:length_a], ua, prob.p)
    return @views last(prob.f.bc)(residual[(length_a + 1):(length_a + length_b)], ub, prob.p)
end

function eval_bc_residual(y, t, prob, pt::TwoPointBVProblem)
    ua = y isa ODESolution ? y(first(t)) : y[1]
    ub = y isa ODESolution ? y(last(t)) : y[end]
    return vcat(first(prob.f.bc)(ua, prob.p), last(prob.f.bc)(ub, prob.p))
end
