"""
Bosonic Matrix Product Operator wrapper that supports different algorithms.
Contains an underlying ITensorMPS.MPO object and algorithm specification.

Fields:
- mpo::M: The underlying ITensorMPS.MPO object
- alg::A: Algorithm specification (Truncated, PseudoSite, or LocalBasis)
"""
struct BMPO{M<:ITensorMPS.MPO,A<:MabsAlg}
    mpo::M
    alg::A
end

"""
    BMPO(mpo::ITensorMPS.MPO, alg::Truncated)

Create a BMPO from an existing MPO using the Truncated algorithm.

Arguments:
- mpo::ITensorMPS.MPO: Input matrix product operator
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Wrapped bosonic MPO
"""
function BMPO(mpo::ITensorMPS.MPO, alg::Truncated)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a BMPO directly from an OpSum and sites using the Truncated algorithm.

Arguments:
- opsum::ITensors.OpSum: Operator sum specification
- sites::Vector{<:ITensors.Index}: Vector of site indices  
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Bosonic MPO constructed from OpSum
"""
function BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

ITensorMPS.siteinds(bmpo::BMPO) = ITensorMPS.siteinds(bmpo.mpo)
ITensorMPS.maxlinkdim(bmpo::BMPO) = ITensorMPS.maxlinkdim(bmpo.mpo)
ITensorMPS.linkind(bmpo::BMPO, i::Int) = ITensorMPS.linkind(bmpo.mpo, i)
ITensorMPS.siteind(bmpo::BMPO, i::Int) = ITensorMPS.siteind(bmpo.mpo, i)
Base.eltype(bmpo::BMPO) = eltype(bmpo.mpo)
Base.length(bmpo::BMPO) = length(bmpo.mpo)

for f in TRUNCATED_FORWARDED_FUNCTIONS
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}) = ($f)(bmpo.mpo)
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, args...; kwargs...) = ($f)(bmpo.mpo, args...; kwargs...)
end

"""
    copy(bmpo::BMPO)

Create a shallow copy of the BMPO.

Arguments:
- bmpo::BMPO: Input bosonic MPO

Returns:
- BMPO: Copied bosonic MPO
"""
Base.copy(bmpo::BMPO) = BMPO(copy(bmpo.mpo), bmpo.alg)

"""
    deepcopy(bmpo::BMPO)

Create a deep copy of the BMPO.

Arguments:
- bmpo::BMPO: Input bosonic MPO

Returns:
- BMPO: Deep copied bosonic MPO
"""
Base.deepcopy(bmpo::BMPO) = BMPO(deepcopy(bmpo.mpo), bmpo.alg)

"""
    truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Create a truncated copy of the BMPO.

Arguments:
- bmpo::BMPO: Input bosonic MPO

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)

Returns:
- BMPO: Truncated bosonic MPO
"""
function ITensorMPS.truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    truncated_mpo = ITensorMPS.truncate(bmpo.mpo; kwargs...)
    return BMPO(truncated_mpo, bmpo.alg)
end

"""
    truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Truncate the BMPO in place.

Arguments:
- bmpo::BMPO: Bosonic MPO to truncate

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate!
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)

Returns:
- BMPO: The truncated BMPO (same object, modified in place)
"""
function ITensorMPS.truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    ITensorMPS.truncate!(bmpo.mpo; kwargs...)
    return bmpo
end

"""
    +(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Add two BMPO objects with optional truncation.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)

Returns:
- BMPO: Sum of the two bosonic MPO

Note: Adding MPOs with bond dimensions D1 and D2 creates bond dimension D1+D2.
Use truncation kwargs to control the resulting bond dimension.
"""
function Base.:+(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    result_mpo = bmpo1.mpo + bmpo2.mpo
    if !isempty(kwargs)
        result_mpo = ITensorMPS.truncate(result_mpo; kwargs...)
    end
    return BMPO(result_mpo, bmpo1.alg)
end

"""
    add(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Add two BMPO objects using ITensorMPS's add function with truncation control.

This extends ITensorMPS.add for BMPO types, providing a cleaner interface than `+` when you need to specify truncation parameters.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.add
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)

Returns:
- BMPO: Sum of the two bosonic MPO with controlled bond dimension

Examples:
```julia
H_total = add(H1, H2; maxdim=100, cutoff=1e-10)

# Using + operator (no truncation control)  
H_total = H1 + H2
```

Note: This function uses ITensorMPS.add internally, which handles bond dimension
growth (D1 + D2 → controlled dimension) through TTSVD truncation.
"""
function add(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    result_mpo = add(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(result_mpo, bmpo1.alg)
end

"""
    contract(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Contract a BMPO with a BMPS with optional truncation control.

Arguments:
- bmpo::BMPO: Bosonic MPO
- bmps::BMPS: Bosonic MPS

Keyword Arguments:
- kwargs...: Contraction and truncation parameters passed to ITensors.contract
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, alg)

Returns:
- BMPS: Result of MPO-MPS contraction
"""
function ITensors.contract(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    result_mps = ITensors.contract(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

"""
    apply(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Apply a BMPO to a BMPS with optional truncation control.

Arguments:
- bmpo::BMPO: Bosonic MPO to apply
- bmps::BMPS: Bosonic MPS to apply to

Keyword Arguments:
- kwargs...: Parameters passed to ITensors.apply
  (e.g., maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff, apply_dag)

Returns:
- BMPS: Result of applying MPO to MPS
"""
function ITensors.apply(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    result_mps = ITensors.apply(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

Base.iterate(bmpo::BMPO) = Base.iterate(bmpo.mpo)
Base.iterate(bmpo::BMPO, state) = Base.iterate(bmpo.mpo, state)
Base.eachindex(bmpo::BMPO) = Base.eachindex(bmpo.mpo)
Base.getindex(bmpo::BMPO, i) = bmpo.mpo[i]
Base.setindex!(bmpo::BMPO, val, i) = (bmpo.mpo[i] = val)
Base.firstindex(bmpo::BMPO) = Base.firstindex(bmpo.mpo)
Base.lastindex(bmpo::BMPO) = Base.lastindex(bmpo.mpo)