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
    truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated})

Create a truncated copy of the BMPO.

Arguments:
- bmpo::BMPO: Input bosonic MPO

Returns:
- BMPO: Truncated bosonic MPO
"""
function ITensorMPS.truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    truncated_mpo = ITensorMPS.truncate(bmpo.mpo; kwargs...)
    return BMPO(truncated_mpo, bmpo.alg)
end

"""
    truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated})

Truncate the BMPO in place.

Arguments:
- bmpo::BMPO: Bosonic MPO to truncate

Returns:
- BMPO: The truncated BMPO (same object, modified in place)
"""
function ITensorMPS.truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    ITensorMPS.truncate!(bmpo.mpo; kwargs...)
    return bmpo
end

"""
    +(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated})

Add two BMPO objects.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Returns:
- BMPO: Sum of the two bosonic MPO
"""
function Base.:+(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated})
    result_mpo = bmpo1.mpo + bmpo2.mpo
    return BMPO(result_mpo, bmpo1.alg)
end

"""
    contract(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated})

Contract a BMPO with a BMPS.

Arguments:
- bmpo::BMPO: Bosonic MPO
- bmps::BMPS: Bosonic MPS

Returns:
- BMPS: Result of MPO-MPS contraction
"""
function ITensors.contract(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated})
    result_mps = ITensors.contract(bmpo.mpo, bmps.mps)
    return BMPS(result_mps, bmps.alg)
end

"""
    apply(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated})
Apply a BMPO to a BMPS.

Arguments:
- bmpo::BMPO: Bosonic MPO to apply
- bmps::BMPS: Bosonic MPS to apply to

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