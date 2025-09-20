"""
Bosonic Matrix Product State wrapper that supports different algorithms.
Contains an underlying ITensorMPS.MPS object and algorithm specification.

Fields:
- mps::M: The underlying ITensorMPS.MPS object
- alg::A: Algorithm specification (Truncated, PseudoSite, or LocalBasis)
"""
mutable struct BMPS{M<:ITensorMPS.MPS,A<:MabsAlg}
    mps::M
    alg::A
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::Truncated)

Create a BMPS from an existing MPS using the Truncated algorithm.

Arguments:
- mps::ITensorMPS.MPS: Input matrix product state
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Wrapped bosonic MPS
"""
function BMPS(mps::ITensorMPS.MPS, alg::Truncated)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

ITensorMPS.siteinds(bmps::BMPS) = ITensorMPS.siteinds(bmps.mps)
ITensorMPS.maxlinkdim(bmps::BMPS) = ITensorMPS.maxlinkdim(bmps.mps)
ITensorMPS.linkind(bmps::BMPS, i::Int) = ITensorMPS.linkind(bmps.mps, i)
ITensorMPS.siteind(bmps::BMPS, i::Int) = ITensorMPS.siteind(bmps.mps, i)
Base.eltype(bmps::BMPS) = eltype(bmps.mps[1])  
Base.length(bmps::BMPS) = length(bmps.mps)

const TRUNCATED_FORWARDED_FUNCTIONS = [
    :(ITensorMPS.findsite),
    :(ITensorMPS.findsites),
    :(ITensorMPS.firstsiteinds),
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensorMPS.expect),
    :(ITensors.dag),
    :(ITensorMPS.inner),
    :(LinearAlgebra.dot),
    :(ITensorMPS.loginner),
    :(ITensorMPS.logdot),
    :(LinearAlgebra.norm),
    :(ITensorMPS.lognorm),
    :(Base.collect),
    :(Base.length),
    :(Base.size)
]

for f in TRUNCATED_FORWARDED_FUNCTIONS
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, args...; kwargs...) = ($f)(bmps.mps, args...; kwargs...)
end

"""
    copy(bmps::BMPS)

Create a shallow copy of the BMPS.

Arguments:
- bmps::BMPS: Input bosonic MPS

Returns:
- BMPS: Copied bosonic MPS
"""
Base.copy(bmps::BMPS) = BMPS(copy(bmps.mps), bmps.alg)

"""
    deepcopy(bmps::BMPS)

Create a deep copy of the BMPS.

Arguments:
- bmps::BMPS: Input bosonic MPS

Returns:
- BMPS: Deep copied bosonic MPS
"""
Base.deepcopy(bmps::BMPS) = BMPS(deepcopy(bmps.mps), bmps.alg)

"""
    normalize!(bmps::BMPS{<:ITensorMPS.MPS,Truncated})

Normalize the BMPS in place.

Arguments:
- bmps::BMPS: Bosonic MPS to normalize

Returns:
- BMPS: The normalized BMPS (same object, modified in place)
"""
function LinearAlgebra.normalize!(bmps::BMPS{<:ITensorMPS.MPS,Truncated})
    LinearAlgebra.normalize!(bmps.mps)
    return bmps
end

"""
    normalize(bmps::BMPS{<:ITensorMPS.MPS,Truncated})

Create a normalized copy of the BMPS.

Arguments:
- bmps::BMPS: Input bosonic MPS

Returns:
- BMPS: Normalized bosonic MPS
"""
function LinearAlgebra.normalize(bmps::BMPS{<:ITensorMPS.MPS,Truncated})
    normalized_mps = LinearAlgebra.normalize(bmps.mps)
    return BMPS(normalized_mps, bmps.alg)
end

"""
    orthogonalize!(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, j::Int)

Orthogonalize the BMPS to site j in place.

Arguments:
- bmps::BMPS: Bosonic MPS to orthogonalize
- j::Int: Site to orthogonalize to

Returns:
- BMPS: The orthogonalized BMPS (same object, modified in place)
"""
function ITensorMPS.orthogonalize!(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, j::Int; kwargs...)
    ITensorMPS.orthogonalize!(bmps.mps, j; kwargs...)
    return bmps
end

"""
    orthogonalize(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, j::Int)

Create an orthogonalized copy of the BMPS to site j.

Arguments:
- bmps::BMPS: Input bosonic MPS
- j::Int: Site to orthogonalize to

Returns:
- BMPS: Orthogonalized bosonic MPS
"""
function ITensorMPS.orthogonalize(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, j::Int; kwargs...)
    orthog_mps = ITensorMPS.orthogonalize(bmps.mps, j; kwargs...)
    return BMPS(orthog_mps, bmps.alg)
end

"""
    truncate(bmps::BMPS{<:ITensorMPS.MPS,Truncated})

Create a truncated copy of the BMPS.

Arguments:
- bmps::BMPS: Input bosonic MPS

Returns:
- BMPS: Truncated bosonic MPS
"""
function ITensorMPS.truncate(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    truncated_mps = ITensorMPS.truncate(bmps.mps; kwargs...)
    return BMPS(truncated_mps, bmps.alg)
end

"""
    truncate!(bmps::BMPS{<:ITensorMPS.MPS,Truncated})

Truncate the BMPS in place.

Arguments:
- bmps::BMPS: Bosonic MPS to truncate

Returns:
- BMPS: The truncated BMPS (same object, modified in place)
"""
function ITensorMPS.truncate!(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    ITensorMPS.truncate!(bmps.mps; kwargs...)
    return bmps
end

"""
    +(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})

Add two BMPS objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Returns:
- BMPS: Sum of the two bosonic MPS
"""
function Base.:+(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})
    result_mps = bmps1.mps + bmps2.mps
    return BMPS(result_mps, bmps1.alg)
end

"""
    contract(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})

Contract two BMPS objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Returns:
- Result of contraction (typically a scalar)
"""
function ITensors.contract(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})
    return ITensors.contract(bmps1.mps, bmps2.mps)
end

"""
    outer(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})

Compute outer product of two BMPS objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Returns:
- Outer product result
"""
function ITensorMPS.outer(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})
    outer_result = ITensorMPS.outer(bmps1.mps, bmps2.mps)
    return BMPO(outer_result, bmps1.alg)
end
Base.iterate(bmps::BMPS) = Base.iterate(bmps.mps)
Base.iterate(bmps::BMPS, state) = Base.iterate(bmps.mps, state)
Base.eachindex(bmps::BMPS) = Base.eachindex(bmps.mps)
Base.getindex(bmps::BMPS, i) = bmps.mps[i]
Base.setindex!(bmps::BMPS, val, i) = (bmps.mps[i] = val)
Base.firstindex(bmps::BMPS) = Base.firstindex(bmps.mps)
Base.lastindex(bmps::BMPS) = Base.lastindex(bmps.mps)

function LinearAlgebra.dot(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})
    return LinearAlgebra.dot(bmps1.mps, bmps2.mps)
end

function ITensorMPS.inner(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated})
    return ITensorMPS.inner(bmps1.mps, bmps2.mps)
end