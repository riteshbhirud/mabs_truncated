"""
    random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; kwargs...)

Create a random bosonic MPS using the Truncated algorithm.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Random bosonic MPS
"""
function random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; kwargs...)
    new_kwargs = Dict{Symbol, Any}()
    
    for (key, value) in kwargs
        if key == :linkdim
            new_kwargs[:linkdims] = value
        else
            new_kwargs[key] = value
        end
    end
    
    mps = ITensorMPS.randomMPS(sites; new_kwargs...)
    return BMPS(mps, alg)
end

"""
    bosonic_product_mps(sites::Vector{<:ITensors.Index}, states::Vector, alg::Truncated)

Create a product state BMPS from a vector of local state specifications.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- states::Vector: Vector of state specifications (integers for Fock states)
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Product state bosonic MPS
"""
function bosonic_product_mps(sites::Vector{<:ITensors.Index}, states::Vector, alg::Truncated)
    mps = ITensorMPS.productMPS(sites, states)
    return BMPS(mps, alg)
end

"""
    bosonic_mpo_from_opsum(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a bosonic MPO from an OpSum using the Truncated algorithm.

Arguments:
- opsum::ITensors.OpSum: Operator sum specification
- sites::Vector{<:ITensors.Index}: Vector of site indices  
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Bosonic MPO constructed from OpSum
"""
function bosonic_mpo_from_opsum(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, alg)
end

"""
    bosonic_sites(N::Int, max_occ::Int; conserve_qns::Bool=false)

Create a vector of bosonic site indices with proper SiteType.

Arguments:
- N::Int: Number of sites
- max_occ::Int: Maximum occupation number per site
- conserve_qns::Bool: Whether to conserve quantum numbers (default: false)

Returns:
- Vector{ITensors.Index}: Vector of bosonic site indices
"""
function bosonic_sites(N::Int, max_occ::Int; conserve_qns::Bool=false)
    if conserve_qns
        return ITensors.siteinds("Boson", N; dim=max_occ+1, conserve_qns=true)
    else
        return ITensors.siteinds("Boson", N; dim=max_occ+1)
    end
end

"""
    vacuum_mps(sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a vacuum state |0,0,...,0⟩ BMPS.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Vacuum state bosonic MPS
"""
function vacuum_mps(sites::Vector{<:ITensors.Index}, alg::Truncated)
    states = fill(1, length(sites))
    return bosonic_product_mps(sites, states, alg)
end

"""
    coherent_state_mps(sites::Vector{<:ITensors.Index}, α::Number, alg::Truncated)
    coherent_state_mps(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::Truncated)

Create an approximate coherent state BMPS using truncated expansion.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- α::Number: Single coherent state amplitude (applied to all modes)
- αs::Vector{<:Number}: Vector of coherent state amplitudes (one per mode)
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Coherent state bosonic MPS (approximated by truncation)
"""
function coherent_state_mps(sites::Vector{<:ITensors.Index}, α::Number, alg::Truncated)
    αs = fill(α, length(sites))
    return coherent_state_mps(sites, αs, alg)
end

function coherent_state_mps(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::Truncated)
    N = length(sites)
    length(αs) == N || error("Number of amplitudes ($(length(αs))) must match number of sites ($N)")
    tensors = ITensors.ITensor[]
    for (i, site) in enumerate(sites)
        α = αs[i]
        max_occ = ITensors.dim(site) - 1
        coeffs = ComplexF64[]
        normalization = exp(-abs2(α)/2)
        for n in 0:max_occ
            coeff = normalization * (α^n) / sqrt(safe_factorial(n))
            push!(coeffs, convert(ComplexF64, coeff))
        end
        norm_factor = sqrt(sum(abs2, coeffs))
        coeffs ./= norm_factor
        if i == 1
            if N == 1
                tensor = ITensors.ITensor(ComplexF64, site)
                for n in 0:max_occ
                    tensor[n+1] = coeffs[n+1]
                end
            else
                right_link = ITensors.Index(1, "Link,l=$i")
                tensor = ITensors.ITensor(ComplexF64, site, right_link)
                for n in 0:max_occ
                    tensor[n+1, 1] = coeffs[n+1]
                end
            end
        elseif i == N
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            tensor = ITensors.ITensor(ComplexF64, left_link, site)
            for n in 0:max_occ
                tensor[1, n+1] = coeffs[n+1]
            end
        else
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            right_link = ITensors.Index(1, "Link,l=$i")
            tensor = ITensors.ITensor(ComplexF64, left_link, site, right_link)
            for n in 0:max_occ
                tensor[1, n+1, 1] = coeffs[n+1]
            end
        end
        push!(tensors, tensor)
    end
    mps = ITensorMPS.MPS(tensors)
    return BMPS(mps, alg)
end
