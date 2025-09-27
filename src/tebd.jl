"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm (non-mutating version).
"""
function tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)

Perform time evolution using TEBD algorithm with single gate (non-mutating version).
"""
function tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)
    evolved_mps = ITensors.apply(gate, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm.

Arguments:
- psi::BMPS: Input bosonic MPS
- H::BMPO: Hamiltonian as bosonic MPO
- dt::Number: Time step

Keyword Arguments:
- kwargs...: Additional parameters passed to ITensorMPS.tdvp

Returns:
- BMPS: Time-evolved bosonic MPS
"""
function tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)
    evolved_mps = ITensorMPS.tdvp(H.mpo, dt, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

#helper funcs
"""
    _validate_site_index(site_idx::Int, total_sites::Int)

Validate that a site index is within valid range.
"""
function _validate_site_index(site_idx::Int, total_sites::Int)
    if !(1 <= site_idx <= total_sites)
        throw(ArgumentError("Site index $site_idx out of range [1, $total_sites]"))
    end
end

"""
    _build_number_gate(site::ITensors.Index, ω::Real, dt::Real)

Build evolution gate for H = ω*n using direct construction.
"""
function _build_number_gate(site::ITensors.Index, ω::Real, dt::Real)
    dim = ITensors.dim(site)
    gate = ITensors.ITensor(ComplexF64, site', site)
    
    for n in 0:(dim-1)
        phase = exp(-1im * ω * dt * n)
        gate[site'=>(n+1), site=>(n+1)] = phase
    end
    
    return gate
end

"""
    _build_hopping_gate(site1::ITensors.Index, site2::ITensors.Index, J::Real, dt::Real)

Build evolution gate for H = J*(a†b + ab†) using matrix exponentiation.
"""
function _build_hopping_gate(site1::ITensors.Index, site2::ITensors.Index, J::Real, dt::Real)
    dim1, dim2 = ITensors.dim(site1), ITensors.dim(site2)
    
    H_matrix = zeros(ComplexF64, dim1*dim2, dim1*dim2)
    
    for n1 in 0:(dim1-1), n2 in 0:(dim2-1)
        idx_in = n1*dim2 + n2 + 1
        
        if n2 > 0 && n1 < dim1-1
            idx_out = (n1+1)*dim2 + (n2-1) + 1
            H_matrix[idx_out, idx_in] += J * sqrt((n1+1) * n2)
        end
        
        if n1 > 0 && n2 < dim2-1
            idx_out = (n1-1)*dim2 + (n2+1) + 1
            H_matrix[idx_out, idx_in] += J * sqrt(n1 * (n2+1))
        end
    end
    
    U_matrix = exp(-1im * dt * H_matrix)
    
    gate = ITensors.ITensor(ComplexF64, site1', site2', site1, site2)
    
    for n1 in 0:(dim1-1), n2 in 0:(dim2-1)
        for m1 in 0:(dim1-1), m2 in 0:(dim2-1)
            idx_in = n1*dim2 + n2 + 1
            idx_out = m1*dim2 + m2 + 1
            
            gate[site1'=>(m1+1), site2'=>(m2+1), site1=>(n1+1), site2=>(n2+1)] = U_matrix[idx_out, idx_in]
        end
    end
    
    return gate
end

"""
    _build_kerr_gate(site::ITensors.Index, χ::Real, dt::Real)

Build evolution gate for H = χ*n² using direct construction.
"""
function _build_kerr_gate(site::ITensors.Index, χ::Real, dt::Real)
    dim = ITensors.dim(site)
    gate = ITensors.ITensor(ComplexF64, site', site)
    
    for n in 0:(dim-1)
        phase = exp(-1im * χ * dt * n^2)
        gate[site'=>(n+1), site=>(n+1)] = phase
    end
    
    return gate
end