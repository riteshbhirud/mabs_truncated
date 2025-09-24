"""
Time Evolution Block Decimation (TEBD) functions for bosonic MPS.
Production-ready implementation using direct mathematical gate construction.
"""

"""
    tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm (in-place version).
"""
function tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)
    psi.mps = ITensors.apply(gates, psi.mps; kwargs...)
    return psi
end

"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm (non-mutating version).
"""
function tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)

Perform time evolution using TEBD algorithm with single gate (in-place version).
"""
function tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)
    psi.mps = ITensors.apply(gate, psi.mps; kwargs...)
    return psi
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
    build_evolution_gate(sites::Vector{<:ITensors.Index}, gate_type::String, params::NamedTuple, dt::Real)

Build a single evolution gate exp(-i*dt*H) using direct mathematical construction.
This is the primary, production-ready interface for gate construction.

# Arguments
- `sites::Vector{<:ITensors.Index}`: Vector of site indices
- `gate_type::String`: Type of gate ("number", "hopping", "kerr")
- `params::NamedTuple`: Parameters for the specific gate type
- `dt::Real`: Time step

# Gate Types and Parameters
- `"number"`: Single-site number operator evolution
  - `params = (site=i, omega=ω)` → H = ω*n_i
- `"hopping"`: Two-site hopping evolution  
  - `params = (sites=[i,j], J=J)` → H = J*(a†_i*a_j + a_i*a†_j)
- `"kerr"`: Single-site Kerr nonlinearity
  - `params = (site=i, chi=χ)` → H = χ*n_i²

# Examples
```julia
sites = bosonic_sites(4, 6)
dt = 0.01

# Number operator gate: H = ω*n₁
number_gate = build_evolution_gate(sites, "number", (site=1, omega=1.0), dt)

# Hopping gate: H = J*(a†₁a₂ + a₁a†₂)  
hopping_gate = build_evolution_gate(sites, "hopping", (sites=[1,2], J=0.5), dt)

# Kerr gate: H = χ*n₁²
kerr_gate = build_evolution_gate(sites, "kerr", (site=1, chi=0.1), dt)
```
"""
function build_evolution_gate(sites::Vector{<:ITensors.Index}, gate_type::String, params::NamedTuple, dt::Real)
    if isempty(sites)
        throw(ArgumentError("sites vector cannot be empty"))
    end
    
    if !isfinite(dt)
        throw(DomainError(dt, "time step dt must be finite"))
    end
    
    if gate_type == "number"
        site_idx = params.site
        ω = get(params, :omega, 1.0)
        _validate_site_index(site_idx, length(sites))
        return _build_number_gate(sites[site_idx], ω, dt)
        
    elseif gate_type == "hopping"
        site_indices = params.sites
        J = get(params, :J, 1.0)
        
        if length(site_indices) != 2
            throw(ArgumentError("hopping gate requires exactly 2 sites, got $(length(site_indices))"))
        end
        
        site1_idx, site2_idx = site_indices
        _validate_site_index(site1_idx, length(sites))
        _validate_site_index(site2_idx, length(sites))
        
        if site1_idx == site2_idx
            throw(ArgumentError("hopping gate requires different sites, got [$site1_idx, $site2_idx]"))
        end
        
        return _build_hopping_gate(sites[site1_idx], sites[site2_idx], J, dt)
        
    elseif gate_type == "kerr"
        site_idx = params.site
        χ = get(params, :chi, 1.0)
        _validate_site_index(site_idx, length(sites))
        return _build_kerr_gate(sites[site_idx], χ, dt)
        
    else
        throw(ArgumentError("Unknown gate type: '$gate_type'. Supported types: 'number', 'hopping', 'kerr'"))
    end
end

"""
    build_trotter_gates(sites::Vector{<:ITensors.Index}, H_terms::Vector, dt::Real; order::Int=2)

Build Trotter decomposition gates for time evolution.

# Arguments
- `sites::Vector{<:ITensors.Index}`: Vector of site indices
- `H_terms::Vector`: Vector of Hamiltonian terms, each as (gate_type, params)
- `dt::Real`: Time step
- `order::Int`: Trotter order (1, 2, or 4)

# H_terms Format
Each element should be a tuple: `(gate_type::String, params::NamedTuple)`

# Examples
```julia
sites = bosonic_sites(4, 6)
dt = 0.01

# Define Hamiltonian: H = Σᵢ ω*nᵢ + Σᵢ J*(a†ᵢaᵢ₊₁ + aᵢa†ᵢ₊₁)
H_terms = [
    ("number", (site=1, omega=1.0)),
    ("number", (site=2, omega=1.0)),
    ("hopping", (sites=[1,2], J=0.5)),
    ("kerr", (site=1, chi=0.1))
]

gates = build_trotter_gates(sites, H_terms, dt; order=2)
psi_evolved = tebd(psi, gates)
```
"""
function build_trotter_gates(
    sites::Vector{<:ITensors.Index}, 
    H_terms::Vector, 
    dt::Real; 
    order::Int=2
)::Vector{ITensors.ITensor}
    
    # Input validation
    if isempty(sites)
        throw(ArgumentError("sites vector cannot be empty"))
    end
    
    if isempty(H_terms)
        throw(ArgumentError("H_terms vector cannot be empty"))
    end
    
    if !isfinite(dt)
        throw(DomainError(dt, "time step dt must be finite"))
    end
    
    if !(order in [1, 2, 4])
        throw(ArgumentError("Unsupported Trotter order: $order. Supported orders are 1, 2, and 4."))
    end
    
    for (i, term) in enumerate(H_terms)
        if !isa(term, Tuple) || length(term) != 2
            throw(ArgumentError("H_terms[$i]: each term must be a tuple of (gate_type, params)"))
        end
        
        gate_type, params = term
        
        if !isa(gate_type, String)
            throw(ArgumentError("H_terms[$i]: gate_type must be String, got $(typeof(gate_type))"))
        end
        
        if !isa(params, NamedTuple)
            throw(ArgumentError("H_terms[$i]: params must be NamedTuple, got $(typeof(params))"))
        end
    end
    
    gates = ITensors.ITensor[]
    
    if order == 1
        for term in H_terms
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, dt)
            push!(gates, gate)
        end
        
    elseif order == 2
        forward_gates = ITensors.ITensor[]
        for term in H_terms
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, dt/2)
            push!(forward_gates, gate)
        end
        
        append!(gates, forward_gates)
        append!(gates, reverse(forward_gates))
        
    elseif order == 4
        coeff1 = 1/(2 - 2^(1/3))
        coeff2 = 1 - 2*coeff1
        
        for term in H_terms
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, coeff1*dt/2)
            push!(gates, gate)
        end
        
        for term in H_terms
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, coeff2*dt/2)
            push!(gates, gate)
        end
        
        for term in reverse(H_terms)
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, coeff2*dt/2)
            push!(gates, gate)
        end
        
        for term in reverse(H_terms)
            gate_type, params = term
            gate = build_evolution_gate(sites, gate_type, params, coeff1*dt/2)
            push!(gates, gate)
        end
    end
    
    return gates
end

"""
    tdvp!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm (in-place version).
"""
function tdvp!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)
    evolved_mps = ITensorMPS.tdvp(H.mpo, dt, psi.mps; kwargs...)
    psi.mps = evolved_mps
    return psi
end

"""
    tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm (non-mutating version).
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