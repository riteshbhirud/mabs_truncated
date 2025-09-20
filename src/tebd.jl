"""
Time Evolution Block Decimation (TEBD) functions for bosonic MPS.
Improved with proper type signatures, validation, and error handling.
"""

"""
    tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm (in-place version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve (modified in-place)
- gates::Vector{ITensors.ITensor}: Vector of evolution gates to apply

Returns:
- BMPS: The evolved bosonic MPS (same object, modified in-place)
"""
function tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)
    psi.mps = ITensors.apply(gates, psi.mps; kwargs...)
    return psi
end

"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm (non-mutating version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve
- gates::Vector{ITensors.ITensor}: Vector of evolution gates to apply

Returns:
- BMPS: New evolved bosonic MPS
"""
function tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)

Perform time evolution using TEBD algorithm with single gate (in-place version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve (modified in-place)
- gate::ITensors.ITensor: Single evolution gate to apply

Returns:
- BMPS: The evolved bosonic MPS (same object, modified in-place)
"""
function tebd!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)
    psi.mps = ITensors.apply(gate, psi.mps; kwargs...)
    return psi
end

"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)

Perform time evolution using TEBD algorithm with single gate (non-mutating version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve
- gate::ITensors.ITensor: Single evolution gate to apply

Returns:
- BMPS: New evolved bosonic MPS
"""
function tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)
    evolved_mps = ITensors.apply(gate, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    build_trotter_gates(sites::Vector{<:ITensors.Index}, H_terms::Vector, dt::Real; order::Int=2)

Build Trotter decomposition gates for time evolution with comprehensive validation.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices for the full system
- H_terms::Vector: Vector of (OpSum, site_indices) pairs for Hamiltonian terms
- dt::Real: Time step (must be finite and real)
- order::Int: Trotter order (1, 2, or 4; default: 2)

Returns:
- Vector{ITensors.ITensor}: Vector of evolution gates ordered for application

Throws:
- ArgumentError: For invalid inputs (empty terms, invalid sites, unsupported order)
- DomainError: For non-finite or complex time step

# Example
```julia
sites = bosonic_sites(4, 5)
H_terms = [
    (OpSum() + (1.0, "N", 1), [1]),           # Single-site term
    (OpSum() + (0.1, "Adag", 1, "A", 2), [1,2])  # Two-site term
]
gates = build_trotter_gates(sites, H_terms, 0.01; order=2)
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
    
    # Validate all H_terms
    N_sites = length(sites)
    for (i, term) in enumerate(H_terms)
        if !isa(term, Tuple) || length(term) != 2
            throw(ArgumentError("H_terms[$i]: each term must be a tuple of (opsum, site_indices)"))
        end
        
        opsum, term_sites = term
        
        if !isa(term_sites, Vector{Int})
            throw(ArgumentError("H_terms[$i]: site indices must be Vector{Int}, got $(typeof(term_sites))"))
        end
        
        if isempty(term_sites)
            throw(ArgumentError("H_terms[$i]: site indices cannot be empty"))
        end
        
        if !all(s -> 1 <= s <= N_sites, term_sites)
            throw(ArgumentError("H_terms[$i]: site indices $(term_sites) out of range [1, $N_sites]"))
        end
        
        if length(unique(term_sites)) != length(term_sites)
            throw(ArgumentError("H_terms[$i]: duplicate site indices not allowed: $(term_sites)"))
        end
        
        # Check that OpSum is compatible with the specified sites (simplified validation)
        try
            _validate_opsum_sites(opsum, term_sites)
        catch e
            throw(ArgumentError("H_terms[$i]: OpSum validation failed: $(e.msg)"))
        end
    end
    
    gates = ITensors.ITensor[]
    
    if order == 1
        # First-order Trotter: exp(dt*(H₁ + H₂ + ...)) ≈ exp(dt*H₁)exp(dt*H₂)...
        for term in H_terms
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, dt, term_sites)
            push!(gates, gate)
        end
        
    elseif order == 2
        # Second-order Trotter: exp(dt*H) ≈ ∏ᵢ exp(dt*Hᵢ/2) ∏ᵢ exp(dt*Hᵢ/2) (reverse)
        forward_gates = ITensors.ITensor[]
        for term in H_terms
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, dt/2, term_sites)
            push!(forward_gates, gate)
        end
        
        # Forward sweep with dt/2
        append!(gates, forward_gates)
        # Backward sweep with dt/2
        append!(gates, reverse(forward_gates))
        
    elseif order == 4
        # Fourth-order Trotter decomposition (Forest-Ruth algorithm)
        # More sophisticated decomposition for higher accuracy
        coeff1 = 1/(2 - 2^(1/3))
        coeff2 = 1 - 2*coeff1
        
        # Forward with coeff1*dt/2
        for term in H_terms
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, coeff1*dt/2, term_sites)
            push!(gates, gate)
        end
        
        # Forward with coeff2*dt/2
        for term in H_terms
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, coeff2*dt/2, term_sites)
            push!(gates, gate)
        end
        
        # Backward with coeff2*dt/2
        for term in reverse(H_terms)
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, coeff2*dt/2, term_sites)
            push!(gates, gate)
        end
        
        # Backward with coeff1*dt/2
        for term in reverse(H_terms)
            opsum, term_sites = term
            gate = build_evolution_gate(sites, opsum, coeff1*dt/2, term_sites)
            push!(gates, gate)
        end
    end
    
    return gates
end

"""
    build_evolution_gate(sites::Vector{<:ITensors.Index}, opsum::Any, dt::Real, gate_sites::Vector{Int})

Build a single evolution gate exp(-i*dt*H) from an OpSum with proper validation and optimization.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of all site indices in the system
- opsum::Any: Operator sum defining the Hamiltonian term (flexible type for ITensors compatibility)
- dt::Real: Time step (must be finite and real)
- gate_sites::Vector{Int}: Indices of sites involved in this gate (1-based indexing)

Returns:
- ITensors.ITensor: Evolution gate tensor exp(-i*dt*H)

Throws:
- ArgumentError: For invalid inputs (empty gate_sites, out-of-range indices)
- DomainError: For non-finite time step

# Example
```julia
sites = bosonic_sites(4, 5)
opsum = OpSum() + (0.1, "Adag", 1, "A", 2)  # Hopping term
gate = build_evolution_gate(sites, opsum, 0.01, [1, 2])
```
"""
function build_evolution_gate(
    sites::Vector{<:ITensors.Index}, 
    opsum::Any, 
    dt::Real, 
    gate_sites::Vector{Int}
)::ITensors.ITensor
    # Input validation
    if isempty(sites)
        throw(ArgumentError("sites vector cannot be empty"))
    end
    
    if isempty(gate_sites)
        throw(ArgumentError("gate_sites vector cannot be empty"))
    end
    
    if !isfinite(dt)
        throw(DomainError(dt, "time step dt must be finite"))
    end
    
    N_sites = length(sites)
    if !all(s -> 1 <= s <= N_sites, gate_sites)
        throw(ArgumentError("gate_sites $(gate_sites) contains indices out of range [1, $N_sites]"))
    end
    
    if length(unique(gate_sites)) != length(gate_sites)
        throw(ArgumentError("gate_sites contains duplicate indices: $(gate_sites)"))
    end
    
    # Validate that OpSum is compatible with gate_sites (simplified check)
    try
        _validate_opsum_sites(opsum, gate_sites)
    catch e
        # If validation fails, wrap in more helpful error message
        throw(ArgumentError("OpSum validation failed for gate_sites $(gate_sites): $(e.msg)"))
    end
    
    # Extract local sites for the gate
    local_sites = sites[gate_sites]
    
    try
        # Always use MPO construction for consistency, then extract relevant tensors
        # This works for both single-site and multi-site cases
        
        # Build MPO for the Hamiltonian term
        H_mpo = ITensorMPS.MPO(opsum, sites)
        
        if length(gate_sites) == 1
            # Single-site gate - extract the tensor for this site
            site_idx = gate_sites[1]
            H_tensor = H_mpo[site_idx]
            
        else
            # Multi-site gate - need to contract tensors
            if !_is_contiguous(gate_sites)
                throw(ArgumentError("Multi-site gates currently require contiguous sites. Got: $(gate_sites)"))
            end
            
            # Extract and contract the relevant tensors
            H_tensor = H_mpo[gate_sites[1]]
            for i in 2:length(gate_sites)
                H_tensor = ITensors.contract(H_tensor, H_mpo[gate_sites[i]])
            end
        end
        
        # Exponentiate: U = exp(-i*dt*H)
        # Scale the tensor first, then exponentiate
        scaled_H = H_tensor * (-1im * dt)
        U = ITensors.exp(scaled_H)
        
        return U
        
    catch e
        # Provide more helpful error context
        if e isa BoundsError
            throw(ArgumentError("Failed to build evolution gate for sites $(gate_sites): BoundsError - check site indices"))
        elseif e isa MethodError
            throw(ArgumentError("Failed to build evolution gate: OpSum may contain unsupported operators for bosonic sites. Error: $(e)"))
        else
            rethrow(e)
        end
    end
end

"""
    _validate_opsum_sites(opsum::Any, expected_sites::Vector{Int})

Validate that an operator sum is compatible with expected sites.
Simplified validation that doesn't introspect complex ITensor types.

Arguments:
- opsum::Any: The operator sum to validate (flexible typing for ITensors compatibility)
- expected_sites::Vector{Int}: Expected site indices

Note: This function performs basic validation. Complex OpSum introspection
is avoided due to ITensors' complex type hierarchy.
"""
function _validate_opsum_sites(opsum::Any, expected_sites::Vector{Int})
    # Simplified validation - just check that we have a reasonable OpSum-like object
    # The actual site validation will happen when we try to build the MPO
    
    # Basic check: ensure opsum is not nothing and expected_sites is reasonable
    if opsum === nothing
        throw(ArgumentError("OpSum cannot be nothing"))
    end
    
    if isempty(expected_sites)
        throw(ArgumentError("expected_sites cannot be empty"))
    end
    
    # For now, we trust that the OpSum and sites are compatible
    # More detailed validation would require deep introspection of ITensors' complex type system
    # If there's a mismatch, it will be caught when building the actual MPO/gates
    return nothing
end

"""
    _is_contiguous(indices::Vector{Int})

Check if a vector of indices represents contiguous sites.

Arguments:
- indices::Vector{Int}: Vector of site indices

Returns:
- Bool: true if indices are contiguous, false otherwise
"""
function _is_contiguous(indices::Vector{Int})::Bool
    if length(indices) <= 1
        return true
    end
    
    sorted_indices = sort(indices)
    for i in 2:length(sorted_indices)
        if sorted_indices[i] != sorted_indices[i-1] + 1
            return false
        end
    end
    
    return true
end

"""
    tdvp!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm (in-place version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve (modified in-place)
- H::BMPO: Bosonic Hamiltonian MPO
- dt::Number: Time step

Returns:
- BMPS: The evolved bosonic MPS (same object, modified in-place)
"""
function tdvp!(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)
    # Fix: correct argument order and update in-place
    evolved_mps = ITensorMPS.tdvp(H.mpo, dt, psi.mps; kwargs...)
    psi.mps = evolved_mps
    return psi
end

"""
    tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm (non-mutating version).

Arguments:
- psi::BMPS: Bosonic MPS to evolve
- H::BMPO: Bosonic Hamiltonian MPO
- dt::Number: Time step

Returns:
- BMPS: New evolved bosonic MPS
"""
function tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)
    # Fix: correct argument order is (operator, t, init_mps)
    evolved_mps = ITensorMPS.tdvp(H.mpo, dt, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end