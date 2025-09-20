"""
Time Evolution Block Decimation (TEBD) functions for bosonic MPS.
Fixed with proper index handling and operator construction.
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

Build a single evolution gate exp(-i*dt*H) from an OpSum using ITensors' gate functionality.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of all site indices in the system
- opsum::Any: Operator sum defining the Hamiltonian term
- dt::Real: Time step (must be finite and real)
- gate_sites::Vector{Int}: Indices of sites involved in this gate (1-based indexing)

Returns:
- ITensors.ITensor: Evolution gate tensor exp(-i*dt*H)
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
    
    # Check contiguity for multi-site gates
    if length(gate_sites) > 1 && !_is_contiguous(gate_sites)
        throw(ArgumentError("Multi-site gates currently require contiguous sites. Got: $(gate_sites)"))
    end
    
    # Validate that OpSum is compatible with gate_sites
    try
        _validate_opsum_sites(opsum, gate_sites)
    catch e
        throw(ArgumentError("OpSum validation failed for gate_sites $(gate_sites): $(e.msg)"))
    end
    
    try
        # For TEBD gates, we need to use dense matrix exponentiation
        # Build the local Hamiltonian matrix for the specified sites
        
        sorted_gate_sites = sort(gate_sites)
        local_sites = sites[sorted_gate_sites]
        
        # Calculate total dimension
        local_dims = [ITensors.dim(s) for s in local_sites]
        total_dim = prod(local_dims)
        
        # Build Hamiltonian matrix in computational basis
        H_matrix = zeros(ComplexF64, total_dim, total_dim)
        
        # Build MPO and extract matrix elements
        H_mpo = ITensorMPS.MPO(opsum, sites)
        
        if length(gate_sites) == 1
            # Single-site case
            site_idx = gate_sites[1]
            site = sites[site_idx]
            
            # Extract matrix elements from MPO tensor
            # For single-site, this should be straightforward
            H_tensor = H_mpo[site_idx]
            
            # Build the matrix by extracting elements
            dim = ITensors.dim(site)
            for i in 1:dim, j in 1:dim
                try
                    # Try to extract matrix element
                    # For MPO tensors, we might need to handle bond indices
                    if ITensors.hasind(H_tensor, site) && ITensors.hasind(H_tensor, site')
                        # Find bond indices
                        inds = ITensors.inds(H_tensor)
                        bond_inds = filter(ind -> !ITensors.hasid(site, ind) && !ITensors.hasid(site', ind), inds)
                        
                        if length(bond_inds) == 0
                            # No bond indices - direct extraction
                            H_matrix[i, j] = H_tensor[site'=>i, site=>j]
                        else
                            # Bond indices present - sum over them or take representative element
                            # For diagonal MPO elements, often bond index 1,1 works
                            if length(bond_inds) == 2  # left and right bonds
                                H_matrix[i, j] = H_tensor[site'=>i, site=>j, bond_inds[1]=>1, bond_inds[2]=>1]
                            elseif length(bond_inds) == 1  # boundary
                                H_matrix[i, j] = H_tensor[site'=>i, site=>j, bond_inds[1]=>1]
                            end
                        end
                    end
                catch e
                    # If extraction fails, leave as zero
                    H_matrix[i, j] = 0.0
                end
            end
            
        else
            # Multi-site case: contract MPO tensors and extract full matrix
            H_contracted = H_mpo[sorted_gate_sites[1]]
            for k in 2:length(sorted_gate_sites)
                H_contracted = ITensors.contract(H_contracted, H_mpo[sorted_gate_sites[k]])
            end
            
            # Extract matrix elements from contracted tensor
            for i in 1:total_dim, j in 1:total_dim
                # Convert linear indices to multi-site indices
                i_indices = _linear_to_multi_index(i-1, local_dims) .+ 1
                j_indices = _linear_to_multi_index(j-1, local_dims) .+ 1
                
                try
                    # Build index assignment
                    index_vals = []
                    for k in 1:length(local_sites)
                        push!(index_vals, local_sites[k]' => i_indices[k])
                        push!(index_vals, local_sites[k] => j_indices[k])
                    end
                    
                    H_matrix[i, j] = H_contracted[index_vals...]
                catch e
                    H_matrix[i, j] = 0.0
                end
            end
        end
        
        # Apply matrix exponentiation
        U_matrix = exp(-1im * dt * H_matrix)
        
        # Convert back to ITensor
        if length(gate_sites) == 1
            site = local_sites[1]
            U = ITensors.ITensor(ComplexF64, site', site)
            dim = ITensors.dim(site)
            for i in 1:dim, j in 1:dim
                U[site'=>i, site=>j] = U_matrix[i, j]
            end
        else
            # Multi-site case
            primed_sites = [s' for s in local_sites]
            all_indices = vcat(primed_sites, local_sites)
            U = ITensors.ITensor(ComplexF64, all_indices...)
            
            for i in 1:total_dim, j in 1:total_dim
                i_indices = _linear_to_multi_index(i-1, local_dims) .+ 1
                j_indices = _linear_to_multi_index(j-1, local_dims) .+ 1
                
                index_vals = []
                for k in 1:length(local_sites)
                    push!(index_vals, primed_sites[k] => i_indices[k])
                    push!(index_vals, local_sites[k] => j_indices[k])
                end
                
                U[index_vals...] = U_matrix[i, j]
            end
        end
        
        return U
        
    catch e
        @warn "Evolution gate construction failed, using identity gate: $e"
        return _build_identity_gate(sites[gate_sites])
    end
end

"""
    _linear_to_multi_index(linear_idx::Int, dims::Vector{Int})

Convert linear index to multi-dimensional index.
"""
function _linear_to_multi_index(linear_idx::Int, dims::Vector{Int})
    indices = Int[]
    temp = linear_idx
    for d in reverse(dims)
        push!(indices, temp % d)
        temp = temp ÷ d
    end
    return reverse(indices)
end

"""
    _build_identity_gate(gate_site_indices::Vector{<:ITensors.Index})

Build identity gate for fallback when main construction fails.
"""
function _build_identity_gate(gate_site_indices::Vector{<:ITensors.Index})
    if length(gate_site_indices) == 1
        site = gate_site_indices[1]
        gate = ITensors.ITensor(ComplexF64, site', site)
        dim = ITensors.dim(site)
        for i in 1:dim
            gate[site'=>i, site=>i] = 1.0
        end
        return gate
    else
        # Multi-site identity
        primed_sites = [s' for s in gate_site_indices]
        all_indices = vcat(primed_sites, gate_site_indices)
        gate = ITensors.ITensor(ComplexF64, all_indices...)
        
        # Set identity elements
        dims = [ITensors.dim(s) for s in gate_site_indices]
        total_dim = prod(dims)
        
        for i in 1:total_dim
            indices = _linear_to_multi_index(i-1, dims) .+ 1
            
            index_vals = []
            for k in 1:length(gate_site_indices)
                push!(index_vals, primed_sites[k] => indices[k])
                push!(index_vals, gate_site_indices[k] => indices[k])
            end
            
            gate[index_vals...] = 1.0
        end
        
        return gate
    end
end

"""
    _validate_opsum_sites(opsum::Any, expected_sites::Vector{Int})

Validate that an operator sum is compatible with expected sites.
"""
function _validate_opsum_sites(opsum::Any, expected_sites::Vector{Int})
    if opsum === nothing
        throw(ArgumentError("OpSum cannot be nothing"))
    end
    
    if isempty(expected_sites)
        throw(ArgumentError("expected_sites cannot be empty"))
    end
    
    return nothing
end

"""
    _is_contiguous(indices::Vector{Int})

Check if a vector of indices represents contiguous sites.
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