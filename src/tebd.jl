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
    build_trotter_gates(sites::Vector{ITensors.Index}, H_terms::Vector{Tuple{ITensors.OpSum,Vector{Int}}}, dt::Number; order::Int=2)

Build Trotter decomposition gates for time evolution.

Arguments:
- sites::Vector{ITensors.Index}: Vector of site indices
- H_terms::Vector{Tuple{ITensors.OpSum,Vector{Int}}}: Vector of (OpSum, site_indices) pairs for Hamiltonian terms
- dt::Number: Time step
- order::Int: Trotter order (1 or 2, default: 2)

Returns:
- Vector{ITensors.ITensor}: Vector of evolution gates
"""
function build_trotter_gates(sites::Vector{ITensors.Index}, H_terms::Vector{Tuple{ITensors.OpSum,Vector{Int}}}, dt::Number; order::Int=2)
    gates = ITensors.ITensor[]
    
    if order == 1
        for (opsum, term_sites) in H_terms
            gate = build_evolution_gate(sites, opsum, dt, term_sites)
            push!(gates, gate)
        end
    elseif order == 2
        forward_gates = ITensors.ITensor[]
        for (opsum, term_sites) in H_terms
            gate = build_evolution_gate(sites, opsum, dt/2, term_sites)
            push!(forward_gates, gate)
        end
        append!(gates, forward_gates)
        append!(gates, reverse(forward_gates))
    else
        error("Unsupported Trotter order: $order. Supported orders are 1 and 2.")
    end
    return gates
end

"""
    build_evolution_gate(sites::Vector{ITensors.Index}, opsum::ITensors.OpSum, dt::Number, gate_sites::Vector{Int})

Build a single evolution gate from an OpSum.

Arguments:
- sites::Vector{ITensors.Index}: Vector of all site indices
- opsum::ITensors.OpSum: Operator sum for the gate
- dt::Number: Time step
- gate_sites::Vector{Int}: Sites involved in this gate

Returns:
- ITensors.ITensor: Evolution gate tensor
"""
function build_evolution_gate(sites::Vector{ITensors.Index}, opsum::ITensors.OpSum, dt::Number, gate_sites::Vector{Int})
    local_sites = sites[gate_sites]
    H_mpo = ITensorMPS.MPO(opsum, sites)
    H_tensor = H_mpo[gate_sites[1]]
    for i in 2:length(gate_sites)
        H_tensor = ITensors.contract(H_tensor, H_mpo[gate_sites[i]])
    end
    U = ITensors.exp(H_tensor, -1im * dt)
    
    return U
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