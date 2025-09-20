"""
    dmrg(H::BMPO{<:ITensorMPS.MPO,Truncated}, psi0::BMPS{<:ITensorMPS.MPS,Truncated}; nsweeps::Int, kwargs...)

Perform DMRG calculation to find ground state of bosonic Hamiltonian.

Arguments:
- H::BMPO: Bosonic Hamiltonian as matrix product operator
- psi0::BMPS: Initial guess for ground state
- nsweeps::Int: Number of DMRG sweeps to perform

Returns:
- Tuple: (ground_state_energy::Real, ground_state::BMPS) if convergence info returned
- BMPS: Ground state if only state returned
"""
function dmrg(
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    psi0::BMPS{<:ITensorMPS.MPS,Truncated}; 
    nsweeps::Int,
    kwargs...
)
    result = ITensorMPS.dmrg(H.mpo, psi0.mps; nsweeps=nsweeps, kwargs...)
    if result isa Tuple
        energy, converged_mps = result
        return energy, BMPS(converged_mps, psi0.alg)
    else
        return BMPS(result, psi0.alg)
    end
end

"""
    dmrg(H::BMPO{<:ITensorMPS.MPO,Truncated}, psi0::Vector{BMPS{<:ITensorMPS.MPS,Truncated}}; nsweeps::Int, kwargs...)

Perform DMRG calculation for excited states of bosonic Hamiltonian.

Arguments:
- H::BMPO: Bosonic Hamiltonian as matrix product operator
- psi0::Vector{BMPS}: Vector of initial MPS (first is ground state guess, rest are excited state guesses)
- nsweeps::Int: Number of DMRG sweeps to perform

Returns:
- Tuple: (energies::Vector{Real}, states::Vector{BMPS}) if convergence info returned
- Vector{BMPS}: Vector of eigenstates if only states returned
"""
function dmrg(
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    psi0::Vector{BMPS{M,Truncated}}; 
    nsweeps::Int,
    kwargs...
) where {M<:ITensorMPS.MPS}
    mps_vector = [bmps.mps for bmps in psi0]
    result = ITensorMPS.dmrg(H.mpo, mps_vector; nsweeps=nsweeps, kwargs...)
    if result isa Tuple
        energies, converged_states = result
        bmps_states = [BMPS(mps, psi0[1].alg) for mps in converged_states]
        return energies, bmps_states
    else
        bmps_states = [BMPS(mps, psi0[1].alg) for mps in result]
        return bmps_states
    end
end