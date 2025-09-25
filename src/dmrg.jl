"""
    dmrg(H::BMPO{<:ITensorMPS.MPO,Truncated}, psi0::BMPS{<:ITensorMPS.MPS,Truncated}; nsweeps::Int, kwargs...)

Perform DMRG calculation to find ground state of bosonic Hamiltonian.

Arguments:
- H::BMPO: Bosonic Hamiltonian as matrix product operator
- psi0::BMPS: Initial guess for ground state
- nsweeps::Int: Number of DMRG sweeps to perform

Keyword Arguments:
- kwargs...: Additional parameters passed to ITensorMPS.dmrg

Returns:
- Tuple: (energy::Real, psi_gs::BMPS) - ground state energy and state
"""
function dmrg(
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    psi0::BMPS{<:ITensorMPS.MPS,Truncated}; 
    nsweeps::Int,
    kwargs...
)
    energy, converged_mps = ITensorMPS.dmrg(H.mpo, psi0.mps; nsweeps=nsweeps, kwargs...)
    return energy, BMPS(converged_mps, psi0.alg)
end

"""
    dmrg(H::BMPO{<:ITensorMPS.MPO,Truncated}, Ms::Vector{BMPS{<:ITensorMPS.MPS,Truncated}}, psi0::BMPS{<:ITensorMPS.MPS,Truncated}; weight=1.0, nsweeps::Int, kwargs...)

Perform DMRG calculation for excited states of bosonic Hamiltonian.

Arguments:
- H::BMPO: Bosonic Hamiltonian as matrix product operator
- Ms::Vector{BMPS}: Vector of MPS to orthogonalize against (e.g., ground state and lower excited states)
- psi0::BMPS: Initial guess for the excited state
- nsweeps::Int: Number of DMRG sweeps to perform

Keyword Arguments:
- weight::Real: Penalty weight for orthogonality constraint (default: 1.0)
- kwargs...: Additional parameters passed to ITensorMPS.dmrg

Returns:
- Tuple: (energy::Real, psi_excited::BMPS) - excited state energy and state

Note: The returned energy includes the penalty term. For the true H expectation value,
compute it separately with inner(psi', H, psi).
"""
function dmrg(
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    Ms::Vector{BMPS{M,Truncated}},
    psi0::BMPS{<:ITensorMPS.MPS,Truncated}; 
    weight::Real=1.0,
    nsweeps::Int,
    kwargs...
) where {M<:ITensorMPS.MPS}
    mps_vector = [bmps.mps for bmps in Ms]
    energy, converged_mps = ITensorMPS.dmrg(H.mpo, mps_vector, psi0.mps; weight=weight, nsweeps=nsweeps, kwargs...)
    return energy, BMPS(converged_mps, psi0.alg)
end