struct BMPO{M<:ITensorMPS.MPO,A<:MabsAlg}
    mpo::M
    alg::A
end
function BMPO(mpo::ITensorMPS.MPO, alg::PseudoSite)
    N = log2(length(mpo))
    isinteger(N) || throw(PSEUDOSITE_ERROR)
    ps_mpo = _create_ps(mpo, alg)
    return BMPO(ps_mpo, alg)
end
#=
function BMPO(opsum::ITensorMPS.OpSum, alg::PseudoSite)

end
=#

ITensorMPS.siteinds(bmpo::BMO) = ITensorMPS.siteinds(bmpo.mpo)
ITensorMPS.maxlinkdim(bmpo::BMPO) = ITensorMPS.maxlinkdim(bmpo.mpo)
ITensorMPS.linkind(bmpo::BMPO, i::Int) = ITensorMPS.linkind(bmpo.mpo, i)
ITensorMPS.siteind(bmpo::BMPO, i::Int) = ITensorMPS.siteind(bmpo.mpo, i)

Base.copy(bmpo::BMPO) = BMPO(copy(bmpo.mpo), bmpo.alg)
Base.deepcopy(bmpo::BMPO) = BMPO(deepcopy(bmpo.mpo), bmpo.alg)
Base.length(bmpo::BMPO) = length(bmpo.mpo)