struct BMPS{M<:ITensorMPS.MPS,A<:MabsAlg}
    mps::M
    alg::A
end
function BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)
    N = log2(length(mps))
    isinteger(N) || throw(PSEUDOSITE_ERROR)
    ps_mps = _create_ps(mps, alg)
    return BMPS(ps_mps, alg)
end

ITensorMPS.siteinds(bmps::BMPS) = ITensorMPS.siteinds(bmps.mps)
ITensorMPS.maxlinkdim(bmps::BMPS) = ITensorMPS.maxlinkdim(bmps.mps)
ITensorMPS.linkind(bmps::BMPS, i::Int) = ITensorMPS.linkind(bmps.mps, i)
ITensorMPS.siteind(bmps::BMPS, i::Int) = ITensorMPS.siteind(bmps.mps, i)

Base.copy(bmps::BMPS) = BMPS(copy(bmps.mps), bmps.alg)
Base.deepcopy(bmps::BMPS) = BMPS(deepcopy(bmps.mps), bmps.alg)
Base.length(bmps::BMPS) = length(bmps.mps)