struct PseudoSite{S<:Vector{ITensors.Index}} <: MabsAlg 
    sites::S
end

function _create_ps(bmps::BMPS)
    throw(ArgumentError("unsupported!"))
end
function _create_ps(bmpo::BMPO)
    throw(ArgumentError("unsupported!"))
end