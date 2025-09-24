
#Helper function for safe factorial calculation to avoid performance issues.
function safe_factorial(n::Int)
    if n <= 20
        return factorial(n)
    else
        return factorial(big(n))
    end
end

ITensors.op(::ITensors.OpName"N", ::ITensors.SiteType"Boson", s::ITensors.Index) = number_op(s)
ITensors.op(::ITensors.OpName"Adag", ::ITensors.SiteType"Boson", s::ITensors.Index) = creation_op(s)
ITensors.op(::ITensors.OpName"A", ::ITensors.SiteType"Boson", s::ITensors.Index) = annihilation_op(s)

"""
    creation_op(site_ind::ITensors.Index)

Create the bosonic creation operator (raising operator) for a given site.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Creation operator tensor
"""
function creation_op(site_ind::ITensors.Index)
    max_occ = ITensors.dim(site_ind) - 1
    op = ITensors.ITensor(ComplexF64, site_ind', site_ind)
    
    for n in 0:(max_occ-1)
        op[n+2, n+1] = sqrt(n+1)
    end
    return op
end

"""
    annihilation_op(site_ind::ITensors.Index)

Create the bosonic annihilation operator (lowering operator) for a given site.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Annihilation operator tensor
"""
function annihilation_op(site_ind::ITensors.Index)
    max_occ = ITensors.dim(site_ind) - 1
    op = ITensors.ITensor(ComplexF64, site_ind', site_ind)
    
    for n in 1:max_occ
        op[n, n+1] = sqrt(n)
    end
    return op
end

"""
    number_op(site_ind::ITensors.Index)

Create the bosonic number operator for a given site.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag

Returns:
- ITensors.ITensor: Number operator tensor
"""
function number_op(site_ind::ITensors.Index)
    max_occ = ITensors.dim(site_ind) - 1
    op = ITensors.ITensor(ComplexF64, site_ind', site_ind)
    
    for n in 0:max_occ
        op[n+1, n+1] = n
    end
    return op
end

"""
    displacement_op(site_ind::ITensors.Index, α::Number)

Create the displacement operator D(α) = exp(α*a† - α*a) for a given site.
Uses matrix exponentiation for numerical stability.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag
- α::Number: Displacement amplitude (can be complex)

Returns:
- ITensors.ITensor: Displacement operator tensor
"""
function displacement_op(site_ind::ITensors.Index, α::Number)
    #  G = α*a† - α*a
    a_dag = creation_op(site_ind)
    a = annihilation_op(site_ind)
    
    generator = α * a_dag - conj(α) * a
    
    op = ITensors.exp(generator)
    return op
end

"""
    squeezing_op(site_ind::ITensors.Index, ξ::Number)

Create the squeezing operator S(ξ) = exp(0.5*(ξ*a†² - ξ*a²)) for a given site.
Uses direct matrix element construction for numerical stability.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag  
- ξ::Number: Squeezing parameter (can be complex)

Returns:
- ITensors.ITensor: Squeezing operator tensor
"""
function squeezing_op(site_ind::ITensors.Index, ξ::Number)
    max_occ = ITensors.dim(site_ind) - 1
    op = ITensors.ITensor(ComplexF64, site_ind', site_ind)
    
    r = abs(ξ)
    φ = angle(ξ)
    
    for n in 0:max_occ
        for m in 0:max_occ
            if (n + m) % 2 == 0 
                k_max = min(n, m) ÷ 2
                element = 0.0 + 0.0im
                
                for k in 0:k_max
                    coeff = sqrt(safe_factorial(n) * safe_factorial(m)) / 
                           (safe_factorial(k) * safe_factorial((n-2k)) * safe_factorial((m-2k)))
                    coeff *= (-0.5 * tanh(r) * exp(2im*φ))^k / sqrt(cosh(r))
                    
                    if n == m && k == 0
                        coeff /= sqrt(cosh(r))  #.. additional factor for diagonal terms
                    end
                    
                    element += coeff
                end
                
                op[m+1, n+1] = convert(ComplexF64, element)
            end
        end
    end
    
    return op
end

"""
    kerr_evolution_op(site_ind::ITensors.Index, χ::Real, t::Real)

Create the Kerr evolution operator exp(-i*χ*t*n²) for a given site.

Arguments:
- site_ind::ITensors.Index: Site index with bosonic tag
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

Returns:
- ITensors.ITensor: Kerr evolution operator tensor
"""
function kerr_evolution_op(site_ind::ITensors.Index, χ::Real, t::Real)
    max_occ = ITensors.dim(site_ind) - 1
    op = ITensors.ITensor(ComplexF64, site_ind', site_ind)
    
    for n in 0:max_occ
        phase = exp(-1im * χ * t * n^2)
        op[n+1, n+1] = phase
    end
    return op
end

"""
    build_harmonic_chain_mpo(sites::Vector{<:ITensors.Index}; ω::Real=1.0, J::Real=0.0)

Build MPO for a chain of harmonic oscillators with optional nearest-neighbor coupling.
H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + aᵢaᵢ₊₁†)

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- ω::Real: Harmonic oscillator frequency (default: 1.0)
- J::Real: Nearest-neighbor hopping strength (default: 0.0)

Returns:
- BMPO: Matrix product operator for harmonic chain
"""
function build_harmonic_chain_mpo(sites::Vector{<:ITensors.Index}; ω::Real=1.0, J::Real=0.0)
    opsum = ITensors.OpSum()
    
    for i in 1:length(sites)
        opsum += ω, "N", i
    end
    
    if J != 0.0
        for i in 1:(length(sites)-1)
            opsum += J, "Adag", i, "A", i+1
            opsum += J, "A", i, "Adag", i+1
        end
    end
    
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, Truncated())
end

"""
    build_kerr_chain_mpo(sites::Vector{<:ITensors.Index}; ω::Real=1.0, χ::Real=0.1)

Build MPO for a chain of Kerr oscillators.
H = Σᵢ (ω*nᵢ + χ*nᵢ²)

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- ω::Real: Linear frequency (default: 1.0)
- χ::Real: Kerr nonlinearity strength (default: 0.1)

Returns:
- BMPO: Matrix product operator for Kerr chain
"""
function build_kerr_chain_mpo(sites::Vector{<:ITensors.Index}; ω::Real=1.0, χ::Real=0.1)
    opsum = ITensors.OpSum()
    
    for i in 1:length(sites)
        opsum += ω, "N", i
    end
    
    for i in 1:length(sites)
        opsum += χ, "N", i, "N", i
    end
    
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, Truncated())
end