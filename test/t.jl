using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Truncated Algorithm Tests" begin
    
    @testset "Basic BMPS Operations" begin
        N = 4
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi = random_bmps(sites, Truncated())
        @test length(psi) == N
        @test maxlinkdim(psi) > 0
        
        @test siteinds(psi) == sites
        @test eltype(psi) <: Number  
        
        psi_copy = copy(psi)
        @test psi_copy !== psi
        @test psi_copy.mps !== psi.mps
        
        psi_deepcopy = deepcopy(psi)
        @test psi_deepcopy !== psi
        @test psi_deepcopy.mps !== psi.mps
    end
    
    @testset "Product State Construction" begin
        N = 3
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        vac = vacuumstate(sites, Truncated())
        @test length(vac) == N
        @test abs(norm(vac) - 1.0) < 1e-10  
        
        states = [1, 2, 3] 
        psi = BMPS(sites, states, Truncated())
        @test length(psi) == N
        @test abs(norm(psi) - 1.0) < 1e-10
    end
    
    @testset "Coherent State Construction" begin
        max_occ = 10
        site = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        α = 0.5 + 0.3im
        
        coherent_mps = coherentstate(site, α, Truncated())
        @test length(coherent_mps) == 1
        @test abs(norm(coherent_mps) - 1.0) < 1e-8  
    end
    
    @testset "Operator Construction" begin
        max_occ = 5
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
        
        a_dag = create(site)
        @test a_dag isa ITensors.ITensor
        
        a = destroy(site)
        @test a isa ITensors.ITensor
        
        n = number(site)
        @test n isa ITensors.ITensor
        
        vac = vacuumstate(sites, Truncated())
        one_photon = BMPS(sites, [2], Truncated()) 
        two_photon = BMPS(sites, [3], Truncated())  
        
        a_dag_vac = ITensors.apply(a_dag, vac.mps)
        normalize!(a_dag_vac)
        overlap_1 = abs(ITensors.inner(a_dag_vac, one_photon.mps))
        @test overlap_1 > 0.99  
        
        a_one = ITensors.apply(a, one_photon.mps)
        normalize!(a_one)
        overlap_0 = abs(ITensors.inner(a_one, vac.mps))
        @test overlap_0 > 0.99
        
        n_one = ITensors.apply(n, one_photon.mps)
        expectation_n = real(ITensors.inner(one_photon.mps, n_one))
        @test abs(expectation_n - 1.0) < 1e-8 
    end
    
    @testset "Hamiltonian Construction" begin
        N = 4
        max_occ = 6
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H_harmonic = harmonic_chain(sites; ω=1.0, J=0.1)
        @test H_harmonic isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_harmonic) == N
        
        H_kerr = kerr(sites; ω=1.0, χ=0.05)
        @test H_kerr isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_kerr) == N
    end

    @testset "BMPO OpSum Constructor" begin
    N = 3
    max_occ = 5
    sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
    
    @testset "Direct OpSum to BMPO construction" begin
        opsum = ITensors.OpSum()
        for i in 1:N
            opsum += 1.0, "N", i  
        end
        for i in 1:(N-1)
            opsum += 0.5, "Adag", i, "A", i+1  
            opsum += 0.5, "A", i, "Adag", i+1
        end
        
        H_direct = BMPO(opsum, sites, Truncated())
        
        @test H_direct isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_direct) == N
        mpo_unprimed_sites = [s[end] for s in siteinds(H_direct)] 
        @test mpo_unprimed_sites == sites
        @test H_direct.alg isa Truncated
        
        mpo_manual = ITensorMPS.MPO(opsum, sites)
        H_manual = BMPO(mpo_manual, Truncated())
        
        @test length(H_direct) == length(H_manual)
        @test siteinds(H_direct) == siteinds(H_manual)
        @test maxlinkdim(H_direct) == maxlinkdim(H_manual)
    end
    
    @testset "OpSum constructor with different Hamiltonians" begin
        opsum_harmonic = ITensors.OpSum()
        for i in 1:N
            opsum_harmonic += 2.0, "N", i
        end
        
        H_harmonic = BMPO(opsum_harmonic, sites, Truncated())
        @test H_harmonic isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_harmonic) == N
        
        opsum_hopping = ITensors.OpSum()
        for i in 1:(N-1)
            opsum_hopping += 1.0, "Adag", i, "A", i+1
            opsum_hopping += 1.0, "A", i, "Adag", i+1
        end
        
        H_hopping = BMPO(opsum_hopping, sites, Truncated())
        @test H_hopping isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_hopping) == N
        
        opsum_single = ITensors.OpSum()
        opsum_single += 3.0, "N", 1
        opsum_single += 1.5, "N", 2
        
        H_single = BMPO(opsum_single, sites, Truncated())
        @test H_single isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_single) == N
    end
    
    @testset "OpSum BMPO with DMRG" begin
        opsum = ITensors.OpSum()
        for i in 1:N
            opsum += 1.0, "N", i
        end
        
        H = BMPO(opsum, sites, Truncated())
        psi0 = random_bmps(sites, Truncated(); linkdims=4)
        
        result = Mabs.dmrg(H, psi0; nsweeps=2, maxdim=20, cutoff=1e-10)
        
        if result isa Tuple
            energy, psi_gs = result
            @test energy isa Real
            @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
        else
            psi_gs = result
            @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
    end
    
    @testset "OpSum BMPO expectation values" begin
        opsum = ITensors.OpSum()
        opsum += 1.0, "N", 1  
        
        H = BMPO(opsum, sites, Truncated())
        
        psi = BMPS(sites, [2, 1, 1], Truncated())  
        normalize!(psi)
        
        expectation_val = real(ITensors.inner(psi.mps, ITensors.Apply(H.mpo, psi.mps)))
        @test abs(expectation_val - 1.0) < 1e-10
    end
end
    
    @testset "DMRG Ground State Calculation" begin
        N = 2  
        max_occ = 6
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H = harmonic_chain(sites; ω=1.0, J=0.0)
        psi0 = random_bmps(sites, Truncated())
        
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=8, maxdim=100, cutoff=1e-10, noise=1e-11)
        
        @test energy isa Real
        @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test abs(energy - N/2) < 1.0
    end

    @testset "DMRG Excited State Calculation" begin
        N = 2
        max_occ = 6
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H = harmonic_chain(sites; ω=1.0, J=0.0)
        
        psi0 = random_bmps(sites, Truncated())
        E0, psi_gs = Mabs.dmrg(H, psi0; nsweeps=8, maxdim=100, cutoff=1e-10)
        
        psi1_init = random_bmps(sites, Truncated())
        E1, psi_excited = Mabs.dmrg(H, [psi_gs], psi1_init; weight=10.0, nsweeps=8, maxdim=100, cutoff=1e-10)
        
        @test E1 > E0  
        @test psi_excited isa BMPS{<:ITensorMPS.MPS,Truncated}
        
        overlap = abs(dot(psi_gs, psi_excited))
        @test overlap < 0.1  
    end
    
    @testset "Time Evolution" begin
        N = 2
        max_occ = 4 
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi0 = vacuumstate(sites, Truncated())
        
        dt = 0.001  
        id_gate = ITensors.op("Id", sites[1])
        
        psi_evolved = tebd(psi0, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
        
        psi_evolved = copy(psi0)
        psi_evolved = tebd(psi_evolved, id_gate)
        @test abs(norm(psi_evolved) - 1.0) < 1e-10
    end
    
    @testset "TDVP Evolution" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H = harmonic_chain(sites; ω=1.0)
        psi0 = random_bmps(sites, Truncated(); linkdims=4)
        normalize!(psi0)
        
        dt = 0.01
        
        psi_evolved = Mabs.tdvp(psi0, H, dt; cutoff=1e-8)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
    end
    
    @testset "Arithmetic Operations" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi1 = random_bmps(sites, Truncated(); linkdims=2)
        psi2 = random_bmps(sites, Truncated(); linkdims=2)
        
        psi_sum = psi1 + psi2
        @test psi_sum isa BMPS{<:ITensorMPS.MPS,Truncated}
        
        overlap = dot(psi1, psi2)
        @test overlap isa Number
        @test isfinite(overlap)
        
        H = harmonic_chain(sites; ω=1.0)
        normalize!(psi1)
        expectation_val = real(ITensors.inner(psi1.mps, ITensors.Apply(H.mpo, psi1.mps)))
        @test expectation_val isa Real
        @test isfinite(expectation_val)
    end
    
    @testset "Multi-mode Coherent States" begin
    single_sites = ITensors.siteinds("Boson", 1; dim=7)
    psi_single = coherentstate(single_sites, 0.5, Truncated())
    @test length(psi_single) == 1
    @test abs(norm(psi_single) - 1.0) < 1e-8
    
    multi_sites = ITensors.siteinds("Boson", 3; dim=7)
    psi_uniform = coherentstate(multi_sites, 0.3, Truncated())
    @test length(psi_uniform) == 3
    @test abs(norm(psi_uniform) - 1.0) < 1e-8
    
    αs = [0.2, 0.4 + 0.1im, 0.1]
    psi_mixed = coherentstate(multi_sites, αs, Truncated())
    @test length(psi_mixed) == 3
    @test abs(norm(psi_mixed) - 1.0) < 1e-8
    
    @test_throws ErrorException coherentstate(multi_sites, [0.1, 0.2], Truncated())  
end
    @testset "Operator Matrix Elements" begin
        max_occ = 3
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
        
        a_dag = create(site)
        a = destroy(site)
        
        state_0 = BMPS(sites, [1], Truncated()) 
        state_1 = BMPS(sites, [2], Truncated())  
        state_2 = BMPS(sites, [3], Truncated())  
        
        result_01 = ITensors.apply(a_dag, state_0.mps)
        overlap_01 = abs(ITensors.inner(state_1.mps, result_01))
        @test abs(overlap_01 - 1.0) < 1e-10  
        
        result_12 = ITensors.apply(a_dag, state_1.mps)
        overlap_12 = abs(ITensors.inner(state_2.mps, result_12))
        @test abs(overlap_12 - sqrt(2)) < 1e-10
        
        result_10 = ITensors.apply(a, state_1.mps)
        overlap_10 = abs(ITensors.inner(state_0.mps, result_10))
        @test abs(overlap_10 - 1.0) < 1e-10
    end
    
    @testset "Displacement Operator" begin
        max_occ = 8
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        α = 0.5  
        
        D = displace(sites[1], α)
        @test D isa ITensors.ITensor
        
        @test ITensors.hasinds(D, sites[1]', sites[1])
        
        vac = vacuumstate(sites, Truncated())
        displaced_vac = ITensors.apply(D, vac.mps)
        
        overlap = abs(ITensors.inner(vac.mps, displaced_vac))
        @test overlap < 1.0  
        @test overlap > 0.0  
    end

    @testset "TEBD Tests" begin
    
    @testset "TEBD Evolution" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        @testset "Basic evolution with identity gate" begin
            psi = vacuumstate(sites, Truncated())
            id_gate = ITensors.op("Id", sites[1])
            
            psi_evolved = tebd(psi, id_gate)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
            @test abs(norm(psi_evolved) - 1.0) < 1e-10
        end
        
        @testset "Evolution preserves normalization" begin
            psi = random_bmps(sites, Truncated(); linkdims=2)
            normalize!(psi)
            
            # Use simple number operator evolution
            n_op = number(sites[1])
            dt = 0.01
            gate = exp(-1im * dt * n_op)
            
            psi_evolved = tebd(psi, gate)
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
        end
    end
    
    @testset "Backwards Compatibility" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi = random_bmps(sites, Truncated(); linkdims=4)
        normalize!(psi)
        
        id_gate = ITensors.op("Id", sites[1])
        
        psi_evolved = tebd(psi, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi
        
        psi_evolved = copy(psi)
        psi_evolved = tebd(psi_evolved, id_gate)
        @test abs(norm(psi_evolved) - 1.0) < 1e-10
    end
end
end