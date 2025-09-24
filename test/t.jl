using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Truncated Algorithm Tests" begin
    
    @testset "Basic BMPS Operations" begin
        N = 4
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        
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
        sites = bosonic_sites(N, max_occ)
        
        vac = vacuum_mps(sites, Truncated())
        @test length(vac) == N
        @test abs(norm(vac) - 1.0) < 1e-10  
        
        states = [1, 2, 3] 
        psi = bosonic_product_mps(sites, states, Truncated())
        @test length(psi) == N
        @test abs(norm(psi) - 1.0) < 1e-10
    end
    
    @testset "Coherent State Construction" begin
        max_occ = 10
        site = bosonic_sites(1, max_occ)
        α = 0.5 + 0.3im
        
        coherent_mps = coherent_state_mps(site, α, Truncated())
        @test length(coherent_mps) == 1
        @test abs(norm(coherent_mps) - 1.0) < 1e-8  
    end
    
    @testset "Operator Construction" begin
        max_occ = 5
        sites = bosonic_sites(1, max_occ)
        site = sites[1]
        
        a_dag = creation_op(site)
        @test a_dag isa ITensors.ITensor
        
        a = annihilation_op(site)
        @test a isa ITensors.ITensor
        
        n = number_op(site)
        @test n isa ITensors.ITensor
        
        vac = vacuum_mps(sites, Truncated())
        one_photon = bosonic_product_mps(sites, [2], Truncated())  
        two_photon = bosonic_product_mps(sites, [3], Truncated())  
        
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
        sites = bosonic_sites(N, max_occ)
        
        H_harmonic = build_harmonic_chain_mpo(sites; ω=1.0, J=0.1)
        @test H_harmonic isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_harmonic) == N
        
        H_kerr = build_kerr_chain_mpo(sites; ω=1.0, χ=0.05)
        @test H_kerr isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_kerr) == N
    end
    
    @testset "DMRG Ground State Calculation" begin
        N = 2  
        max_occ = 6
        sites = bosonic_sites(N, max_occ)
        
        H = build_harmonic_chain_mpo(sites; ω=1.0, J=0.0)
        psi0 = random_bmps(sites, Truncated())
        
        result = Mabs.dmrg(H, psi0; nsweeps=8, maxdim=100, cutoff=1e-10, noise=1e-11)
        
        if result isa Tuple
            energy, psi_gs = result
            @test energy isa Real
            @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}

            @test abs(energy - N/2) < 1.0  
        else
            psi_gs = result
            @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
    end
    
    @testset "Time Evolution" begin
        N = 2
        max_occ = 4 
        sites = bosonic_sites(N, max_occ)
        
        psi0 = vacuum_mps(sites, Truncated())
        
        dt = 0.001  
        id_gate = ITensors.op("Id", sites[1])
        
        psi_evolved = tebd(psi0, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
        
        psi_inplace = copy(psi0)
        tebd!(psi_inplace, id_gate)
        @test abs(norm(psi_inplace) - 1.0) < 1e-10  
    end
    
    @testset "TDVP Evolution" begin
        N = 2
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        
        H = build_harmonic_chain_mpo(sites; ω=1.0)
        psi0 = random_bmps(sites, Truncated(); linkdim=4)
        normalize!(psi0)
        
        dt = 0.01
        
        psi_evolved = Mabs.tdvp(psi0, H, dt; cutoff=1e-8)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
        
        psi_inplace = copy(psi0)
        Mabs.tdvp!(psi_inplace, H, dt; cutoff=1e-8)
        @test abs(norm(psi_inplace) - 1.0) < 0.05  
    end
    
    @testset "Arithmetic Operations" begin
        N = 2
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        
        psi1 = random_bmps(sites, Truncated(); linkdim=2)
        psi2 = random_bmps(sites, Truncated(); linkdim=2)
        
        psi_sum = psi1 + psi2
        @test psi_sum isa BMPS{<:ITensorMPS.MPS,Truncated}
        
        overlap = dot(psi1, psi2)
        @test overlap isa Number
        @test isfinite(overlap)
        
        H = build_harmonic_chain_mpo(sites; ω=1.0)
        normalize!(psi1)
        expectation_val = real(ITensors.inner(psi1.mps, ITensors.Apply(H.mpo, psi1.mps)))
        @test expectation_val isa Real
        @test isfinite(expectation_val)
    end
    
    @testset "Multi-mode Coherent States" begin
    single_sites = bosonic_sites(1, 6)
    psi_single = coherent_state_mps(single_sites, 0.5, Truncated())
    @test length(psi_single) == 1
    @test abs(norm(psi_single) - 1.0) < 1e-8
    
    multi_sites = bosonic_sites(3, 6)
    psi_uniform = coherent_state_mps(multi_sites, 0.3, Truncated())
    @test length(psi_uniform) == 3
    @test abs(norm(psi_uniform) - 1.0) < 1e-8
    
    αs = [0.2, 0.4 + 0.1im, 0.1]
    psi_mixed = coherent_state_mps(multi_sites, αs, Truncated())
    @test length(psi_mixed) == 3
    @test abs(norm(psi_mixed) - 1.0) < 1e-8
    
    @test_throws ErrorException coherent_state_mps(multi_sites, [0.1, 0.2], Truncated())  
end
    @testset "Operator Matrix Elements" begin
        max_occ = 3
        sites = bosonic_sites(1, max_occ)
        site = sites[1]
        
        a_dag = creation_op(site)
        a = annihilation_op(site)
        
        state_0 = bosonic_product_mps(sites, [1], Truncated()) 
        state_1 = bosonic_product_mps(sites, [2], Truncated())  
        state_2 = bosonic_product_mps(sites, [3], Truncated())  
        
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
        sites = bosonic_sites(1, max_occ)
        α = 0.5  
        
        D = displacement_op(sites[1], α)
        @test D isa ITensors.ITensor
        
        @test ITensors.hasinds(D, sites[1]', sites[1])
        
        vac = vacuum_mps(sites, Truncated())
        displaced_vac = ITensors.apply(D, vac.mps)
        
        overlap = abs(ITensors.inner(vac.mps, displaced_vac))
        @test overlap < 1.0  
        @test overlap > 0.0  
    end

    @testset "Production-Ready TEBD Tests" begin
    
    @testset "Direct Gate Construction" begin
        N = 4
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Number operator gates" begin
            ω = 1.0
            
            gate_n = build_evolution_gate(sites, "number", (site=1, omega=ω), dt)
            @test gate_n isa ITensors.ITensor
            @test ITensors.hasinds(gate_n, sites[1]', sites[1])
            
            site = sites[1]
            dim = ITensors.dim(site)
            elements = []
            for i in 1:dim
                push!(elements, gate_n[site'=>i, site=>i])
            end
            

            @test !(all(x -> isapprox(x, elements[1]), elements))
            
            expected_0 = exp(-1im * ω * dt * 0) 
            expected_1 = exp(-1im * ω * dt * 1)  
            
            @test abs(elements[1] - expected_0) < 1e-12
            @test abs(elements[2] - expected_1) < 1e-12
        end
        
        @testset "Hopping gates" begin
            J = 0.1
            
            gate_hop = build_evolution_gate(sites, "hopping", (sites=[1, 2], J=J), dt)
            @test gate_hop isa ITensors.ITensor
            @test ITensors.hasinds(gate_hop, sites[1]', sites[1], sites[2]', sites[2])
            
            site1, site2 = sites[1], sites[2]
            off_diag_element = gate_hop[site1'=>1, site2'=>2, site1=>2, site2=>1]  
            @test abs(off_diag_element) > 1e-10  
        end
        
        @testset "Kerr gates" begin
            χ = 0.1
            
            gate_kerr = build_evolution_gate(sites, "kerr", (site=1, chi=χ), dt)
            @test gate_kerr isa ITensors.ITensor
            @test ITensors.hasinds(gate_kerr, sites[1]', sites[1])
            
            site = sites[1]
            dim = ITensors.dim(site)
            
            for n in 0:(dim-1)
                expected = exp(-1im * χ * dt * n^2)
                actual = gate_kerr[site'=>(n+1), site=>(n+1)]
                @test abs(actual - expected) < 1e-12
            end
        end
    end
    
    @testset "Trotter Gate Construction" begin
        N = 3
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        dt = 0.02
        
        @testset "Simple Hamiltonian" begin
            H_terms = [
                ("number", (site=1, omega=1.0)),
                ("number", (site=2, omega=1.0)),
                ("number", (site=3, omega=1.0)),
                ("hopping", (sites=[1,2], J=0.5)),
                ("hopping", (sites=[2,3], J=0.5))
            ]
            
            for order in [1, 2, 4]
                gates = build_trotter_gates(sites, H_terms, dt; order=order)
                @test gates isa Vector{ITensors.ITensor}
                @test !isempty(gates)
                
                for gate in gates
                    @test gate isa ITensors.ITensor
                end
            end
        end
        
        @testset "Kerr chain" begin
            H_terms = [
                ("number", (site=1, omega=1.0)),
                ("kerr", (site=1, chi=0.1)),
                ("number", (site=2, omega=1.0)),
                ("kerr", (site=2, chi=0.1)),
                ("number", (site=3, omega=1.0)),
                ("kerr", (site=3, chi=0.1))
            ]
            
            gates = build_trotter_gates(sites, H_terms, dt; order=2)
            @test length(gates) == 2 * length(H_terms)  
        end
    end
    
    @testset "Evolution with New Interface" begin
        N = 3
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        
        @testset "Harmonic chain evolution" begin
            dt = 0.02
            
            H_terms = [
                ("number", (site=1, omega=0.1)),
                ("number", (site=2, omega=0.1)),
                ("number", (site=3, omega=0.1)),
                ("hopping", (sites=[1,2], J=0.5)),
                ("hopping", (sites=[2,3], J=0.5))
            ]
            
            gates = build_trotter_gates(sites, H_terms, dt; order=2)
            
            psi0 = bosonic_product_mps(sites, [2, 1, 1], Truncated())  # |1,0,0⟩
            normalize!(psi0)
            
            psi = copy(psi0)
            n_steps = 25
            
            for step in 1:n_steps
                tebd!(psi, gates)
                
                @test abs(norm(psi) - 1.0) < 1e-8
            end
            
            total_particles = 0.0
            for i in 1:N
                expectation = real(ITensorMPS.expect(psi.mps, "N"; sites=i))
                total_particles += expectation
            end
            @test abs(total_particles - 1.0) < 1e-6
            
            particle_on_first = real(ITensorMPS.expect(psi.mps, "N"; sites=1))
            
            @test particle_on_first < 0.95
            
            particle_on_second = real(ITensorMPS.expect(psi.mps, "N"; sites=2))
            particle_on_third = real(ITensorMPS.expect(psi.mps, "N"; sites=3))
            
            @test (particle_on_second + particle_on_third) > 0.05
        end
        
        @testset "Kerr evolution" begin
            sites_single = bosonic_sites(1, 8)
            
            χ = 0.2
            dt = 0.02
            
            H_terms = [("kerr", (site=1, chi=χ))]
            gates = build_trotter_gates(sites_single, H_terms, dt; order=1)
            
            α = 1.0
            psi_coherent = coherent_state_mps(sites_single, α, Truncated())
            normalize!(psi_coherent)
            
            psi_evolved = copy(psi_coherent)
            for step in 1:30
                tebd!(psi_evolved, gates)
            end
            
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
            
            n_initial = real(ITensorMPS.expect(psi_coherent.mps, "N"; sites=1))
            n_final = real(ITensorMPS.expect(psi_evolved.mps, "N"; sites=1))
            @test abs(n_final - n_initial) < 1e-5
            
            overlap = abs(dot(psi_coherent, psi_evolved))
            @test overlap < 0.95
        end
        
        @testset "Simple number operator evolution" begin
            sites_single = bosonic_sites(1, 4)
            
            ω = 1.0
            dt = 0.1
            
            gate_n = build_evolution_gate(sites_single, "number", (site=1, omega=ω), dt)
            
            psi_one = bosonic_product_mps(sites_single, [2], Truncated())  # |1⟩ is index 2
            normalize!(psi_one)
            
            psi_evolved = tebd(psi_one, gate_n)
            
            @test abs(norm(psi_evolved) - 1.0) < 1e-10
            
            n_initial = real(ITensorMPS.expect(psi_one.mps, "N"; sites=1))
            n_final = real(ITensorMPS.expect(psi_evolved.mps, "N"; sites=1))
            @test abs(n_final - n_initial) < 1e-10
            
            overlap = dot(psi_one, psi_evolved)
            expected_phase = exp(-1im * ω * dt * 1)  
            @test abs(overlap - expected_phase) < 1e-10
        end
    end
    
    @testset "Gate Properties" begin
        max_occ = 4
        sites = bosonic_sites(1, max_occ)
        dt = 0.1
        
        @testset "Unitarity check" begin
            gate_n = build_evolution_gate(sites, "number", (site=1, omega=1.0), dt)
            
            site = sites[1]
            dim = ITensors.dim(site)
            
            for n in 1:min(3, dim) 
                state_vec = zeros(ComplexF64, dim)
                state_vec[n] = 1.0
                
                state_tensor = ITensors.ITensor(state_vec, site)
                evolved_state = gate_n * state_tensor
                
                original_norm = norm(state_tensor)
                evolved_norm = norm(evolved_state)
                @test abs(evolved_norm - original_norm) < 1e-10
            end
            
            gate_reverse = build_evolution_gate(sites, "number", (site=1, omega=1.0), -dt)
            
            test_state = ITensors.ITensor(ComplexF64, site)
            test_state[site=>1] = 1.0  
            
            evolved_forward = gate_n * test_state
            evolved_back = gate_reverse * evolved_forward
            
            overlap = abs(ITensors.scalar(ITensors.dag(test_state) * evolved_back))
            @test abs(overlap - 1.0) < 1e-10
        end
        
        @testset "Time reversal symmetry" begin
            gate_forward = build_evolution_gate(sites, "number", (site=1, omega=1.0), dt)
            gate_backward = build_evolution_gate(sites, "number", (site=1, omega=1.0), -dt)
            
            vac = vacuum_mps(sites, Truncated())
            state1 = tebd(vac, gate_forward)
            state2 = tebd(state1, gate_backward)
            
            overlap = abs(dot(vac, state2))
            @test overlap > 0.999
        end
    end
    
    @testset "Input Validation" begin
        sites = bosonic_sites(3, 4)
        dt = 0.01
        
        @testset "build_evolution_gate validation" begin
            @test_throws ArgumentError build_evolution_gate(sites, "invalid", (site=1,), dt)
            
            @test_throws ArgumentError build_evolution_gate(sites, "number", (site=0, omega=1.0), dt)
            @test_throws ArgumentError build_evolution_gate(sites, "number", (site=5, omega=1.0), dt)
            
            @test_throws ArgumentError build_evolution_gate(sites, "hopping", (sites=[1], J=1.0), dt) 
            @test_throws ArgumentError build_evolution_gate(sites, "hopping", (sites=[1,1], J=1.0), dt) 
            
            @test_throws DomainError build_evolution_gate(sites, "number", (site=1, omega=1.0), Inf)
            @test_throws DomainError build_evolution_gate(sites, "number", (site=1, omega=1.0), NaN)
        end
        
        @testset "build_trotter_gates validation" begin
            @test_throws ArgumentError build_trotter_gates(ITensors.Index[], [], dt)
            @test_throws ArgumentError build_trotter_gates(sites, [], dt)
            
            H_terms = [("number", (site=1, omega=1.0))]
            @test_throws ArgumentError build_trotter_gates(sites, H_terms, dt; order=3)
            @test_throws ArgumentError build_trotter_gates(sites, H_terms, dt; order=0)
            
            @test_throws ArgumentError build_trotter_gates(sites, ["invalid"], dt)
            @test_throws ArgumentError build_trotter_gates(sites, [(1, 2, 3)], dt) 
        end
    end
    
    @testset "Backwards Compatibility" begin
        N = 2
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        
        psi = random_bmps(sites, Truncated(); linkdim=4)
        normalize!(psi)
        
        id_gate = ITensors.op("Id", sites[1])
        
        psi_evolved = tebd(psi, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi
        
        psi_inplace = copy(psi)
        tebd!(psi_inplace, id_gate)
        @test abs(norm(psi_inplace) - 1.0) < 1e-10
    end
end
end