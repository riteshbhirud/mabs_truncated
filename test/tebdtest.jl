using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Production-Ready TEBD Tests" begin
    
    @testset "Direct Gate Construction" begin
        N = 4
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Number operator gates" begin
            # Build number operator gate directly
            site = sites[1]
            ω = 1.0
            
            gate_n = build_simple_gates(sites, "number", (site=1, omega=ω), dt)
            @test gate_n isa ITensors.ITensor
            @test ITensors.hasinds(gate_n, site', site)
            
            # Test that it's not the identity (should have different diagonal elements)
            dim = ITensors.dim(site)
            elements = []
            for i in 1:dim
                push!(elements, gate_n[site'=>i, site=>i])
            end
            
            # For number operator evolution, diagonal elements should be exp(-i*ω*dt*n)
            # where n = 0, 1, 2, ... so they should be different
            @test !(all(x -> isapprox(x, elements[1]), elements))
            
            # Check specific values
            expected_0 = exp(-1im * ω * dt * 0)  # |0⟩ element
            expected_1 = exp(-1im * ω * dt * 1)  # |1⟩ element
            
            @test abs(elements[1] - expected_0) < 1e-12
            @test abs(elements[2] - expected_1) < 1e-12
        end
        
        @testset "Hopping gates" begin
            # Build hopping gate directly
            J = 0.1
            
            gate_hop = build_simple_gates(sites, "hopping", (sites=[1, 2], J=J), dt)
            @test gate_hop isa ITensors.ITensor
            @test ITensors.hasinds(gate_hop, sites[1]', sites[1], sites[2]', sites[2])
            
            # Test that it's not the identity by checking some off-diagonal elements
            site1, site2 = sites[1], sites[2]
            # For hopping, we should have off-diagonal terms
            off_diag_element = gate_hop[site1'=>1, site2'=>2, site1=>2, site2=>1]  # |0,1⟩ → |1,0⟩
            @test abs(off_diag_element) > 1e-10  # Should be non-zero for hopping
        end
        
        @testset "Kerr gates" begin
            # Build Kerr gate directly
            χ = 0.1
            
            gate_kerr = build_simple_gates(sites, "kerr", (site=1, chi=χ), dt)
            @test gate_kerr isa ITensors.ITensor
            @test ITensors.hasinds(gate_kerr, sites[1]', sites[1])
            
            # Test diagonal elements for Kerr evolution
            site = sites[1]
            dim = ITensors.dim(site)
            
            for n in 0:(dim-1)
                expected = exp(-1im * χ * dt * n^2)
                actual = gate_kerr[site'=>(n+1), site=>(n+1)]
                @test abs(actual - expected) < 1e-12
            end
        end
    end
    
    @testset "Evolution with Direct Gates" begin
        N = 3
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        
        @testset "Harmonic chain evolution" begin
            dt = 0.02
            
            # Build gates directly
            gates = ITensors.ITensor[]
            
            # On-site number operator terms
            for i in 1:N
                gate_n = build_simple_gates(sites, "number", (site=i, omega=0.1), dt/2)
                push!(gates, gate_n)
            end
            
            # Hopping terms
            for i in 1:(N-1)
                gate_hop = build_simple_gates(sites, "hopping", (sites=[i, i+1], J=0.5), dt/2)
                push!(gates, gate_hop)
            end
            
            # Add reverse gates for second-order Trotter
            for i in (N-1):-1:1
                gate_hop = build_simple_gates(sites, "hopping", (sites=[i, i+1], J=0.5), dt/2)
                push!(gates, gate_hop)
            end
            
            for i in N:-1:1
                gate_n = build_simple_gates(sites, "number", (site=i, omega=0.1), dt/2)
                push!(gates, gate_n)
            end
            
            # Initial state: one particle on first site
            psi0 = bosonic_product_mps(sites, [2, 1, 1], Truncated())  # |1,0,0⟩
            normalize!(psi0)
            
            # Evolve for several steps
            psi = copy(psi0)
            n_steps = 25
            
            for step in 1:n_steps
                tebd!(psi, gates)
                
                # Check normalization is preserved
                @test abs(norm(psi) - 1.0) < 1e-8
            end
            
            # Check total particle number is conserved
            total_particles = 0.0
            for i in 1:N
                expectation = real(ITensorMPS.expect(psi.mps, "N"; sites=i))
                total_particles += expectation
            end
            @test abs(total_particles - 1.0) < 1e-6
            
            # After evolution, particle should have spread
            particle_on_first = real(ITensorMPS.expect(psi.mps, "N"; sites=1))
            
            # With hopping, particle should spread
            @test particle_on_first < 0.95
            
            # Check that particle has moved to other sites
            particle_on_second = real(ITensorMPS.expect(psi.mps, "N"; sites=2))
            particle_on_third = real(ITensorMPS.expect(psi.mps, "N"; sites=3))
            
            @test (particle_on_second + particle_on_third) > 0.05
        end
        
        @testset "Kerr evolution" begin
            # Single-site Kerr evolution
            sites_single = bosonic_sites(1, 8)
            
            χ = 0.2
            dt = 0.02
            
            # Build Kerr gate
            gate_kerr = build_simple_gates(sites_single, "kerr", (site=1, chi=χ), dt)
            
            # Initial coherent state (approximate)
            α = 1.0
            psi_coherent = coherent_state_mps(sites_single, α, Truncated())
            normalize!(psi_coherent)
            
            # Evolve
            psi_evolved = copy(psi_coherent)
            for step in 1:30
                tebd!(psi_evolved, gate_kerr)
            end
            
            # Norm should be preserved
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
            
            # Particle number expectation should remain the same
            n_initial = real(ITensorMPS.expect(psi_coherent.mps, "N"; sites=1))
            n_final = real(ITensorMPS.expect(psi_evolved.mps, "N"; sites=1))
            @test abs(n_final - n_initial) < 1e-5
            
            # But the state should have evolved (phases changed)
            overlap = abs(dot(psi_coherent, psi_evolved))
            @test overlap < 0.95
        end
        
        @testset "Simple number operator evolution" begin
            # Test the simplest case: single-site number operator evolution
            sites_single = bosonic_sites(1, 4)
            
            ω = 1.0
            dt = 0.1
            
            # Build number operator gate
            gate_n = build_simple_gates(sites_single, "number", (site=1, omega=ω), dt)
            
            # Initial state: |1⟩
            psi_one = bosonic_product_mps(sites_single, [2], Truncated())  # |1⟩ is index 2
            normalize!(psi_one)
            
            # Evolution should only add a phase: |1⟩ → exp(-iωt)|1⟩
            psi_evolved = tebd(psi_one, gate_n)
            
            # Norm should be preserved
            @test abs(norm(psi_evolved) - 1.0) < 1e-10
            
            # Particle number should be unchanged
            n_initial = real(ITensorMPS.expect(psi_one.mps, "N"; sites=1))
            n_final = real(ITensorMPS.expect(psi_evolved.mps, "N"; sites=1))
            @test abs(n_final - n_initial) < 1e-10
            
            # Check the phase evolution
            overlap = dot(psi_one, psi_evolved)
            expected_phase = exp(-1im * ω * dt * 1)  # For |1⟩, adds phase exp(-iωdt)
            @test abs(overlap - expected_phase) < 1e-10
        end
    end
    
    @testset "Gate Properties" begin
        max_occ = 4
        sites = bosonic_sites(1, max_occ)
        dt = 0.1
        
        @testset "Unitarity check" begin
            # Number operator gate should be unitary
            gate_n = build_simple_gates(sites, "number", (site=1, omega=1.0), dt)
            
            # Check unitarity by verifying that U preserves norms and is invertible
            site = sites[1]
            dim = ITensors.dim(site)
            
            # Test 1: Check that gate preserves norms when applied to states
            for n in 1:min(3, dim)  # Test a few states
                # Create a state |n⟩
                state_vec = zeros(ComplexF64, dim)
                state_vec[n] = 1.0
                
                state_tensor = ITensors.ITensor(state_vec, site)
                evolved_state = gate_n * state_tensor
                
                # Norm should be preserved
                original_norm = norm(state_tensor)
                evolved_norm = norm(evolved_state)
                @test abs(evolved_norm - original_norm) < 1e-10
            end
            
            # Test 2: Check that evolution is reversible
            gate_reverse = build_simple_gates(sites, "number", (site=1, omega=1.0), -dt)
            
            # Create a test state
            test_state = ITensors.ITensor(ComplexF64, site)
            test_state[site=>1] = 1.0  # |0⟩ state
            
            # Apply forward then reverse evolution
            evolved_forward = gate_n * test_state
            evolved_back = gate_reverse * evolved_forward
            
            # Should get back original state
            overlap = abs(ITensors.scalar(ITensors.dag(test_state) * evolved_back))
            @test abs(overlap - 1.0) < 1e-10
        end
        
        @testset "Time reversal symmetry" begin
            # Test that exp(-i*dt*H) * exp(i*dt*H) ≈ I
            gate_forward = build_simple_gates(sites, "number", (site=1, omega=1.0), dt)
            gate_backward = build_simple_gates(sites, "number", (site=1, omega=1.0), -dt)
            
            # Apply both gates to vacuum state
            vac = vacuum_mps(sites, Truncated())
            state1 = tebd(vac, gate_forward)
            state2 = tebd(state1, gate_backward)
            
            # Should get back original state
            overlap = abs(dot(vac, state2))
            @test overlap > 0.999
        end
    end
    
    @testset "Helper Functions" begin
        @testset "_is_contiguous" begin
            @test Mabs._is_contiguous([1])
            @test Mabs._is_contiguous([1, 2])
            @test Mabs._is_contiguous([1, 2, 3])
            @test Mabs._is_contiguous([2, 3, 4])
            
            @test !Mabs._is_contiguous([1, 3])
            @test !Mabs._is_contiguous([1, 2, 4])
            @test !Mabs._is_contiguous([1, 3, 5])
            
            # Order shouldn't matter
            @test Mabs._is_contiguous([3, 1, 2])
            @test !Mabs._is_contiguous([3, 1, 5])
            
            # Edge cases
            @test Mabs._is_contiguous(Int[])  # Empty
        end
    end
    
    @testset "Backwards Compatibility" begin
        # Test that basic TEBD operations still work
        N = 2
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        
        psi = random_bmps(sites, Truncated(); linkdim=4)
        normalize!(psi)
        
        # Create a simple identity gate
        id_gate = ITensors.op("Id", sites[1])
        
        # Test non-mutating evolution
        psi_evolved = tebd(psi, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi
        
        # Test in-place evolution
        psi_inplace = copy(psi)
        tebd!(psi_inplace, id_gate)
        @test abs(norm(psi_inplace) - 1.0) < 1e-10
    end
end