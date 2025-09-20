using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Fixed TEBD Gate Construction Tests" begin
    
    @testset "build_evolution_gate - Basic Functionality" begin
        N = 4
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Single-site gates" begin
            # Test single-site number operator gate
            opsum_n = ITensors.OpSum()
            opsum_n += 1.0, "N", 1
            
            gate_n = build_evolution_gate(sites, opsum_n, dt, [1])
            @test gate_n isa ITensors.ITensor
            @test ITensors.hasinds(gate_n, sites[1]', sites[1])
            
            # Test single-site creation-annihilation gate
            opsum_adag_a = ITensors.OpSum()
            opsum_adag_a += 1.0, "Adag", 2
            opsum_adag_a += 1.0, "A", 2
            
            gate_adag_a = build_evolution_gate(sites, opsum_adag_a, dt, [2])
            @test gate_adag_a isa ITensors.ITensor
            @test ITensors.hasinds(gate_adag_a, sites[2]', sites[2])
        end
        
        @testset "Two-site gates" begin
            # Test hopping gate
            opsum_hop = ITensors.OpSum()
            opsum_hop += 0.1, "Adag", 1, "A", 2
            opsum_hop += 0.1, "A", 1, "Adag", 2
            
            gate_hop = build_evolution_gate(sites, opsum_hop, dt, [1, 2])
            @test gate_hop isa ITensors.ITensor
            @test ITensors.hasinds(gate_hop, sites[1]', sites[1], sites[2]', sites[2])
        end
        
        @testset "Time reversal symmetry" begin
            # Test that exp(-i*dt*H) * exp(i*dt*H) ≈ I
            opsum = ITensors.OpSum()
            opsum += 1.0, "N", 1
            
            gate_forward = build_evolution_gate(sites, opsum, dt, [1])
            gate_backward = build_evolution_gate(sites, opsum, -dt, [1])
            
            # Apply both gates to vacuum state
            vac = vacuum_mps(sites, Truncated())
            state1 = tebd(vac, gate_forward)
            state2 = tebd(state1, gate_backward)
            
            # Should get back original state
            overlap = abs(dot(vac, state2))
            @test overlap > 0.999  # Very close to 1, allowing for numerical errors
        end
    end
    
    @testset "build_evolution_gate - Error Handling" begin
        N = 3
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Input validation" begin
            opsum = ITensors.OpSum()
            opsum += 1.0, "N", 1
            
            # Empty sites
            @test_throws ArgumentError build_evolution_gate(ITensors.Index[], opsum, dt, [1])
            
            # Empty gate_sites
            @test_throws ArgumentError build_evolution_gate(sites, opsum, dt, Int[])
            
            # Non-finite dt
            @test_throws DomainError build_evolution_gate(sites, opsum, Inf, [1])
            @test_throws DomainError build_evolution_gate(sites, opsum, NaN, [1])
            
            # Out-of-range gate sites
            @test_throws ArgumentError build_evolution_gate(sites, opsum, dt, [0])
            @test_throws ArgumentError build_evolution_gate(sites, opsum, dt, [N+1])
            @test_throws ArgumentError build_evolution_gate(sites, opsum, dt, [-1])
            
            # Duplicate gate sites
            @test_throws ArgumentError build_evolution_gate(sites, opsum, dt, [1, 1])
            
            # Non-contiguous multi-site gates
            opsum_multisite = ITensors.OpSum()
            opsum_multisite += 1.0, "Adag", 1, "A", 3  # Sites 1 and 3 are not contiguous
            @test_throws ArgumentError build_evolution_gate(sites, opsum_multisite, dt, [1, 3])
        end
    end
    
    @testset "build_trotter_gates - Basic Functionality" begin
        N = 4
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Single term, different orders" begin
            # Simple harmonic oscillator terms - create as tuples directly
            H_terms = []
            
            # On-site terms
            opsum1 = ITensors.OpSum()
            opsum1 += 1.0, "N", 1
            push!(H_terms, (opsum1, [1]))
            
            opsum2 = ITensors.OpSum()
            opsum2 += 1.0, "N", 2  
            push!(H_terms, (opsum2, [2]))
            
            # Hopping term
            opsum_hop = ITensors.OpSum()
            opsum_hop += 0.1, "Adag", 1, "A", 2
            opsum_hop += 0.1, "A", 1, "Adag", 2
            push!(H_terms, (opsum_hop, [1, 2]))
            
            # Test order 1
            gates_1 = build_trotter_gates(sites, H_terms, dt; order=1)
            @test length(gates_1) == 3
            @test all(g -> g isa ITensors.ITensor, gates_1)
            
            # Test order 2
            gates_2 = build_trotter_gates(sites, H_terms, dt; order=2)
            @test length(gates_2) == 6  # 3 forward + 3 backward
            @test all(g -> g isa ITensors.ITensor, gates_2)
            
            # Test order 4
            gates_4 = build_trotter_gates(sites, H_terms, dt; order=4)
            @test length(gates_4) == 12  # More complex decomposition
            @test all(g -> g isa ITensors.ITensor, gates_4)
        end
        
        @testset "Evolution accuracy comparison" begin
            # Simple two-site system for exact comparison
            N_small = 2
            sites_small = bosonic_sites(N_small, 6)
            
            # Build H_terms properly
            H_terms = []
            
            opsum1 = ITensors.OpSum()
            opsum1 += 1.0, "N", 1
            push!(H_terms, (opsum1, [1]))
            
            opsum2 = ITensors.OpSum()
            opsum2 += 1.0, "N", 2
            push!(H_terms, (opsum2, [2]))
            
            opsum_hop = ITensors.OpSum()
            opsum_hop += 0.1, "Adag", 1, "A", 2
            opsum_hop += 0.1, "A", 1, "Adag", 2
            push!(H_terms, (opsum_hop, [1, 2]))
            
            # Initial state
            psi0 = bosonic_product_mps(sites_small, [2, 1], Truncated())  # |1,0⟩
            normalize!(psi0)
            
            dt_small = 0.001  # Very small time step
            
            # Compare different Trotter orders
            gates_1 = build_trotter_gates(sites_small, H_terms, dt_small; order=1)
            gates_2 = build_trotter_gates(sites_small, H_terms, dt_small; order=2)
            
            psi_1 = tebd(psi0, gates_1)
            psi_2 = tebd(psi0, gates_2)
            
            # Both should preserve norm
            @test abs(norm(psi_1) - 1.0) < 1e-10
            @test abs(norm(psi_2) - 1.0) < 1e-10
            
            # For very small dt, both should be close to each other
            overlap_12 = abs(dot(psi_1, psi_2))
            @test overlap_12 > 0.99  # Should be very similar for small dt
        end
    end
    
    @testset "build_trotter_gates - Error Handling" begin
        N = 3
        max_occ = 4
        sites = bosonic_sites(N, max_occ)
        dt = 0.01
        
        @testset "Input validation" begin
            # Create a valid H_terms first
            opsum = ITensors.OpSum()
            opsum += 1.0, "N", 1
            H_terms = [(opsum, [1])]
            
            # Empty sites
            @test_throws ArgumentError build_trotter_gates(ITensors.Index[], H_terms, dt)
            
            # Empty H_terms
            @test_throws ArgumentError build_trotter_gates(sites, Tuple{Any,Vector{Int}}[], dt)
            
            # Non-finite dt
            @test_throws DomainError build_trotter_gates(sites, H_terms, Inf)
            @test_throws DomainError build_trotter_gates(sites, H_terms, NaN)
            
            # Unsupported order
            @test_throws ArgumentError build_trotter_gates(sites, H_terms, dt; order=3)
            @test_throws ArgumentError build_trotter_gates(sites, H_terms, dt; order=0)
        end
        
        @testset "H_terms validation" begin
            # Empty site indices in H_terms
            opsum = ITensors.OpSum()
            opsum += 1.0, "N", 1
            H_empty = [(opsum, Int[])]
            @test_throws ArgumentError build_trotter_gates(sites, H_empty, dt)
            
            # Out-of-range site indices
            H_range = [(opsum, [N+1])]
            @test_throws ArgumentError build_trotter_gates(sites, H_range, dt)
            
            # Duplicate site indices within a term
            H_dup = [(opsum, [1, 1])]
            @test_throws ArgumentError build_trotter_gates(sites, H_dup, dt)
        end
    end
    
    @testset "Integration with TEBD" begin
        N = 3
        max_occ = 5
        sites = bosonic_sites(N, max_occ)
        
        @testset "Harmonic chain evolution" begin
            # Build harmonic chain Hamiltonian properly
            H_terms = []
            
            # On-site terms
            for i in 1:N
                opsum_i = ITensors.OpSum()
                opsum_i += 1.0, "N", i
                push!(H_terms, (opsum_i, [i]))
            end
            
            # Hopping terms
            for i in 1:(N-1)
                hop_term = ITensors.OpSum()
                hop_term += 0.1, "Adag", i, "A", i+1
                hop_term += 0.1, "A", i, "Adag", i+1
                push!(H_terms, (hop_term, [i, i+1]))
            end
            
            # Build gates
            dt = 0.01
            gates = build_trotter_gates(sites, H_terms, dt; order=2)
            
            # Initial state: one particle on first site
            psi0 = bosonic_product_mps(sites, [2, 1, 1], Truncated())  # |1,0,0⟩
            normalize!(psi0)
            
            # Evolve for a few steps
            psi = copy(psi0)
            n_steps = 10
            
            for step in 1:n_steps
                tebd!(psi, gates)
                
                # Check normalization is preserved
                @test abs(norm(psi) - 1.0) < 1e-8
                
                # Check total particle number is conserved (approximately)
                total_particles = 0.0
                for i in 1:N
                    expectation = real(ITensorMPS.expect(psi.mps, "N"; sites=i))  # Simplified: use string operator
                    total_particles += expectation
                end
                @test abs(total_particles - 1.0) < 1e-6  # One particle total
            end
            
            # After evolution, particle should have spread
            psi_final = psi
            particle_on_first = real(ITensorMPS.expect(psi_final.mps, "N"; sites=1))  # Simplified: use string operator
            @test particle_on_first < 0.9  # Should be less than initial value of 1.0
        end
        
        @testset "Kerr evolution" begin
            # Single-site Kerr evolution
            sites_single = bosonic_sites(1, 8)
            
            # Kerr Hamiltonian: H = χ * n²
            χ = 0.1
            opsum_kerr = ITensors.OpSum()
            opsum_kerr += χ, "N", 1, "N", 1
            H_kerr = [(opsum_kerr, [1])]
            
            dt = 0.01
            gates_kerr = build_trotter_gates(sites_single, H_kerr, dt; order=2)
            
            # Initial coherent state (approximate)
            α = 0.5
            psi_coherent = coherent_state_mps(sites_single, α, Truncated())
            normalize!(psi_coherent)
            
            # Evolve and check that particle number expectation changes
            n_initial = real(ITensorMPS.expect(psi_coherent.mps, "N"; sites=1))  # Simplified: use string operator
            
            psi_evolved = copy(psi_coherent)
            for step in 1:20
                tebd!(psi_evolved, gates_kerr)
            end
            
            # Norm should be preserved
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
            
            # Particle number expectation should remain approximately the same
            n_final = real(ITensorMPS.expect(psi_evolved.mps, "N"; sites=1))  # Simplified: use string operator
            @test abs(n_final - n_initial) < 1e-6
            
            # But the state should have evolved (phases changed)
            overlap = abs(dot(psi_coherent, psi_evolved))
            @test overlap < 0.99  # Should be noticeably different
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
        
        @testset "_validate_opsum_sites" begin
            # Valid cases should not throw
            opsum1 = ITensors.OpSum()
            opsum1 += 1.0, "N", 1
            @test_nowarn Mabs._validate_opsum_sites(opsum1, [1])
            
            opsum2 = ITensors.OpSum()
            opsum2 += 1.0, "Adag", 1, "A", 2
            @test_nowarn Mabs._validate_opsum_sites(opsum2, [1, 2])
            @test_nowarn Mabs._validate_opsum_sites(opsum2, [1, 2, 3])  # Extra sites OK
            
            # Test with nothing (should throw)
            @test_throws ArgumentError Mabs._validate_opsum_sites(nothing, [1])
        end
    end
    
    @testset "Performance and Scaling" begin
        @testset "Large system construction" begin
            # Test that gate construction scales reasonably
            N_large = 20
            max_occ_small = 3  # Keep Hilbert space manageable
            sites_large = bosonic_sites(N_large, max_occ_small)
            
            # Build nearest-neighbor Hamiltonian
            H_terms_large = []
            
            # On-site terms
            for i in 1:N_large
                opsum_i = ITensors.OpSum()
                opsum_i += 1.0, "N", i
                push!(H_terms_large, (opsum_i, [i]))
            end
            
            # Nearest-neighbor terms
            for i in 1:(N_large-1)
                hop_term = ITensors.OpSum()
                hop_term += 0.1, "Adag", i, "A", i+1
                hop_term += 0.1, "A", i, "Adag", i+1
                push!(H_terms_large, (hop_term, [i, i+1]))
            end
            
            # This should complete without errors
            dt = 0.01
            @time gates_large = build_trotter_gates(sites_large, H_terms_large, dt; order=2)
            
            expected_length = 2 * length(H_terms_large)  # Forward + backward for order 2
            @test length(gates_large) == expected_length
            @test all(g -> g isa ITensors.ITensor, gates_large)
        end
    end
end