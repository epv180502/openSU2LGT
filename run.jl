using ITensors
using ITensorMPS
using LinearAlgebra
using HDF5
using SparseArrays
using Arpack
using TupleTools
#using OpenQuantumTools
using Statistics
using ArgParse
include("utilities.jl")

"""
    parseCommandline()

Parse the command line arguments in a convenient way
"""
function parseCommandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N"
        help = "the number of sites"
        arg_type = Int
        required = true

        "--g2"
        help = "parameter g squared of the the Hamiltonian"
        arg_type = Float64
        required = true

        "--m"
        help = "parameter m of the the Hamiltonian"
        arg_type = Float64
        required = true

        "--T"
        help = "Temperature"
        arg_type = Float64
        required = true

        "--D"
        help = "Dissipator"
        arg_type = Float64
        required = true

        "--tau"
        help = "time step"
        arg_type = Float64
        required = true

        "--maxdim"
        help = "maximum bond dimension"
        arg_type = Int
        required = true

        "--len"
        help = "string length"
        arg_type = Int
        required = true

        "--a"
        help = "parameter a of the the Hamiltonian"
        arg_type = Float64
        default = 1

        "--lambda"
        help = "penalty strength"
        arg_type = Float64
        default = 0.0

        # "--nsteps"
        # help = "number of steps"
        # arg_type = Int
        # default = 100

        "--tF"
        help = "final time"
        arg_type = Float64
        default = 1.0

        "--l_0"
        help = "background field"
        arg_type = Float64
        default = 0.0

        "--env_corr_type"
        help = "type of environment correlator"
        arg_type = String
        default = "delta"

        "--teo"
        help = "order of Taylor expansion"
        arg_type = Int
        default = 1

        "--tec_1"
        help = "cutoff 1 for Taylor expansion"
        arg_type = Float64
        default = 1E-5

        "--tec_2"
        help = "cutoff 2 for Taylor expansion"
        arg_type = Float64
        default = 1E-5

        "--cutoff"
        help = "cutoff for evolution"
        arg_type = Float64
        default = 1E-6

        "--folder"
        help = "path to folder where data will be stored"
        arg_type = String
        default = "./"

        "--qns"
        help = "conserve quantum numbers"
        arg_type = Bool
        default = false

        "--T2"
        help = "how T2_n is calculated"
        arg_type = Symbol
        default = :Efficient 

        "--mEnergy"
        help = "measure energy"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end

"""
    evolve()

Function performing the actual time evolution
"""
function evolve()
    start_tot = time()

    # ------------------------------------- Setup parameters of the simulation -------------------------------------
    # Parse the arguments from the command line
    parsedArgs = parseCommandline()
    N = parsedArgs["N"]                             # Number of staggered sites
    g2 = parsedArgs["g2"]                           # Gauge coupling
    m = parsedArgs["m"]                             # Mass
    a = parsedArgs["a"]                             # Lattice spacing
    lambda = parsedArgs["lambda"]                   # Gauge protection term
    T = parsedArgs["T"]                             # Temperature
    D = parsedArgs["D"]                             # Self-correlation
    env_corr_type = parsedArgs["env_corr_type"]     # Type of correlator
    l_0 = parsedArgs["l_0"]                         # Background electric field
    # number_of_time_steps = parsedArgs["nsteps"]     # Number of timesteps (does not include the 0th time)
    final_time = parsedArgs["tF"]                   # Final time to be reached
    len = parsedArgs["len"]                         # Length of the string 
    tau = parsedArgs["tau"]                         # Timestep
    dissipator_sites = collect(1:N)                 # Site to be acted upon by the env. Defaulted to all
    taylor_expansion_order = parsedArgs["teo"]      # Order of Taylor expansion
    taylor_expansion_cutoff_1 = parsedArgs["tec_1"] # Cutoff 1 for Taylor expansion
    taylor_expansion_cutoff_2 = parsedArgs["tec_2"] # Cutoff 2 for Taylor expansion
    cutoff = parsedArgs["cutoff"]                   # Cutoff of the SVD. Maximum normalization to be truncated in SVD
    maxdim = parsedArgs["maxdim"]                   # Maximum bond dimension
    folder = parsedArgs["folder"]                   # Folder to store results        
    quantum_number_flag = parsedArgs["qns"]         # Whether to conserve quantum numbers
    compute_T2 = parsedArgs["T2"]                   # How to calculate the electric field
    measure_energy_flag = parsedArgs["mEnergy"]     # Whether to measure energy

    number_of_time_steps = Int(round((final_time - 0) / tau))

    # Print the parameters we parsed
    println("Parsed args:")
    for (arg, val) in parsedArgs
        println("$arg  =>  $val")
    end

    # Construct filename 
    simulation_desc = string("SU2_timeSim")
    system_desc = string("_N", N, "a", a, "g", g2, "m", m, "D", D, "T", T, "env", env_corr_type)
    dyn_desc = string("_dt", tau, "nsteps", number_of_time_steps, "len", len)
    TNparameters_desc = string("_chi", maxdim, "SVD", cutoff, "teo", taylor_expansion_order, "tec", taylor_expansion_cutoff_1, "y", taylor_expansion_cutoff_2, "QN", quantum_number_flag, "T2", compute_T2)
    path_to_results = string(folder, simulation_desc, system_desc, dyn_desc, TNparameters_desc, ".h5")
    println("Results will be saved in $path_to_results")
    flush(stdout)

    # ------------------------------------- Create the initial state -------------------------------------
    start = time()
    # Sites which need to be flipped
    ind_start = Int(round(N / 2 - len / 2)) + 1
    flip_sites = collect(ind_start:ind_start+len-1) # If len = 0, this will not flip any sites and effectively return the vacuum 

    # Prepare the initial state
    "Local helper function to get the Dirac vacuum with a string of lenght 'len' on top"
    function get_initial_state()
        # Prepare the dirac vacuum with a string on top
        sites_initial_state = siteinds("SU2_packed", N; conserve_qns=quantum_number_flag)
        
        # This will be a normal non-purified MPS
        if len == 0 
            psi = get_dirac_vacuum_mps(sites_initial_state; flip_sites)
        else
            psi = get_string_on_dirac_vacuum_mps(sites_initial_state, len)
            println("USING STRING STATE")
        end
        println("Initial State Links = $(linkdims(psi))")
        rho = outer(psi', psi) # Get the density matrix
        rho_vec = convert(MPS, rho) # Convert the density matrix to a purified MPS
        mps = rho_vec_to_mps(rho_vec) # Split the sites so that each site has one physical index of dimension 4
        orthogonalize!(mps, 1) # Bring the center of orthogonalization to the very left        
        return mps

    end
    mps = get_initial_state()
    sites = siteinds(mps)
    
    println("Finished getting the initial state in $(time() - start) seconds")
    flush(stdout)
    # ------------------------------------- Create the hamiltonian -------------------------------------
    # Get the taylor, odd and even opsum groups without the l0 terms
    start = time()
    
    side = "left"
    H = get_double_aH_Hamiltonian(sites, g2, m, a, side)
    H = MPO(H, sites)
    H_kin, H_el, H_m = get_double_aH_Hamiltonian_individual_terms(N, g2, m, a, side)
    H_kin, H_el, H_m = MPO(H_kin, sites), MPO(H_el, sites), MPO(H_m, sites)
    println("Finished getting the odd, even and taylor MPO's in $(time() - start) seconds")
    flush(stdout)

    start = time()

    # Create the MPO's that will be needed for calculating T2_n
    if compute_T2 === :Full
        T2n = [MPO(get_T2n(n), sites) for n in 1:N]
    elseif compute_T2 === :Separate
        Q2_mpos = [MPO(OpSum() + (1, "Q2", 2p-1), sites) for p in 1:N]
        Cxx_mpos = Matrix{MPO}(undef, N, N)
        Cyy_mpos = Matrix{MPO}(undef, N, N)
        Czz_mpos = Matrix{MPO}(undef, N, N)
        for p in 1:N
            for q in p+1:N
                Cxx_mpos[p,q] = MPO(OpSum() + (1, "Qx", 2p-1, "Qx", 2q-1), sites)
                Cyy_mpos[p,q] = MPO(OpSum() + (1, "Qy", 2p-1, "Qy", 2q-1), sites)
                Czz_mpos[p,q] = MPO(OpSum() + (1, "Qz", 2p-1, "Qz", 2q-1), sites)
            end
        end
    # elseif compute_T2 === :Efficient
    #    Do nothing
    end

    println("Finished creating the T2n MPO's in $(time() - start) seconds")
    flush(stdout)
    
    # This is done so that the odd, even gates and taylor MPO have physical legs matching the purified MPS and 
    # combining this with the swapprime done on the operators later the transpose is taken on the operators acting 
    # on the even sites which correspond to operators acting on the right of the density matrix
    for i in 2:2:length(sites) 
        sites[i] = dag(sites[i])
    end

    start = time()
    opsum_without_l0_terms = get_Lindblad_opsum_without_l0_terms(sites, g2, m, a, T, D, env_corr_type, dissipator_sites)
    nn_odd_without_l0_terms, nn_even_without_l0_terms, taylor = get_odd_even_taylor_groups(opsum_without_l0_terms, sites)

    # Gather the two odd and even gates
    odd = get_odd(sites, tau / 2, nn_odd_without_l0_terms)
    even = get_even(sites, tau, nn_even_without_l0_terms)
    println("Finished getting the odd, even and taylor gates without the l0 terms in $(time() - start) seconds")
    flush(stdout)

    # Get the MPO for the Taylor expansion
    start = time()
    taylor_mpo_tmp = 0.5 * tau * MPO(taylor, sites)
    truncate!(taylor_mpo_tmp; cutoff=taylor_expansion_cutoff_1)

    for i in 2:2:length(taylor_mpo_tmp) # Transpose the MPO on the even sites which would correspond to the bottom legs of the MPO
        taylor_mpo_tmp[i] = swapprime(taylor_mpo_tmp[i], 0, 1; :tags => "Site")
    end

    taylor_mpo = get_mpo_taylor_expansion(taylor_mpo_tmp, taylor_expansion_order, taylor_expansion_cutoff_2, sites)
    println("The taylor_mpo with taylor order $(taylor_expansion_order) and cutoffs $(taylor_expansion_cutoff_1), $(taylor_expansion_cutoff_2) has bond dimensions ", linkdims(taylor_mpo))
    println("Finished getting taylor MPO in $(time() - start) seconds")
    flush(stdout)

    # ------------------------------------- Get the observables to be measured -------------------------------------
    # Starting the lists of the observables we want to keep track of
    start = time()
    single_configs = zeros(ComplexF64, number_of_time_steps + 1, N) 
    single_configs[1, :] = measure_op_config(mps, "N_single")
    pair_configs = zeros(ComplexF64, number_of_time_steps + 1, N) 
    pair_configs[1, :] = measure_op_config(mps, "N_pair")
    zero_configs = zeros(ComplexF64, number_of_time_steps + 1, N) 
    zero_configs[1, :] = measure_op_config(mps, "N_zero")
    total_configs = zeros(ComplexF64, number_of_time_steps + 1, N) 
    total_configs[1, :] = measure_op_config(mps, "N_tot")
    link_dims = zeros(Int64, number_of_time_steps + 1, 2 * N - 1)
    link_dims[1, :] = linkdims(mps)
    energy = zeros(ComplexF64, number_of_time_steps + 1)
    energy[1] = measure_mpo(mps, H)
    kin_energy = zeros(ComplexF64, number_of_time_steps + 1)
    kin_energy[1] = measure_mpo(mps, H_kin; alg="naive")
    m_energy = zeros(ComplexF64, number_of_time_steps + 1)
    m_energy[1] = measure_mpo(mps, H_m; alg="naive")
    el_energy = zeros(ComplexF64, number_of_time_steps + 1)
    el_energy[1] = measure_mpo(mps, H_el; alg="naive")

    T2_configs = zeros(ComplexF64, number_of_time_steps + 1, N) 
    if compute_T2 === :Full
        for (n, T2) in enumerate(T2n)
            T2_configs[1,n] = measure_mpo(mps, T2; alg="naive")
        end
    elseif compute_T2 === :Separate
        T2_configs[1, :] = measure_T2_configs(mps, N, Q2_mpos, Cxx_mpos, Cyy_mpos, Czz_mpos)
    elseif compute_T2 === :Efficient
        T2_configs[1, :] = measure_T2_sweep(mps)
    end

    println("Finished getting the lists for the tracked observables in $(time() - start) seconds")
    flush(stdout)

    # Open the HDF5 file for the results
    results_file = h5open(path_to_results, "w")

    # ------------------------------------- Run Simulation -------------------------------------
    # Code with time overview for debugging/optimization
    time_start = time()
    for step in 1:number_of_time_steps
        start = time()

        # Time evolution
        t_odd1 = @elapsed apply_odd!(odd, mps, cutoff, maxdim)
        t_taylor1 = @elapsed mps = apply(taylor_mpo, mps; cutoff=cutoff, maxdim=maxdim)
        t_even = @elapsed apply_even!(even, mps, cutoff, maxdim)
        t_taylor2 = @elapsed mps = apply(taylor_mpo, mps; cutoff=cutoff, maxdim=maxdim)
        t_odd2 = @elapsed apply_odd!(odd, mps, cutoff, maxdim)

        # Normalizing the purified MPS
        t_trace = @elapsed mps /= trace_mps(mps)

        # Measure local observables
        t_obs_single = @elapsed single_configs[step+1, :] = measure_op_config(mps, "N_single")
        t_obs_pair = @elapsed pair_configs[step+1, :] = measure_op_config(mps, "N_pair")
        t_obs_zero = @elapsed zero_configs[step+1, :] = measure_op_config(mps, "N_zero")
        t_obs_total = @elapsed total_configs[step+1, :] = measure_op_config(mps, "N_tot")
        link_dims[step+1, :] = linkdims(mps)

        # Measure energy expectation values
        if measure_energy_flag
            t_energy = @elapsed energy[step+1] = measure_mpo(mps, H)
            t_kin = @elapsed kin_energy[step+1] = measure_mpo(mps, H_kin)
            t_mass = @elapsed m_energy[step+1] = (m == 0) ? 0.0 : measure_mpo(mps, H_m)
            t_el = @elapsed el_energy[step+1] = measure_mpo(mps, H_el)
        else
            t_energy = t_kin = t_mass = t_el = 0.0
        end

        # Measure gauge fields which must be reconstructed from the charges
        if compute_T2 === :Full
            t_T2 = @elapsed begin
                for (n, T2) in enumerate(T2n)
                    T2_configs[step+1, n] = measure_mpo(mps, T2; alg="naive")
                end
            end
        elseif compute_T2 === :Separate
            t_T2 = @elapsed begin
                T2_configs[step+1, :] =
                    measure_T2_configs(mps, N, Q2_mpos, Cxx_mpos, Cyy_mpos, Czz_mpos)
            end
        elseif compute_T2 === :Efficient
            t_T2 = @elapsed T2_configs[step+1, :] = measure_T2_sweep(mps)
        else
            t_T2 = 0.0
        end

        println("Step = $(step), Total = $(round(time() - start, digits=3))s, Links = $(linkdims(mps))")
        println("  odd1=$(round(t_odd1,digits=3)) taylor1=$(round(t_taylor1,digits=3)) even=$(round(t_even,digits=3)) taylor2=$(round(t_taylor2,digits=3)) odd2=$(round(t_odd2,digits=3))")
        println("  trace=$(round(t_trace,digits=3)) energy=$(round(t_energy,digits=3)) kin=$(round(t_kin,digits=3)) mass=$(round(t_mass,digits=3)) el=$(round(t_el,digits=3))")
        println("  obs=$(round(t_obs_single+t_obs_pair+t_obs_zero+t_obs_total,digits=3)) T2=$(round(t_T2,digits=3))")
        flush(stdout)
    end
    println("Time simulation done in: $(time() - time_start) seconds")

    # ------------------------------------- Save Simulation -------------------------------------
    # Write tracked observables to results h5 file
    write(results_file, "single_configs", single_configs)
    write(results_file, "pair_configs", pair_configs)
    write(results_file, "zero_configs", zero_configs)
    write(results_file, "total_configs", total_configs)
    write(results_file, "T2_configs", T2_configs)
    write(results_file, "link_dims", link_dims)
    write(results_file, "energy", energy)
    write(results_file, "kin_energy", kin_energy)
    write(results_file, "m_energy", m_energy)
    write(results_file, "el_energy", el_energy)
    total_runtime = time() - start_tot
    write(results_file, "total_runtime", total_runtime)

    println("Total simulation done in $total_runtime seconds")
    close(results_file)
    flush(stdout)
end

# Run the function performing the evolution
evolve()


