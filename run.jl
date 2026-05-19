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

        "--nsteps"
        help = "number of steps"
        arg_type = Int
        default = 100

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
    end
    return parse_args(s)
end

"""
    evolve()

Function performing the actual time evolution
"""
function evolve()
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
    number_of_time_steps = parsedArgs["nsteps"]     # Number of timesteps
    len = parsedArgs["len"]                         # Length of the string 
    tau = parsedArgs["tau"]                         # Timestep
    dissipator_sites = collect(1:N)                 # Site to be acted upon by the env. Defaulted to all
    taylor_expansion_order = parsedArgs["teo"]      # Order of Taylor expansion
    taylor_expansion_cutoff_1 = parsedArgs["tec_1"] # Cutoff 1 for Taylor expansion
    taylor_expansion_cutoff_2 = parsedArgs["tec_2"] # Cutoff 2 for Taylor expansion
    cutoff = parsedArgs["cutoff"]                   # Cutoff of the SVD. Maximum normalization to be truncated in SVD
    maxdim = parsedArgs["maxdim"]                   # Maximum bond dimension
    folder = parsedArgs["folder"]                   # Folder to store results        

    # Print the parameters we parsed
    println("Parsed args:")
    for (arg, val) in parsedArgs
        println("  $arg  =>  $val")
    end

    # Construct filename 
    path_to_results = string(folder, "os_N_", N, "_g2", g2, "_m", m, ".h5")

    # ------------------------------------- Create the initial state -------------------------------------
    # Sites which need to be flipped
    ind_start = Int(round(N / 2 - len / 2)) + 1
    flip_sites = collect(ind_start:ind_start+len-1) # If len = 0, this will not flip any sites and effectively return the vacuum 
    println("flip sites ", flip_sites)

    # Prepare the initial state
    println("Now getting the initial state")

    "Local helper function to get the Dirac vacuum with a string of lenght 'len' on top"
    function get_initial_state()
        # Prepare the dirac vacuum with a string on top
        sites_initial_state = siteinds("SU2_packed", N)   # TODO: Add conserve_qns
        psi = get_dirac_vacuum_mps(sites_initial_state; flip_sites) # This is a normal non-purified MPS
        rho = outer(psi', psi) # Get the density matrix
        rho_vec = convert(MPS, rho) # Convert the density matrix to a purified MPS
        mps = rho_vec_to_mps(rho_vec) # Split the sites so that each site has one physical index of dimension 2
        orthogonalize!(mps, 1) # Bring the center of orthogonalization to the very left        
        return mps

    end
    mps = get_initial_state()
    sites = siteinds(mps)
    println("Finished getting the initial state")

    # ------------------------------------- Create the hamiltonian -------------------------------------
    # Get the taylor, odd and even opsum groups without the l0 terms
    println("Now getting the odd, even and taylor gates without the l0 terms")

    side = "left"
    H = get_double_aH_Hamiltonian(sites, g2, m, a, side)
    H = MPO(H, sites)
    H_kin, H_el, H_m = get_double_aH_Hamiltonian_individual_terms(N, g2, m, a, side)
    H_kin, H_el, H_m = MPO(H_kin, sites), MPO(H_el, sites), MPO(H_m, sites)

    # This is done so that the odd, even gates and taylor MPO have physical legs matching the purified MPS and 
    # combining this with the swapprime done on the operators later the transpose is taken on the operators acting 
    # on the even sites which correspond to operators acting on the right of the density matrix
    for i in 2:2:length(sites) 
        sites[i] = dag(sites[i])
    end

    opsum_without_l0_terms = get_Lindblad_opsum_without_l0_terms(sites, g2, m, a, T, D, env_corr_type, dissipator_sites)
    nn_odd_without_l0_terms, nn_even_without_l0_terms, taylor = get_odd_even_taylor_groups(opsum_without_l0_terms, sites)
    println("Finished getting the odd, even and taylor gates without the l0 terms")

    # NOTE: Currently we are not using a background field but if we do we need to add it in the odd/even terms 

    # # Get the odd and even opsum groups with just the l0 terms
    # println("Now getting the odd, even and taylor gates with just the l0 terms")

    # opsum_just_l0_terms = get_Lindblad_opsum_just_l0_terms(sites, x, l_0, lambda)
    # nn_odd_just_l0_terms, nn_even_just_l0_terms, _ = get_odd_even_taylor_groups(opsum_just_l0_terms, sites)
    # println("Finished getting the odd, even and taylor gates with just the l0 terms")

    # Gather the two odd and even gates
    println("Now putting all the even and odd together")

    odd = get_odd(sites, tau / 2, nn_odd_without_l0_terms)
    even = get_even(sites, tau, nn_even_without_l0_terms)
    println("Finished putting all the even and odd together")

    # Get the MPO for the Taylor expansion
    println("Now getting the MPO for the taylor expansion")
    taylor_mpo_tmp = 0.5 * tau * MPO(taylor, sites)
    truncate!(taylor_mpo_tmp; cutoff=taylor_expansion_cutoff_1)

    for i in 2:2:length(taylor_mpo_tmp) # Transpose the MPO on the even sites which would correspond to the bottom legs of the MPO
        taylor_mpo_tmp[i] = swapprime(taylor_mpo_tmp[i], 0, 1; :tags => "Site")
    end

    taylor_mpo = get_mpo_taylor_expansion(taylor_mpo_tmp, taylor_expansion_order, taylor_expansion_cutoff_2, sites)
    println("The taylor_mpo with taylor order $(taylor_expansion_order) and cutoffs $(taylor_expansion_cutoff_1), $(taylor_expansion_cutoff_2) has bond dimensions ", linkdims(taylor_mpo))

    println("Finished getting the MPO for the taylor expansion")

    # ------------------------------------- Get the observables to be measured -------------------------------------
    # Starting the lists of the observables we want to keep track of
    println("Now getting the lists for the tracked observables")
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
    println("Finished getting the lists for the tracked observables")


    # Open the HDF5 file for the results
    results_file = h5open(path_to_results, "w")

    # ------------------------------------- Run Simulation -------------------------------------
    for step in 1:number_of_time_steps
        # One time step with ATDDMRG
        apply_odd!(odd, mps, cutoff, maxdim)
        mps = apply(taylor_mpo, mps; cutoff=cutoff, maxdim=maxdim)
        apply_even!(even, mps, cutoff, maxdim)
        mps = apply(taylor_mpo, mps; cutoff=cutoff, maxdim=maxdim)
        apply_odd!(odd, mps, cutoff, maxdim)

        # Fix trace
        mps /= trace_mps(mps)

        # Compute the tracked observables
        single_configs[step+1, :] = measure_op_config(mps, "N_single")
        pair_configs[step+1, :] = measure_op_config(mps, "N_pair")
        zero_configs[step+1, :] = measure_op_config(mps, "N_zero")
        total_configs[step+1, :] = measure_op_config(mps, "N_tot")
        linkdims_of_step = linkdims(mps)
        link_dims[step+1, :] = linkdims_of_step
        energy[step+1] = measure_mpo(mps, H)
        kin_energy[step+1] = measure_mpo(mps, H_kin)
        m_energy[step+1] = measure_mpo(mps, H_m)
        el_energy[step+1] = measure_mpo(mps, H_el)

        # Monitor bond dimension
        println("Step = $(step), Links = $(linkdims(mps))")
    end

    # ------------------------------------- Save Simulation -------------------------------------
    # Write tracked observables to results h5 file
    println("Now writing the observables to results HDF5 file")
    write(results_file, "single_configs", single_configs)
    write(results_file, "pair_configs", pair_configs)
    write(results_file, "zero_configs", zero_configs)
    write(results_file, "total_configs", total_configs)
    write(results_file, "link_dims", link_dims)
    write(results_file, "energy", energy)
    write(results_file, "kin_energy", kin_energy)
    write(results_file, "m_energy", m_energy)
    write(results_file, "el_energy", el_energy)
    close(results_file)
    println("Finished writing the observables to results h5 file")
end

# Run the function performing the evolution
evolve()



