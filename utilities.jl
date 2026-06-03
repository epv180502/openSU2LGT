using ITensors
using LinearAlgebra

# TODO: Add lamb_shift term

# -------------------------------------------- SiteType Creation ---------------------------------------------
""" 
We create a custom SiteType "SU2_packed" which is essentially a composite site made up of two spin-1/2 particles.
it would have also been possible to use a previously built one but for readeability we do this to identify each 
state as a combination of red and green. The order is such that the final site is |red> tensor |green> and all the
operators are similarly defined as the tensor product of whatever operator acts on each spin-1/2 system. 

Convention used is Sz|0> = -|0>, Sz|a> = |a> and N|0> = 0, N|a> = |a>; where a can be red or green.

I also call Id = S0 for convenience later on and to align with notes.
Some of the operators are superfluous/repeated but I write them out explicitaly for readability.
"""


const Id = Float64[1 0; 
                   0 1]

const Sz = Float64[1 0;
                   0 -1]

const Sp = Float64[0 1;
                   0 0]

const Sm = Float64[0 0;
                   1 0]

const N = Float64[1 0;
                  0 0]

# =========================
# SITE DEFINITION
# =========================

function ITensors.space(::SiteType"SU2_packed"; conserve_qns=false)
    if !conserve_qns
        return 4
    end
    return [
        QN(("Nr", 1), ("Ng", 1)) => 1,  # |rg⟩
        QN(("Nr", 1), ("Ng", 0)) => 1,  # |r0⟩
        QN(("Nr", 0), ("Ng", 1)) => 1,  # |0g⟩
        QN(("Nr", 0), ("Ng", 0)) => 1,  # |00⟩
    ]
end

function ITensors.state(::StateName"rg", ::SiteType"SU2_packed", s::Index)
    v = zeros(4); v[1] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"r0", ::SiteType"SU2_packed", s::Index)
    v = zeros(4); v[2] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"0g", ::SiteType"SU2_packed", s::Index)
    v = zeros(4); v[3] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"00", ::SiteType"SU2_packed", s::Index)
    v = zeros(4); v[4] = 1
    return ITensor(v, s)
end

# =========================
# HELPER
# =========================

make_op(mat, s) = ITensor(mat, prime(s), dag(s))

# =========================
# IDENTITY
# =========================

function ITensors.op(::OpName"IdId", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Id, Id),s)
end

# =========================
# HOPPING OPERATORS
# =========================

function ITensors.op(::OpName"SpSz", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sp, Sz),s)
end

function ITensors.op(::OpName"SmSz", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sm, Sz),s)
end

function ITensors.op(::OpName"SmS0", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sm, Id),s)
end

function ITensors.op(::OpName"S0Sp", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Id, Sp),s)
end

function ITensors.op(::OpName"SzSm", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sz, Sm),s)
end

function ITensors.op(::OpName"SzSp", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sz, Sp),s)
end


# =========================
# MASS OPERATORS
# =========================

function ITensors.op(::OpName"SpS0", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sp, Id),s)
end

function ITensors.op(::OpName"S0Sm", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Id, Sm),s)
end

function ITensors.op(::OpName"N_r", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(N, Id),s)
end

function ITensors.op(::OpName"N_g", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Id, N),s) 
end

function ITensors.op(::OpName"N_tot", ::SiteType"SU2_packed", s::Index)
    return op("N_r", s) + op("N_g", s)
end

# =========================
# ELECTRIC OPERATORS
# =========================

function ITensors.op(::OpName"SzSz", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sz, Sz),s)
end

function ITensors.op(::OpName"1-SzSz", ::SiteType"SU2_packed", s::Index)
    return op("IdId", s) - op("SzSz", s)
end

function ITensors.op(::OpName"SmSp", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sm, Sp),s)
end

function ITensors.op(::OpName"SpSm", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sp, Sm),s)
end

function ITensors.op(::OpName"SzS0", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Sz, Id),s)
end

function ITensors.op(::OpName"S0Sz", ::SiteType"SU2_packed", s::Index)
    return make_op(LinearAlgebra.kron(Id, Sz),s)
end

function ITensors.op(::OpName"DeltaZ", ::SiteType"SU2_packed", s::Index)
    return op("SzS0", s) - op("S0Sz", s)
end


# =========================
# Number operators
# =========================

function ITensors.op(::OpName"N_pair", ::SiteType"SU2_packed", s::Index)
    return make_op(diagm([1, 0, 0, 0]), s)
end

function ITensors.op(::OpName"N_single", ::SiteType"SU2_packed", s::Index)
    return make_op(diagm([0, 1, 1, 0]), s)
end

function ITensors.op(::OpName"N_zero", ::SiteType"SU2_packed", s::Index)
    return make_op(diagm([0, 0, 0, 1]), s)
end

# =========================
# Charge operators
# =========================

function ITensors.op(::OpName"Qx", ::SiteType"SU2_packed", s::Index)
    return (1/2) * (op("SpSm", s) + op("SmSp", s))
end

function ITensors.op(::OpName"Qy", ::SiteType"SU2_packed", s::Index)
    return (1im/2) * (op("SmSp", s) - op("SpSm", s))
end

function ITensors.op(::OpName"Qz", ::SiteType"SU2_packed", s::Index)
    return (1/4) * (op("SzS0", s) - op("S0Sz", s))
end

function ITensors.op(::OpName"Q2", ::SiteType"SU2_packed", s::Index)
    # The charge square is independent of the x,y,z, dof, it is equal for all three
    return (1/8) * (op("IdId", s) - op("SzSz", s))
end

# -------------------------------------------- System Functions ---------------------------------------------
function get_aH_Hamiltonian(sites, g2, m, a)

    """
    This gives the subsystem Hamiltonian acting on state vectors. 
    Currently (1+1)D gauge-fields integrated SU2 LGT hamiltonian.
    
    Function used if we want to obtain eigenstates of the hamiltonian.

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing
    """

    N = length(sites)

    opsum = OpSum()

    for n in 1:(N-1)
        
        for m in (n+1):(N-1)
            
            # Long range ZZ interaction term
            opsum += (a*g2/2)*(1/8)*(N-m),"DeltaZ",n,"DeltaZ",m

            # Long range hopping interaction term
            opsum += (a*g2/2)*(N-m),"SmSp",n,"SpSm",m
            opsum += (a*g2/2)*(N-m),"SpSm",n,"SmSp",m

        end

        # Kinetic term
        opsum += -(1/(2*a)),"SpSz",n,"SmS0",n+1
        opsum += -(1/(2*a)),"S0Sp",n,"SzSm",n+1
        opsum += -(1/(2*a)),"SmSz",n,"SpS0",n+1
        opsum += -(1/(2*a)),"S0Sm",n,"SzSp",n+1

        # Inverse Z term
        opsum += (a*g2/2)*(3/8)*(N-n),"1-SzSz",n
        
        # Mass term
        opsum += (m*(-1)^(n-1)),"N_tot",n

    end

    # The for loop on top only goes to N-1 so we add the last term manually
    opsum += (m*(-1)^(N-1)),"N_tot",N

    return MPO(opsum, sites)

end

function environment_correlator(type, n, m, D)
    """ Give the environment correlator strength between given sites n and m where D is the self-correlation D_0"""
    if type == "constant"
        return D
    elseif type == "delta"
        if n == m
            return D
        else
            return 0.0
        end
    end
end

function get_dirac_vacuum_mps(sites; flip_sites = [])
    """
    Function to create the dirac vacuum of a (1+1)D gauge-fields integrated SU2 LGT.
    If flip_sites is defined, put a baryon-antibaryon pair on top of that state by flipping the sites outlined in flip_sites
    """ 

    N = length(sites)
    state = []
    for n in 1:N

        # I put not (!) because I want my sites to begin in particles and since julia vector numbering begins in 1 
        # it would normally begin with antiparticles. Similarly, I added a -1 in the exponent of -1 when calculating 
        # the staggered terms. The vacuum is [00, rg, ...]
        if !isodd(n)
            if n in flip_sites
                push!(state, "00")
            else
                push!(state, "rg")
            end
        else
            if n in flip_sites
                push!(state, "rg")
            else
                push!(state, "00") 
            end
        end
    end
    mps = MPS(sites, state)
   
    return mps

end

function get_string_on_dirac_vacuum_mps(sites, len)
    # TODO: Benchmark the three possible strings with N = 6 ED system

    """
    Function to creates a string of length l in the middle of the dirac vacuum of a (1+1)D gauge-fields integrated SU2 LGT.
    
    len = Length of the string. Must be an odd number smaller than the number of sites
    """ 

    N = length(sites)
    psi_vacuum = get_dirac_vacuum_mps(sites)

    # Figure out between which indices the string will be placed
    string_start = div(N,2) - div(len,2)
    string_end = string_start + len
    gates = ITensor[]

    # The hopping operators must be anti-site -> site
    if isodd(string_start) # This is a site since julia starts counting on 1
        # Hops leftward
        for n in string_end:-2:string_start
            push!(gates, op("S0Sp", sites[n-1]) * op("SzSm", sites[n]) +
                op("SpSz", sites[n-1]) * op("SmS0", sites[n]))
        end

        for n in (string_end-1):-2:(string_start+1)
            push!(gates, op("S0Sp", sites[n-1]) * op("SzSm", sites[n]) +
                op("SpSz", sites[n-1]) * op("SmS0", sites[n]))
        end
    else
        # Hops rightward
        for n in string_start:2:string_end
            push!(gates, op("S0Sm", sites[n]) * op("SzSp", sites[n+1]) +
                op("SmSz", sites[n]) * op("SpS0", sites[n+1]))
        end

        for n in (string_start+1):2:(string_end-1)
            push!(gates, op("S0Sm", sites[n]) * op("SzSp", sites[n+1]) +
                op("SmSz", sites[n]) * op("SpS0", sites[n+1]))
        end
    end

    psi_string = apply(gates, psi_vacuum)

    # Out of precaution though shouldn't be necessary normalize
    normalize!(psi_string)
    
    return psi_string
end

function get_double_aH_Hamiltonian(sites, g2, m, a, side)

    """
    This gives H Hamiltonian acting on one side of a vectorized density matrices on a given side.
    Side specifies "left" or "right" to imply H tensor product I or vice versa so that is H*rho or rho*H.
    Currently (1+1)D gauge-fields integrated SU2 LGT hamiltonian

    Function used to calculate energy expectation values

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing

    """

    N = div(length(sites), 2)

    opsum = OpSum()

    # In this loop, n is the physical index, n_idx is the index of the tensor in the MPS
    for n in 1:(N-1)

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in (n+1):(N-1)

            if side == "left"
                m_idx = 2*m-1
            else
                m_idx = 2*m
            end
            
            # Long range ZZ interaction term
            opsum += (a*g2/2)*(1/8)*(N-m),"DeltaZ",n_idx,"DeltaZ",m_idx

            # Long range hopping interaction term
            opsum += (a*g2/2)*(N-m),"SmSp",n_idx,"SpSm",m_idx
            opsum += (a*g2/2)*(N-m),"SpSm",n_idx,"SmSp",m_idx

        end

        # Kinetic term
        opsum += -(1/(2*a)),"SpSz",n_idx,"SmS0",n_idx+2
        opsum += -(1/(2*a)),"S0Sp",n_idx,"SzSm",n_idx+2
        opsum += -(1/(2*a)),"SmSz",n_idx,"SpS0",n_idx+2
        opsum += -(1/(2*a)),"S0Sm",n_idx,"SzSp",n_idx+2

        # Inverse Z term
        opsum += (a*g2/2)*(3/8)*(N-n),"1-SzSz",n_idx
        
        # Mass term
        opsum += (m*(-1)^(n-1)),"N_tot",n_idx

    end

    # The for loop on top only goes to N-1 so we add the last term manually
    if side == "left"
        opsum += (m*(-1)^(N-1)),"N_tot",2*N-1
    else
        opsum += (m*(-1)^(N-1)),"N_tot",2*N
    end

    return opsum

end

function get_aLm_aLndag(n, m, a, T, sites)

    """
    Gives the operator Lm tensor product Ln^dagger so Lm acts on the left side system and Ln^dagger on the right.
    Currently (1+1)D gauge-fields integrated SU2 LGT dissipators.

    For derivations of the long the terms look at notes. 

    n = Site 2
    m = Site 1
    a = Lattice spacing
    T = Temperature

    """

    N = div(length(sites), 2)   # Number of lattice sites
    J = -1/(2*a)                # Prefactor of kinetic term
    prefactor = J/(4*T)         # Prefactor of Sn/Sm terms 

    # n_phys, m_phys refer to the lattice sites while n, m refer to the MPO indices
    n_phys, m_phys = div(n, 2), div(m + 1, 2)
    phys_sign = n_phys + m_phys

    # See derivation for explanation
    Sn_minus = [(1, "SmSz", n-2, "SpS0", n), (1, "S0Sm", n-2, "SzSp", n),  
               (-1, "SpSz", n-2, "SmS0", n), (-1, "S0Sp", n-2, "SzSm", n)]

    Sn_plus = [(-1, "SmSz", n, "SpS0", n+2), (-1, "S0Sm", n, "SzSp", n+2), 
                (1, "SpSz", n, "SmS0", n+2), (1, "S0Sp", n, "SzSm", n+2)]

    Sm_minus = [(1, "SpSz", m-2, "SmS0", m), (1, "S0Sp", m-2, "SzSm", m),
               (-1, "SmSz", m-2, "SpS0", m), (-1, "S0Sm", m-2, "SzSp", m)]
    
    Sm_plus = [(-1, "SpSz", m, "SmS0", m+2), (-1, "S0Sp", m, "SzSm", m+2), 
                (1, "SmSz", m, "SpS0", m+2), (1, "S0Sm", m, "SzSp", m+2)]

    opsum = OpSum()

    # Term (a)
    opsum += (-1)^(phys_sign),"N_tot",m,"N_tot",n

    # Term (b)
    if (n_phys != 1)
        for (coef, op1, idx1, op2, idx2) in Sn_minus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),"N_tot",m,op1,idx1,op2,idx2
        end
    end

    if (n_phys != N)
        for (coef, op1, idx1, op2, idx2) in Sn_plus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),"N_tot",m,op1,idx1,op2,idx2
        end
    end

    # Term (c)
    if (m_phys != 1)
        for (coef, op1, idx1, op2, idx2) in Sm_minus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),op1,idx1,op2,idx2,"N_tot",n
        end
    end

    if (m_phys != N)
        for (coef, op1, idx1, op2, idx2) in Sm_plus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),op1,idx1,op2,idx2,"N_tot",n
        end
    end

    # Term (d)
    if (n_phys != 1) && (m_phys != 1)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sm_minus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sn_minus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (m_phys != 1) && (n_phys != N)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sm_minus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sn_plus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (m_phys != N) && (n_phys != 1)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sm_plus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sn_minus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (n_phys != N) && (m_phys != N)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sm_plus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sn_plus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    return opsum
    
end

function get_aLndag_aLm(n, m, a, T, sites, side)

    """
    Gives the operator Ln^dagger tensor product Lm can be either acting all on the left or all on the right
    Currently (1+1)D gauge-fields integrated SU2 LGT dissipators.

    For derivations of the long terms look at notes. 

    n = Site 2
    m = Site 1
    a = Lattice spacing
    T = Temperature
    side = Whether it is acting on the left or right of the density matrix in lindblad equation

    """

    N = div(length(sites), 2)   # Number of lattice sites
    J = -1/(2*a)                # Prefactor of kinetic term
    prefactor = J/(4*T)         # Prefactor of Sn/Sm terms 

    # n_phys, m_phys refer to the lattice sites while n, m refer to the MPO indices
    if side == "left"
        n_phys, m_phys = div(n + 1, 2), div(m + 1, 2)
    else
        n_phys, m_phys = div(n, 2), div(m, 2)
    end
    phys_sign = n_phys + m_phys

    # See derivation for explanation
    Sn_minus = [(1, "SmSz", n-2, "SpS0", n), (1, "S0Sm", n-2, "SzSp", n),  
               (-1, "SpSz", n-2, "SmS0", n), (-1, "S0Sp", n-2, "SzSm", n)]

    Sn_plus = [(-1, "SmSz", n, "SpS0", n+2), (-1, "S0Sm", n, "SzSp", n+2), 
                (1, "SpSz", n, "SmS0", n+2), (1, "S0Sp", n, "SzSm", n+2)]

    Sm_minus = [(1, "SpSz", m-2, "SmS0", m), (1, "S0Sp", m-2, "SzSm", m),
               (-1, "SmSz", m-2, "SpS0", m), (-1, "S0Sm", m-2, "SzSp", m)]
    
    Sm_plus = [(-1, "SpSz", m, "SmS0", m+2), (-1, "S0Sp", m, "SzSm", m+2), 
                (1, "SmSz", m, "SpS0", m+2), (1, "S0Sm", m, "SzSp", m+2)]

    opsum = OpSum()

    # Term (a)
    opsum += (-1)^(phys_sign),"N_tot",n,"N_tot",m

    # Term (b)
    if (m_phys != 1)
        for (coef, op1, idx1, op2, idx2) in Sm_minus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),"N_tot",n,op1,idx1,op2,idx2
        end
    end

    if (m_phys != N)
        for (coef, op1, idx1, op2, idx2) in Sm_plus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),"N_tot",n,op1,idx1,op2,idx2
        end
    end

    # Term (c)
    if (n_phys != 1)
        for (coef, op1, idx1, op2, idx2) in Sn_minus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),op1,idx1,op2,idx2,"N_tot",m
        end
    end

    if (n_phys != N)
        for (coef, op1, idx1, op2, idx2) in Sn_plus
            opsum -= (coef*(-1)^(phys_sign)*prefactor),op1,idx1,op2,idx2,"N_tot",m
        end
    end

    # Term (d)
    if (n_phys != 1) && (m_phys != 1)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sn_minus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sm_minus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (n_phys != 1) && (m_phys != N)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sn_minus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sm_plus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (n_phys != N) && (m_phys != 1)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sn_plus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sm_minus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    if (n_phys != N) && (m_phys != N)
        for (coef_n, op1_n, idx1_n, op2_n, idx2_n) in Sn_plus
            for (coef_m, op1_m, idx1_m, op2_m, idx2_m) in Sm_plus
                opsum += (coef_n*coef_m*(-1)^(phys_sign)*(prefactor^2)), op1_n, idx1_n, op2_n, idx2_n, op1_m, idx1_m, op2_m, idx2_m
            end
        end
    end

    return opsum

end

function get_Lindblad_opsum_without_l0_terms(sites, g2, m, a, T, D, env_corr_type, dissipator_sites)

    """
    This gives L Lindbladian acting on a vectorized density matrix without background "l_0" terms
    Currently (1+1)D gauge-fields integrated SU2 LGT system

    Function used to get all terms in the lindbladian

    g2 = Gauge coupling square g**2
    m = Mass
    a = Lattice spacing
    T = Temperature
    D = Self-correlation 
    env_corr_type = What environment correlator do we have (Currently delta or constant)
    dissipator_sites = What sites the dissipators will act on
    """

    # Unitary part of the hamiltonian
    res = -1im*get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, "left")
    res += 1im*get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, "right")
    
    # Dissipative part 
    if D != 0
        for n in dissipator_sites
            for m in dissipator_sites
                if env_corr_type == "delta" && n != m
                    continue
                end
                res += environment_correlator(env_corr_type, n, m, D) * ( get_aLm_aLndag(2*n, 2*m-1, a, T, sites) - 0.5 * get_aLndag_aLm(2*n-1, 2*m-1, a, T, sites, "left") - 0.5 * get_aLndag_aLm(2*n, 2*m, a, T, sites, "right") )
            end
        end
    end

    return res

end

function get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, side)

    """
    This gives H Hamiltonian acting on one side of a vectorized density matrices on a given side. Side specifies "left" or 
    "right" to imply H tensor product I or vice versa so that is H*rho or rho*H
    Currently (1+1)D gauge-fields integrated SU2 LGT hamiltonian

    Function used to create the full lindbladian

    At present moment equivalent to 'get_double_aH_Hamiltonian' but if we add l_0 term in the future
    it is neceesary to have two separate functions

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing

    """

    N = div(length(sites), 2)

    opsum = OpSum()

    # In this loop, n is the physical index, n_idx is the index of the tensor in the MPS
    for n in 1:(N-1)

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in (n+1):(N-1)

            if side == "left"
                m_idx = 2*m-1
            else
                m_idx = 2*m
            end
            
            # Long range ZZ interaction term
            opsum += (a*g2/2)*(1/8)*(N-m),"DeltaZ",n_idx,"DeltaZ",m_idx

            # Long range hopping interaction term
            opsum += (a*g2/2)*(N-m),"SmSp",n_idx,"SpSm",m_idx
            opsum += (a*g2/2)*(N-m),"SpSm",n_idx,"SmSp",m_idx

        end

        # Kinetic term
        opsum += -(1/(2*a)),"SpSz",n_idx,"SmS0",n_idx+2
        opsum += -(1/(2*a)),"S0Sp",n_idx,"SzSm",n_idx+2
        opsum += -(1/(2*a)),"SmSz",n_idx,"SpS0",n_idx+2
        opsum += -(1/(2*a)),"S0Sm",n_idx,"SzSp",n_idx+2

        # Inverse Z term
        opsum += (a*g2/2)*(3/8)*(N-n),"1-SzSz",n_idx
        
        # Mass term
        opsum += (m*(-1)^(n-1)),"N_tot",n_idx

    end

    # The for loop on top only goes to N-1 so we add the last term manually
    if side == "left"
        opsum += (m*(-1)^(N-1)),"N_tot",2*N-1
    else
        opsum += (m*(-1)^(N-1)),"N_tot",2*N
    end

    return opsum
end
 
function get_double_aH_Hamiltonian_individual_terms(N, g2, m, a, side)

    """
    This gives H Hamiltonian acting on one side of a vectorized density matrices on a given side. Side specifies "left" or 
    "right" to imply H tensor product I or vice versa so that is H*rho or rho*H. This function gives each contribution 
    to the hamiltonian separately, that is kinetic_terms, electric_terms, and mass_terms
    Currently (1+1)D gauge-fields integrated SU2 LGT hamiltonian

    Function used to calculate the distribution of energy in the system

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing

    """

    opsum_kinetic_term = OpSum()
    opsum_mass_term = OpSum()
    opsum_electric_field_term = OpSum()

    for n in 1:(N-1)

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in (n+1):(N-1)

            if side == "left"
                m_idx = 2*m-1
            else
                m_idx = 2*m
            end

            # Long range ZZ interaction term
            opsum_electric_field_term += (a*g2/2)*(1/8)*(N-m),"DeltaZ",n_idx,"DeltaZ",m_idx

            # Long range hopping interaction term
            opsum_electric_field_term += (a*g2/2)*(N-m),"SmSp",n_idx,"SpSm",m_idx
            opsum_electric_field_term += (a*g2/2)*(N-m),"SpSm",n_idx,"SmSp",m_idx

        end

        # Kinetic term
        opsum_kinetic_term += -(1/(2*a)),"SpSz",n_idx,"SmS0",n_idx+2
        opsum_kinetic_term += -(1/(2*a)),"S0Sp",n_idx,"SzSm",n_idx+2
        opsum_kinetic_term += -(1/(2*a)),"SmSz",n_idx,"SpS0",n_idx+2
        opsum_kinetic_term += -(1/(2*a)),"S0Sm",n_idx,"SzSp",n_idx+2

        # Inverse Z term
        opsum_electric_field_term += (a*g2/2)*(3/8)*(N-n),"1-SzSz",n_idx
        
        # Mass term
        opsum_mass_term += (m*(-1)^(n-1)),"N_tot",n_idx

    end

    # The for loop on top only goes to N-1 so we add the last term manually
    if side == "left"
        opsum_mass_term += (m*(-1)^(N-1)),"N_tot",2*N-1
    else
        opsum_mass_term += (m*(-1)^(N-1)),"N_tot",2*N
    end


    return opsum_kinetic_term, opsum_electric_field_term, opsum_mass_term

end

# -------------------------------------------- Tensor Network Functions ---------------------------------------------
function rho_vec_to_mps(rho_vec)

    """ 
    Transform a purified MPS representing a vectorized density matrix into a proper MPS by splitting each 
    site (which currently has two legs) into two separate sites with each a dimension of 4 
    """

    N = length(rho_vec)

    mps = MPS(2*N)  # Our final purified MPS will have twice as many indices as lattice sites

    for i in 1:N

        # this object has 2 physical legs and we want to svd between them to seperate them into two mps tensors
        M = rho_vec[i]

        # picks the left link and the dashed site indices for the U of SVD
        left_inds_M = (inds(M; :tags => "Link,l=$(i-1)"), inds(M; :plev => 1))

        U, S, V = ITensors.svd(M, left_inds_M; leftdir = ITensors.In, rightdir = ITensors.In)
        V = S*V

        # Fix U indices
        if i != 1
            replacetags!(U, "Link,l=$(i-1)", "Link,l=$(2*i-2)") # change tag of left link to mps convention
        end
        replacetags!(U, "Link,u", "Link,l=$(2*i-1)") # change tag of svd link to mps convention
        replacetags!(U, "SU2_packed,Site,n=$(i)", "SU2_packed,Site,n=$(2*i-1)") # change the label of the second physical index from i to 2*i-1
        mps[2*i-1] = U

        # Fix V indices
        if i != N # if i == N there is no right link
            replacetags!(V, "Link,l=$(i)", "Link,l=$(2*i)") # same as the line above but for the right link
        end
        replacetags!(V, "Link,u", "Link,l=$(2*i-1)") # change tag of svd link to mps convention (here it is Link,u because of V = S*V above)
        replacetags!(V, "SU2_packed,Site,n=$(i)", "SU2_packed,Site,n=$(2*i)") # change the label of the second physical index from i to 2*i
        # V = reverse_dir(V, inds(V; :tags => "Site")[1]) # reverse the physical leg direction from in to out - this is mandatory for constructing MPO with autoMPO
        mps[2*i] = V

    end

    return noprime(mps)

end

function apply_odd!(odd, mps, cutoff, maxdim)

    """ Apply the odd gates to the MPS. Because of the structure of n, n', n+1, n+1'; these are four site operators"""

    l = length(mps)

    for (n_idx, n) in enumerate(1:4:(l-3)) # n is the left most site the gate acts on, the gates are always 4 site in span

        gate = odd[n_idx]

        t = noprime(gate*prod(mps[n:n+3]))

        # Separate the t tensor into individual MPS site tensors
        for idx in n:n+2
            t_indices = inds(t)
            U_indices = filter(i -> hastags(i, "Site,n=$(idx)") || hastags(i, "Link,l=$(idx-1)"), t_indices)
            U, S, V = ITensors.svd(t, U_indices; cutoff = cutoff, maxdim = maxdim, lefttags = "Link,l=$(idx)", righttags = "Link,l=$(idx)")
            t = S*V
            mps[idx] = U
        end
        mps[n+3] = t

        if n+4 <= l
            # Extra SVD as required by ATD DMRG
            t = mps[n+3]*mps[n+4]
            mps[n+3], S, V = ITensors.svd(t, uniqueinds(t, mps[n+4]); cutoff = cutoff, maxdim = maxdim, lefttags = "Link,l=$(n+3)", righttags = "Link,l=$(n+3)")
            mps[n+4] = S*V
        end

    end

end

function apply_even!(even, mps, cutoff, maxdim)

    """ Apply the even gates to the MPS. Because of the structure of n, n', n+1, n+1'; these are four site operators"""

    l = length(mps)
    
    t = mps[1]*mps[2]
    mps[1], mps[2] = ITensors.qr(t, uniqueinds(t, mps[2]); tags = "Link,l=$(1)")

    t = mps[2]*mps[3]
    mps[2], mps[3] = ITensors.qr(t, uniqueinds(t, mps[3]); tags = "Link,l=$(2)")

    for (n_idx, n) in enumerate(3:4:(l-3)) # n is the left most site the gate acts on, the gates are always 4 site in span

        gate = even[n_idx]

        t = noprime(gate*prod(mps[n:n+3]))

        # Separate the t tensor into individual MPS site tensors
        for idx in n:n+2
            t_indices = inds(t)
            U_indices = filter(i -> hastags(i, "Site,n=$(idx)") || hastags(i, "Link,l=$(idx-1)"), t_indices)
            U, S, V = ITensors.svd(t, U_indices; cutoff = cutoff, maxdim = maxdim, lefttags = "Link,l=$(idx)", righttags = "Link,l=$(idx)")
            t = S*V
            mps[idx] = U
        end
        mps[n+3] = t

        if n_idx != length(3:4:(l-3))
            # Extra SVD as required by ATD DMRG
            t = mps[n+3]*mps[n+4]
            mps[n+3], S, V = ITensors.svd(t, uniqueinds(t, mps[n+4]); cutoff = cutoff, maxdim = maxdim, lefttags = "Link,l=$(n+3)", righttags = "Link,l=$(n+3)")
            mps[n+4] = S*V
        end

    end

end

function trace_mps(mps)

    """ 
    Take the trace of the MPS. Since this is a density matrix, not a state vector, this is the correct way to normalize after truncation.
    That means our state is not normalized as |<psi|psi>|^2 = 1, but rather trace(rho), so tracing between adjacent legs that signify the 
    bra and ket of site n
    """

    sites = siteinds(mps)
    l = length(mps)

    i = 1
    left, right = 2*i-1, 2*i
    res = mps[left]*mps[right]*dag(delta(sites[left], sites[right]))

    for i in 2:div(l,2)
        left, right = 2*i-1, 2*i
        res *= mps[left]*mps[right]*dag(delta(sites[left], sites[right]))
    end

    return res[1]

end

function measure_op(mps, opname, site)

    """ Measure any local operator on a given site of the mps (DEPRECATED)"""

    sites = siteinds(mps)
    l = length(mps)

    opTens = op(opname, sites[site])
    mps = apply(opTens, mps)

    i = 1
    left, right = 2*i-1, 2*i
    res = mps[left]*mps[right]*dag(delta(sites[left], sites[right]))

    for i in 2:div(l,2)
        left, right = 2*i-1, 2*i
        res *= mps[left]*mps[right]*dag(delta(sites[left], sites[right]))
    end

    return res[1]

end

function measure_op_config(mps, opname; left = true)

    """ Measure any local operator on all the sites of the mps (DEPRECATED)"""

    n = length(mps)
    op_config = []
    if left
        for site in 1:2:n
            push!(op_config, measure_op(mps, opname, site))
        end
    else
        for site in 2:2:n
            push!(op_config, measure_op(mps, opname, site))
        end
    end

    return op_config

end

function get_T2n(n; side="left")
    """ 
    Get T squared lattice site n 
    This requires its own function as the integrated version does not 'keep track' of the gauge links, 
    instead we must reconstruct it from the charges of all previous sites. 

    Function no longer used as it is much more efficient to calculate the correlators independently and 
    manually reconstruct the expectation value of T2n via those numbers
    """

    # Construct the electric field MPO
    opsum = OpSum()

    for p in 1:n

        if side == "left"
            p_idx = 2*p-1
        else
            p_idx = 2*p 
        end

        # Diagonal part
        opsum += 3, "Q2", p_idx

        for q in (p+1):n

            if side == "left"
                q_idx = 2*q-1
            else
                q_idx = 2*q
            end
            
            # Off-diagonal part
            opsum += 2,"Qx",p_idx,"Qx",q_idx
            opsum += 2,"Qy",p_idx,"Qy",q_idx
            opsum += 2,"Qz",p_idx,"Qz",q_idx

        end

    end

    return opsum

end

function measure_T2_configs(mps, N, 
                             Q2_mpos, Cxx_mpos, Cyy_mpos, Czz_mpos)
    T2 = zeros(ComplexF64, N)
    
    for n in 1:N
        increment = 3 * measure_mpo(mps, Q2_mpos[n]; alg="naive")
        for q in 1:n-1
            increment += 2 * (measure_mpo(mps, Cxx_mpos[q,n]; alg="naive") +
                               measure_mpo(mps, Cyy_mpos[q,n]; alg="naive") +
                               measure_mpo(mps, Czz_mpos[q,n]; alg="naive"))
        end
        T2[n] = (n > 1 ? T2[n-1] : 0.0) + increment
    end
    
    return T2
end

function get_mpo_taylor_expansion(mpo, order, cutoff, sites)

    """
    Returns the taylor expansion to the given input order of the mpo as 1 + mpo + mpo * mpo / factorial(2) + mpo * mpo * mpo / factorial(3) + ... etc
    """

    l = [mpo]
    for i in 2:order
        push!(l, apply(l[end], mpo/i; cutoff = cutoff))
    end

    tmp1 = sum(l)
    tmp2 = MPO(sites, "IdId")

    for i in 2:2:length(tmp2)
        tmp2[i] = swapprime(tmp2[i], 0, 1; :tags => "Site")
    end

    return add(tmp1, tmp2; cutoff = 0)

end

function get_odd(sites, a, gate_lists)

    """
    sites are the site indices of the MPS to which the odd gates will act on

    a is usually the time step or the time step divided by 2

    gate_lists is a list of lists specifically nn_odd from the function get_odd_even_taylor_groups in which nn_odd is explained
    """

    gates = []

    l = length(sites)

    for (n_idx, n) in enumerate(1:4:(l-3))

        group_opsum = gate_lists[n_idx] # an opsum with terms lying in the range n:n+3, the actual operators in this opsum always have site numbers from 1 to 4
        group_opsum_indices = sites[n:n+3] # these are the actual indices that will be given to the ITensor below to apply the gate to the appropriate sites

        gate = ITensor(group_opsum, group_opsum_indices)
        Gj = exp(a * gate)

        # Take the transpose on the even sites which correspond to legs which would have been bottom legs on an MPO
        swapprime!(Gj, 0, 1; :tags => "n=$(n+1)")
        swapprime!(Gj, 0, 1; :tags => "n=$(n+3)")

        push!(gates, Gj)

    end

    return gates

end

function get_even(sites, a, gate_lists)

    """
    sites are the site indices of the MPS to which the even gates will act on

    a is usually the time step or the time step divided by 2

    gate_lists is a list of lists specifically nn_even from the function get_odd_even_taylor_groups in which nn_even is explained
    """

    gates = []

    l = length(sites)

    for (n_idx, n) in enumerate(3:4:(l-3))

        group_opsum = gate_lists[n_idx] # an opsum with terms lying in the range n:n+3, the actual operators in this opsum always have site numbers from 1 to 4
        group_opsum_indices = sites[n:n+3] # these are the actual indices that will be given to the ITensor below to apply the gate to the appropriate sites

        gate = ITensor(group_opsum, group_opsum_indices)
        Gj = exp(a * gate)

        # Take the transpose on the even sites which correspond to legs which would have been bottom legs on an MPO
        swapprime!(Gj, 0, 1; :tags => "n=$(n+1)")
        swapprime!(Gj, 0, 1; :tags => "n=$(n+3)")

        push!(gates, Gj)

    end

    return gates
end

function get_odd_even_taylor_groups(opsum, sites)

    """
    This function takes as input an OpSum and the sites of the MPS and returns nn_odd (nearest neighbour odd), nn_even and taylor.
    
    The gates considered odd are 4 site gates having a left most index 1:4:l-3 where l is the length of sites. 
    
    The gates considered even are 4 site gates having a left most index 3:4:l-3. 
    
    Everything else is considered to be in the taylor group which will be taylor expanded in the
    trotterization scheme: odd/2 taylor/2 even taylor/2 odd/2 (second order trotterization).

    The nn_odd is a list of lists where each list holds all the gates lying within the group of sites "left most index of group up to left most index of group + 3"
    where the left most index of each group is defined by 1:4:l-3. Similarly for nn_even.
    """

    function pad_op(element, group_left)

        """
        This function takes an element from an OpSum e.g. a Scaled{ComplexF64, Prod{Op}} and a group_left index which
        stands for the beginning of a group of 4 sites.

        The purpose of this function is most easily explained with an example:

        element = 0.5,"Z",2,"S-",4
        group_left = 2

        Here group_left is 2 so our sites span 2, 3, 4, 5 and the operators in the element are going to be shifted
        to have indices 1, 2, 3, 4 and then accordingly we will place the operators on the sites putting identities where we had no
        operators resulting in:

        0.5,"I",1,"I",2,"I",3,"I",4,"Z",1,"S-",3 where the first part is padding with identities and then the operators are appropriately placed within 1, 2, 3, 4

        The reason we need this function is to do build the exponential of gates for time evolution where the ITensor function will get the opsum of many of this
        elements but the indices will not be 1, 2, 3, 4 but rather sites[group_left:group_left+3] where sites are the actual site indices of the MPS
        """

        element_coefficient = element.args[1]      
        element_list = [[ITensors.which_op(element[i]), ITensors.site(element[i]) - group_left + 1] for i in 1:length(element)] # result e.g. [["Z", 1], ["S-", 3]] using the example above
        el = [["I", i] for i in 1:4] # put identities on all sites to cover the ones which don't have a non-trivial operator acting on them in the element (padding)
        for e in element_list # put in the non-trivial operators as gathered from the element input
            push!(el, [e[1], e[2]]) 
        end
        # Put both the identities and the non-trivial operators from the element input into one opsum
        complete_opsum = OpSum()
        complete_opsum += element_coefficient,el[1][1],el[1][2]
        for i in 2:length(el)
            complete_opsum *= el[i][1],el[i][2]
        end

        return complete_opsum

    end

    l = length(sites)
    taylor = OpSum() # defined as anything that does not fall into the following two categories below
    nn_odd = [OpSum() for _ in 1:4:l-3] # defined as anything that lies between 1 and 4 inclusive, 5 and 8, 9 and 12 etc
    nn_even = [OpSum() for _ in 3:4:l-3] # defined as anything that lies between 3 and 6 inclusive, 7 and 10
    
    odd_index_groups_list = [[i, i+1, i+2, i+3] for i in 1:4:l-3]
    odd_index_groups = reshape(reduce(vcat, odd_index_groups_list'), (length(odd_index_groups_list), 4)) # make the list of lists into a matrix for convenience
    even_index_groups_list = [odd_index_groups_list[i] .+ 2 for i in 1:length(odd_index_groups_list)-1]
    even_index_groups = reshape(reduce(vcat, even_index_groups_list'), (length(even_index_groups_list), 4)) # make the list of lists into a matrix for convenience
    
    for element in opsum 

        idxs = ITensors.sites(element) # where the element acts non-trivially
        left, right = minimum(idxs), maximum(idxs) # left and right most sites the gate is acting on
        
        gate_span = right - left + 1
        if gate_span <= 4 # if it can fit within an even or odd gate

            odd_group_of_left_index = findfirst(x -> x == left, odd_index_groups) # the index of the group to which the element can belong to (this can also return nothing when the element cannot belong to an odd group)
            if odd_group_of_left_index !== nothing
                odd_exists = true
                odd_group_of_left_index = odd_group_of_left_index[1] # just take the index of the group and discard the second element which is the position within the group
            else
                odd_exists = false
            end
            if odd_exists
                # the left and right most sites of the odd group the element can belong to
                min_of_odd_group, max_of_odd_group = minimum(odd_index_groups[odd_group_of_left_index, :]), maximum(odd_index_groups[odd_group_of_left_index, :])
                if left >= min_of_odd_group && right <= max_of_odd_group
                    odd_possible = true
                else
                    odd_possible = false
                end
            else
                odd_possible = false
            end
    
            # same as above but for even
            even_group_of_left_index = findfirst(x -> x == left, even_index_groups)
            if even_group_of_left_index !== nothing
                even_exists = true
                even_group_of_left_index = even_group_of_left_index[1]
            else
                even_exists = false
            end
            if even_exists
                min_of_even_group, max_of_even_group = minimum(even_index_groups[even_group_of_left_index, :]), maximum(even_index_groups[even_group_of_left_index, :])
                if left >= min_of_even_group && right <= max_of_even_group
                    even_possible = true
                else
                    even_possible = false
                end
            else
                even_possible = false
            end
    
            if odd_possible && even_possible
    
                if length(nn_odd[odd_group_of_left_index]) < length(nn_even[even_group_of_left_index]) # if the gate can go in both an odd and an even choose the one with less number of gates already
    
                    nn_odd[odd_group_of_left_index] += pad_op(element, min_of_odd_group)
                    
                else
    
                    nn_even[even_group_of_left_index] += pad_op(element, min_of_even_group)

                end
            
            elseif odd_possible
    
                nn_odd[odd_group_of_left_index] += pad_op(element, min_of_odd_group)

    
            elseif even_possible
    
                nn_even[even_group_of_left_index] += pad_op(element, min_of_even_group)
    
            else
    
                taylor += element
    
            end
            
        else
    
            taylor += element
    
        end
    end

    return nn_odd, nn_even, taylor

end

function measure_mpo(mps, mpo; alg = "none")

    """ 
    Measure the expectation value of an MPO on an mps. 
    Inneficient but very easy and straight forward to use
    """

    sites = siteinds(mps)
    l = length(mps)

    if alg == "none"
        mps = apply(mpo, mps)
    else
        mps = apply(mpo, mps; alg = alg)
    end

    i = 1
    left, right = 2*i-1, 2*i
    res = mps[left]*mps[right]*dag(delta(sites[left], sites[right]))

    for i in 2:div(l,2)
        left, right = 2*i-1, 2*i
        res *= mps[left]*mps[right]*dag(delta(sites[left], sites[right]))
    end

    return res[1]

end

"""
The following functions employ an algorithm inspired by how ITensors does correlation_matrix, but takes into consideration 
the use of purified MPS which differ in two very big ways. First of all the norm of a purified MPS is not 1 but rather the 
trace is 1. More importantly, an expectation value is not <MPS|O|MPS>, but rather tr(rho*O). In this code we are doubling 
the amount of sites our MPS has by unraveling the MPO. This means the "left" side of rho lives on odd legs and the "right" 
side lives on even legs (odd/even wrt julia index notation). So the trace of our MPS is basically tracing adjacent sites 
with eachother. Taking an expectation value is tracing adjacent sites after applying an MPO on only one side (even/odd parity)
of the purified MPS. With this in mind, taking an expectation value can simply be applying the MPO can tracing. This is 
however not optimal, specially when wanting to calculate site resolved expectation values. 

Considering the present model is highly non-local because we integrated out gauge fields, the following codes calculate the 
expectation values by doing a single rightward sweep and saving up the traced out left environments. 

Namely, the spirit of the algorithm is the following:
1. Precompute right environments. This means compute the trace of the purified MPS from site n+1 to N_phys. 
2. Compute the initial environments. Namely, L_id (same as R_id but for 1,2,...,n) and any non-local environments
you need. For example if you need <Q_n Q_m> correlators build left L_Qa environmen where it represents the traced out
version of all sites to the left of n, having applied one Qa operator at site p. L_Qa is the superpositions of all p < n. 
This means L_Qa = Σ_{p=1}^{n-1} Qa_p. 

    for n = 1:N_phys
            3. For local operators: add Qa operator at site n onto the left identity environment L_id
            4. For non-local operators: add Qa operator at site n onto the left L_Qa environment. 
            5. Contract the resulting (L_id or L_Qa) * Qa with the precomputed right environments R_id
            6. Update the next L_Qa by adding all the left environments obtained by adding Qa at site n onto L_id
            7. Update the L_id to go up to site n+1 
"""

function build_R_id(mps)

    """
    Computes all right identity environments for purified MPS. Namely:
    R_id[n] is the identity contraction of physical sites n, n+1, …, N_phys
    R_id[N_phys+1] = scalar 1 (empty right boundary)

    This right environment is not <MPS|MPS>, but rather trace(MPS)
    """

    sites_all = siteinds(mps)
    N_phys = div(length(mps), 2)

    R_id = Vector{ITensor}(undef, N_phys + 1)
    R_id[N_phys + 1] = ITensor(1.0)
    for n in N_phys:-1:1
        R_id[n] = contract_site_id(R_id[n+1], mps, sites_all, n)
    end

    return R_id

end

function contract_site_id(L, mps, sites, n)

    """Add identity to the right of environment L and contract"""

    ket = mps[2n-1]
    bra = mps[2n]
    s = sites[2n-1]
    sb = sites[2n]

    L = L * ket
    L = L * bra
    L = L * dag(delta(s, sb))
    return L
end

function contract_site_op(L, mps, sites, n, opname)

    """Add operator 'opname' to the right of environment L and contract"""

    ket = mps[2n-1]
    bra = mps[2n]
    s = sites[2n-1]
    sb = sites[2n]

    O = op(opname, s)          
    ket_Oped = noprime(O * ket)        

    L = L * ket_Oped
    L = L * bra
    L = L * dag(delta(s, sb))

    return L

end

function close_right(L, R)::ComplexF64

    """Contract environment left with right to get the expectation value"""

    return scalar(L * R)
end

function measure_T2_sweep(mps, R_id)::Vector{ComplexF64}

    """
    Function to calculate the T2_configs of a purified mps produced by rho_vec_to_mps(). 
    Returns a length N vector T2[n] with the expectation values of T2 at N

    Works by doing a single sweep left to right applying Q2 at site n to an environment of only identities to the left L_id
    and applying Qa at site n to another environment L_Qa which has the operator Qa applied to all p < n left sites
    summed, that is, L_Qa = Σ_{p=1}^{n-1} (Qa_p). Using these two environments combined with prebuilt right environments
    R_id which have the contraction of all sites p > n, we can calculate the expectation value of Q2_n in a left to right 
    sweep and all the correlators <Qa_p Qa_n> for all p < n while we are acting on site n. After calculating the expectation
    values we update L_id by adding identity on site n and we update L_Qa by adding the operator with all identities to the
    left of n and operator Qa on site n. 
    """

    sites_all = siteinds(mps)
    N_phys    = div(length(mps), 2)

    # Initialize results (T2) and left environments of pure identities L_id, and of Qa at all sites p < n L_Qa
    T2 = zeros(ComplexF64, N_phys)
    T2_running = 0.0 + 0.0im
    L_id = ITensor(1.0)
    L_Qx = ITensor(0.0)
    L_Qy = ITensor(0.0)
    L_Qz = ITensor(0.0)

    for n in 1:N_phys

        # Expectation values of Q2_n
        L_mid_Q2 = contract_site_op(L_id, mps, sites_all, n, "Q2") # L_id[n]-Q2
        q2_n = close_right(L_mid_Q2, R_id[n+1])                    # L_id[n]-Q2-R_id[n]

        # Expectation values of correlators
        if n > 1
            # Add Qa on site n to the left L_Qa environments to create all QaQa combinations
            L_cx = contract_site_op(L_Qx, mps, sites_all, n, "Qx")
            L_cy = contract_site_op(L_Qy, mps, sites_all, n, "Qy")
            L_cz = contract_site_op(L_Qz, mps, sites_all, n, "Qz")

            # L_Qa[n]-Qa = Σ_{p=1}^{n-1} <Qa_p Qa_n>
            corr_x = close_right(L_cx, R_id[n+1])
            corr_y = close_right(L_cy, R_id[n+1])
            corr_z = close_right(L_cz, R_id[n+1])
        else
            corr_x = 0.0 + 0.0im
            corr_y = 0.0 + 0.0im
            corr_z = 0.0 + 0.0im
        end

        # Calculate <T2_n> since T2_n[n] = T2_n[n-1] + (terms involving site n)
        T2_running += 3 * q2_n + 2 * (corr_x + corr_y + corr_z)
        T2[n] = T2_running

        # Update left environments for the NEXT iteration

        # Create "seeds" by adding Qa at the end of L_id, so creating the new terms of L_Qa
        new_seed_x = contract_site_op(L_id, mps, sites_all, n, "Qx")
        new_seed_y = contract_site_op(L_id, mps, sites_all, n, "Qy")
        new_seed_z = contract_site_op(L_id, mps, sites_all, n, "Qz")

        if n > 1
            # Extend old environments through site n with identity
            L_Qx_ext = contract_site_id(L_Qx, mps, sites_all, n)
            L_Qy_ext = contract_site_id(L_Qy, mps, sites_all, n)
            L_Qz_ext = contract_site_id(L_Qz, mps, sites_all, n)

            # Add new terms with identities before n and Qa_n
            L_Qx = L_Qx_ext + new_seed_x
            L_Qy = L_Qy_ext + new_seed_y
            L_Qz = L_Qz_ext + new_seed_z
        else
            L_Qx = new_seed_x
            L_Qy = new_seed_y
            L_Qz = new_seed_z
        end

        # Extend the identity environment all the way to n
        L_id = contract_site_id(L_id, mps, sites_all, n)
    end

    return T2
end

function measure_local_op_sweep(mps, opname, R_id)::Vector{ComplexF64}

    """
    Measure the expectation value of single-site operator opname at everypoint of the lattice in the purified MPS
    with the right environment being already given to the function.
        
    Algorithm follows the same idea as the T2_sweep function
    """
    
    sites_all = siteinds(mps)
    N_phys = div(length(mps), 2)

    # Initialize variables
    op_results = zeros(ComplexF64, N_phys)
    L_id = ITensor(1.0)

    # Single rightward sweep
    for n in 1:N_phys
        # Put on opname at the end of left identity environment
        L_with_op = contract_site_op(L_id, mps, sites_all, n, opname)

        # Calculate expectation value
        op_results[n] = close_right(L_with_op, R_id[n+1])

        # Update left environment for next loop
        L_id = contract_site_id(L_id, mps, sites_all, n)
    end

    return op_results

end

function measure_H_sweep(mps, g2, m, a, R_id)

    """
    Measure the expectation value of the energy separated by kinetic, mass and electric contributions
    Algorithm follows the same idea as the T2_sweep function
    """

    sites_all = siteinds(mps)
    N_phys    = div(length(mps), 2)

    E_kin  = 0.0 + 0.0im
    E_el   = 0.0 + 0.0im
    E_mass = 0.0 + 0.0im

    # Initialize left environments
    L_id     = ITensor(1.0)
    L_DeltaZ = ITensor(0.0)
    L_SmSp   = ITensor(0.0)
    L_SpSm   = ITensor(0.0)

    # Singular rightward sweep
    for n in 1:N_phys

        # Mass energy: We use N-1 because julia starts enumeration with 1
        if m != 0.0
            L_Ntot = contract_site_op(L_id, mps, sites_all, n, "N_tot")
            E_mass += m * (-1)^(n-1) * close_right(L_Ntot, R_id[n+1])
        end

        # Electric energy: Diagonal term (if statement because of n-N prefactor)
        if n < N_phys
            L_1mSzSz = contract_site_op(L_id, mps, sites_all, n, "1-SzSz")
            E_el += (a*g2/2) * (3.0/8.0) * (N_phys-n) * close_right(L_1mSzSz, R_id[n+1])
        end

        # Kintetic energy: Two-body terms (if statement because of n-N prefactor)
        if n < N_phys
            coef_kin = -1.0/(2.0*a)
            for (op_left, op_right) in (("SpSz", "SmS0"), ("S0Sp", "SzSm"),
                                        ("SmSz", "SpS0"), ("S0Sm", "SzSp"))

                # We extend the nth L_id twice with op_left and op_right, then close with R_id[n+2]
                L_tmp = contract_site_op(L_id, mps, sites_all, n, op_left)
                L_tmp2 = contract_site_op(L_tmp, mps, sites_all, n+1, op_right)
                E_kin += coef_kin * close_right(L_tmp2, R_id[n+2])
            end
        end

        # Electric energy: Off-diagonal term which has all to all terms 
        if n > 1 && n < N_phys   # n=1 has no left partners, n=N_phys has coef 0
            coef_n = Float64(N_phys - n)  

            if coef_n > 0.0
                # DeltaZ * DeltaZ pairs
                coef_DZ = (a*g2/2) * (1.0/8.0) * coef_n
                L_cDZ = contract_site_op(L_DeltaZ, mps, sites_all, n, "DeltaZ")
                E_el += coef_DZ * close_right(L_cDZ, R_id[n+1])

                # SmSp * SpSm pairs
                coef_hop = (a*g2/2) * coef_n
                L_cSmSp = contract_site_op(L_SmSp, mps, sites_all, n, "SpSm")
                E_el += coef_hop * close_right(L_cSmSp, R_id[n+1])

                # SpSm * SmSp pairs
                L_cSpSm = contract_site_op(L_SpSm, mps, sites_all, n, "SmSp")
                E_el += coef_hop * close_right(L_cSpSm, R_id[n+1])
            end
        end

        # Update left environments for next loop

        # Seeds: L_id extended through n with the operator inserted at n
        seed_DeltaZ = contract_site_op(L_id, mps, sites_all, n, "DeltaZ")
        seed_SmSp = contract_site_op(L_id, mps, sites_all, n, "SmSp")
        seed_SpSm = contract_site_op(L_id, mps, sites_all, n, "SpSm")

        if n > 1
            # Extend existing environment through site n with identity
            L_DeltaZ = contract_site_id(L_DeltaZ, mps, sites_all, n) + seed_DeltaZ
            L_SmSp = contract_site_id(L_SmSp, mps, sites_all, n) + seed_SmSp
            L_SpSm = contract_site_id(L_SpSm, mps, sites_all, n) + seed_SpSm
        else
            L_DeltaZ = seed_DeltaZ
            L_SmSp = seed_SmSp
            L_SpSm = seed_SpSm
        end

        # Extend identity boundary
        L_id = contract_site_id(L_id, mps, sites_all, n)
    end

    # Simply add all energies for total
    E_tot = E_kin + E_el + E_mass

    return E_kin, E_el, E_mass, E_tot
end


# TODO:
# Test cutoffs, test tec_1, test tec_2, test MPO_measuring_algorithm