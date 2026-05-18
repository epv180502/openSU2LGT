using ITensor
using LinearAlgebra: kron

# -------------------------------------------- SiteType Creation ---------------------------------------------
# TODO: DOC STRING. CHECK THE PRIMES ARE CORRECT 
function ITensors.space(::SiteType"SU2_packed"; conserve_qns=false)
    # TODO: May be possible to reduce it to two since r0 and 0g should be r0 - 0g
    return 4
end

function ITensors.state(::StateName"rg", ::SiteType"SU2_packed", s::Index)
    v = zeros(4)
    v[1] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"r0", ::SiteType"SU2_packed", s::Index)
    v = zeros(4)
    v[2] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"0g", ::SiteType"SU2_packed", s::Index)
    v = zeros(4)
    v[3] = 1
    return ITensor(v, s)
end

function ITensors.state(::StateName"00", ::SiteType"SU2_packed", s::Index)
    v = zeros(4)
    v[4] = 1
    return ITensor(v, s)
end
                
# Global identity 
function ITensors.op(::OpName"Id", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Id", s1) ⊗ op("Id", s2), prime(s), s)
end

# Hopping operators
function ITensors.op(::OpName"SpSz", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("S+", s1) ⊗ op("Z", s2), prime(s), s)
end
function ITensors.op(::OpName"SmS0", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("S-", s1) ⊗ op("Id", s2), prime(s), s)
end
function ITensors.op(::OpName"S0Sp", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Id", s1) ⊗ op("S+", s2), prime(s), s)
end
function ITensors.op(::OpName"SzSm", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Z", s1) ⊗ op("S-", s2), prime(s), s)
end

# Mass operators
function ITensors.op(::OpName"SpS0", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("S+", s1) ⊗ op("Id", s2), prime(s), s)
end
function ITensors.op(::OpName"S0Sm", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Id", s1) ⊗ op("S-", s2), prime(s), s)
end

function ITensors.op(::OpName"N_r", ::SiteType"SU2_packed", s::Index)
    return op("SpS0", s) * op("SmS0", s)
end
function ITensors.op(::OpName"N_g", ::SiteType"SU2_packed", s::Index)
    return op("S0Sp", s) * op("S0Sm", s)
end

function ITensors.op(::OpName"N_tot", ::SiteType"SU2_packed", s::Index)
    return op("N_r", s) + op("N_g", s)
end

# Electric operators
function ITensors.op(::OpName"SzSz", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Z", s1) ⊗ op("Z", s2), prime(s), s)
end
function ITensors.op(::OpName"1-SzSz", ::SiteType"SU2_packed", s::Index)
    return op("Id", s) - op("SzSz", s)
end

function ITensors.op(::OpName"SmSp", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("S-", s1) ⊗ op("S+", s2), prime(s), s)
end
function ITensors.op(::OpName"SpSm", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("S+", s1) ⊗ op("S-", s2), prime(s), s)
end

function ITensors.op(::OpName"SzS0", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Z", s1) ⊗ op("Id", s2), prime(s), s)
end
function ITensors.op(::OpName"S0Sz", ::SiteType"SU2_packed", s::Index)
    s1 = siteind("S=1/2")
    s2 = siteind("S=1/2")
    return ITensor(op("Id", s1) ⊗ op("Z", s2), prime(s), s)
end
function ITensors.op(::OpName"DeltaZ", ::SiteType"SU2_packed", s::Index)
    return op("SzS0", s) - op("S0Sz", s)
end

# Extra operators
function ITensors.op(::OpName"N_pair", ::SiteType"SU2_packed", s::Index)
    return op("N_r", s) * op("N_g", s)
end
function ITensors.op(::OpName"N_single", ::SiteType"SU2_packed", s::Index)
    return op("N_tot", s) - 2*op("N_pair", s)
end
function ITensors.op(::OpName"N_zero", ::SiteType"SU2_packed", s::Index)
    return op("Id", s) - op("N_single", s) - op("N_pair", s)
end


# -------------------------------------------- Hamiltonian Creation ---------------------------------------------
# What is lambda => Gauge protection QUESTION: Do we need it?
# QUESTION: Do we need background field? l_0 -> For now I remove them

# NOTE: Hamiltonian Updated
function get_aH_Hamiltonian(sites, g2, m, a)

    """
    This gives aH Hamiltonian acting on state vectors and is used to begin the system in an eigenstate

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing
    """

    N = length(sites)

    opsum = OpSum()

    for n in 1:N-1
        
        for m in n+1:N
            
            # Long range ZZ interaction term
            opsum += (a*g2/2)*(1/8)*(N-m),"DeltaZ",n,"DeltaZ",m

            # Long range hopping interaction term
            opsum += (a*g2/2)*(N-m),"SmSp",n,"SpSm",m
            opsum += (a*g2/2)*(N-m),"SpSm",n,"SmSp",m

        end

        # Kinetic term
        opsum += -(1/(2*a)),"SpSz",n,"SmS0",n+1
        opsum += -(1/(2*a)),"S0Sp",n,"SzSm",n+1

        # Inverse Z term
        opsum += (a*g2/2)*(3/8)*(N-n),"1-SzSz",n
        
        # Mass term
        opsum += (m*(-1)^(n-1)),"N_tot",n

    end

    # The for loop on top only goes to N-1 so we add the last term manually
    opsum += (m*(-1)^(N-1)),"N_tot",N

    return MPO(opsum, sites)

end

function apply_odd!(odd, mps, cutoff, maxdim)

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

function ishermitian(mpo; tol = 1e-14)

    return norm(mpo - dag(swapprime(mpo, 0, 1))) < tol

end

function ispositive(mpo; tol = 1e-14)

    """
    We check this by using DMRG to find the ground state energy of the "density matrix" MPO
    """

    n = length(mpo)
    dmrg_tol = tol
    cutoff = tol
    sites = dag(reduce(vcat, siteinds(mpo; :plev => 0)))
    state = [isodd(i) ? "0" : "1" for i = 1:n]
    psi0 = randomMPS(sites, state)
    obs = DMRGObserver(;energy_tol = dmrg_tol)
    nsweeps = 100
    dmrg_energy, _ = dmrg(mpo, psi0; nsweeps, cutoff, observer=obs, outputlevel=1)

    return abs(dmrg_energy) < tol

end

function lowest_eval_mpo(mpo; tol = 1e-9)

    n = length(mpo)
    dmrg_tol = tol
    cutoff = tol
    sites = dag(reduce(vcat, siteinds(mpo; :plev => 0)))
    state = [isodd(i) ? "0" : "1" for i = 1:n]
    psi0 = randomMPS(sites, state)
    nsweeps = 100
    dmrg_energy, _ = dmrg(mpo, psi0; nsweeps, cutoff, outputlevel=1, ishermitian = false)

    return dmrg_energy

end

# NOTE: Renamed some variables to add a dependence
function environment_correlator(type, n, m, D, inputs)

    if type == "constant"
        return D
    elseif type == "delta"
        if n == m
            return D
        else
            return 0.0
        end
    else # gaussian case
        sigma = inputs["sigma"]
        return D*exp(-0.5*(1/sigma)^2*(n-m)^2)
    end

end

# NOTE: Updated to SU2 vacuum
function get_dirac_vacuum_mps(sites; flip_sites = [])

    N = length(sites)
    state = [isodd(n) ? "rg" : "00" for n = 1:N]
    state = []
    for n in 1:N
        if isodd(n)
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

# NOTE: Updated site name from S=1/2 to SU2_packed
function rho_vec_to_mps(rho_vec)

    N = length(rho_vec)

    mps = MPS(2*N)

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

# NOTE: Hamiltonian Updated
function get_double_aH_Hamiltonian(sites, g2, m, a, side)

    """
    This gives aH Hamiltonian acting on one side of a vectorized density matrices on a given side. Side specifies "left" or 
    "right" to imply H tensor product I or vice versa so that is H*rho or rho*H

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing

    """

    N = div(length(sites), 2)

    opsum = OpSum()

    # In this loop, n is the physical index, n_idx is the index of the tensor in the MPS
    for n in 1:N-1

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in n+1:N

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

function trace_mps(mps)

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

# NOTE: Updated operator to measure to be arbitrary operator you input
function measure_op(mps, opname, site)

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

# NOTE: Updated operator to measure to be arbitrary operator you input
function measure_op_config(mps, opname; left = true)

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

# TODO: Liouvillian Update
function get_aLm_aLndag(n, m, aT, sites)

    """
    Gives the operator aLm tensor product aLndagger so aLm acts on the left side system and aLndagger on the right
    """

    N = div(length(sites), 2)
    
    n_phys, m_phys = div(n, 2), div(m + 1, 2)

    opsum = OpSum()

    opsum += 0.25*(-1)^(n_phys+m_phys),"Z",m,"Z",n

    opsum += 0.25*(-1)^(n_phys+m_phys),"Z",n

    opsum += 0.25*(-1)^(n_phys+m_phys),"Z",m

    opsum += 0.25*(-1)^(n_phys+m_phys),"Id",1

    if (n_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",n-2,"S+",n
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",n-2,"S-",n
    end

    if (n_phys != N)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",n,"S+",n+2
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",n,"S-",n+2
    end

    if (m_phys != 1)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",m-2,"S+",m
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",m-2,"S-",m
    end
    
    if (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",m,"S+",m+2
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",m,"S-",m+2
    end

    if (m_phys != 1)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",m-2,"S+",m,"Z",n
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",m-2,"S-",m,"Z",n
    end
    
    if (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",m,"S+",m+2,"Z",n
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",m,"S-",m+2,"Z",n
    end

    if (n_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"Z",m,"S-",n-2,"S+",n
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"Z",m,"S+",n-2,"S-",n
    end
    
    if (n_phys != N)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"Z",m,"S-",n,"S+",n+2
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"Z",m,"S+",n,"S-",n+2
    end

    if (n_phys != 1) && (m_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m-2,"S-",m,"S+",n-2,"S-",n
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m-2,"S-",m,"S-",n-2,"S+",n
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m-2,"S+",m,"S+",n-2,"S-",n
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m-2,"S+",m,"S-",n-2,"S+",n
    end

    if (m_phys != 1) && (n_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m-2,"S-",m,"S-",n,"S+",n+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m-2,"S-",m,"S+",n,"S-",n+2
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m-2,"S+",m,"S+",n,"S-",n+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m-2,"S+",m,"S-",n,"S+",n+2
    end

    if (m_phys != N) && (n_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m,"S-",m+2,"S-",n-2,"S+",n
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m,"S-",m+2,"S+",n-2,"S-",n
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m,"S+",m+2,"S+",n-2,"S-",n
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m,"S+",m+2,"S-",n-2,"S+",n
    end
    
    if (n_phys != N) && (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m,"S-",m+2,"S+",n,"S-",n+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",m,"S-",m+2,"S-",n,"S+",n+2
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m,"S+",m+2,"S-",n,"S+",n+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",m,"S+",m+2,"S+",n,"S-",n+2
    end

    return opsum
    
end

# TODO: Liouvillian Update
function get_aLndag_aLm(n, m, aT, sites, side)

    """
    Gives the operator aL_n_dagger tensor product aL_m can be either acting all on the left or all on the right
    """

    N = div(length(sites), 2)

    if side == "left"
        n_phys, m_phys = div(n + 1, 2), div(m + 1, 2)
    else
        n_phys, m_phys = div(n, 2), div(m, 2)
    end
    
    opsum = OpSum()

    opsum += 0.25*(-1)^(n_phys + m_phys),"Z",n,"Z",m

    opsum += 0.25*(-1)^(n_phys + m_phys),"Z",n

    opsum += 0.25*(-1)^(n_phys + m_phys),"Z",m

    if side == "left"
        opsum += 0.25*(-1)^(n_phys + m_phys),"Id",1
    else
        opsum += 0.25*(-1)^(n_phys + m_phys),"Id",2
    end

    if (n_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",n-2,"S+",n
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",n-2,"S-",n
    end

    if (n_phys != N)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",n,"S+",n+2
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",n,"S-",n+2
    end

    if (m_phys != 1)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",m-2,"S+",m
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",m-2,"S-",m
    end
    
    if (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",m,"S+",m+2
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",m,"S-",m+2
    end

    if (n_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S-",n-2,"S+",n,"Z",m
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S+",n-2,"S-",n,"Z",m
    end
    
    if (n_phys != N)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"S-",n,"S+",n+2,"Z",m
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"S+",n,"S-",n+2,"Z",m
    end

    if (m_phys != 1)
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"Z",n,"S-",m-2,"S+",m
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"Z",n,"S+",m-2,"S-",m
    end
    
    if (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(16*aT)),"Z",n,"S-",m,"S+",m+2
        opsum += ((-1)^(n_phys + m_phys)/(16*aT)),"Z",n,"S+",m,"S-",m+2
    end

    if (n_phys != 1) && (m_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n-2,"S-",n,"S+",m-2,"S-",m
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n-2,"S-",n,"S-",m-2,"S+",m
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n-2,"S+",n,"S+",m-2,"S-",m
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n-2,"S+",n,"S-",m-2,"S+",m
    end

    if (n_phys != 1) && (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n-2,"S-",n,"S-",m,"S+",m+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n-2,"S-",n,"S+",m,"S-",m+2
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n-2,"S+",n,"S+",m,"S-",m+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n-2,"S+",n,"S-",m,"S+",m+2
    end

    if (n_phys != N) && (m_phys != 1)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n,"S-",n+2,"S-",m-2,"S+",m
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n,"S-",n+2,"S+",m-2,"S-",m
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n,"S+",n+2,"S+",m-2,"S-",m
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n,"S+",n+2,"S-",m-2,"S+",m
    end
    
    if (n_phys != N) && (m_phys != N)
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n,"S-",n+2,"S+",m,"S-",m+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S+",n,"S-",n+2,"S-",m,"S+",m+2
        opsum += (-(-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n,"S+",n+2,"S-",m,"S+",m+2
        opsum += ((-1)^(n_phys + m_phys)/(64*aT^2)),"S-",n,"S+",n+2,"S+",m,"S-",m+2
    end

    return opsum

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
    tmp2 = MPO(sites, "Id")
    
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

# TODO: Liouvillian Update
function get_Lindblad_opsum_without_l0_terms(sites, g2, m, a, aT, aD, env_corr_type, inputs, dissipator_sites)

    N = div(length(sites), 2)
    res = -1im*get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, "left")
    res += 1im*get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, "right")
    
    if aD != 0
        for n in dissipator_sites
            for m in dissipator_sites
                if env_corr_type == "delta" && n != m
                    continue
                end
                res += environment_correlator(env_corr_type, n, m, D, inputs) * ( get_aLm_aLndag(2*n, 2*m-1, aT, sites) - 0.5 * get_aLndag_aLm(2*n-1, 2*m-1, aT, sites, "left") - 0.5 * get_aLndag_aLm(2*n, 2*m, aT, sites, "right") )
            end
        end
    end

    return res

end

# NOTE: Hamiltonian Updated. 
function get_double_aH_Hamiltonian_without_l0_terms(sites, g2, m, a, side)

    """
    This gives aH Hamiltonian acting on one side of a vectorized density matrices on a given side. Side specifies "left" or 
    "right" to imply H tensor product I or vice versa so that is H*rho or rho*H

    g2 = Gauge coupling square g**2
    m = Mass
    a = lattice spacing

    """

    N = div(length(sites), 2)

    opsum = OpSum()

    # In this loop, n is the physical index, n_idx is the index of the tensor in the MPS
    for n in 1:N-1

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in n+1:N

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

function measure_mpo(mps, mpo; alg = "none")

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

# NOTE: Hamiltonian Updated. 
function get_double_aH_Hamiltonian_individual_terms(N, g2, m, a, side)

    """
    This gives aH Hamiltonian

    side specifies "left" or "right" to imply H tensor product I or vice versa

    """

    opsum_kinetic_term = OpSum()
    opsum_mass_term = OpSum()
    opsum_electric_field_term = OpSum()

    for n in 1:N-1

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        for m in n+1:N

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
