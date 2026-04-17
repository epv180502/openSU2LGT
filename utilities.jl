# TODO: Here we define the hamiltonian so we need to change this to SU2 
function get_aH_Hamiltonian(sites, x, l_0, ma, lambda)

    """
    This gives aH Hamiltonian
    """

    N = length(sites)

    opsum = OpSum()

    for n in 1:N-1
        
        for m in n+1:N
            
            # Long range ZZ interaction term
            opsum += 0.25*(1/x)*(N-m+lambda),"Z",n,"Z",m

        end

        # Kinetic term
        opsum += 0.5,"S+",n,"S-",n+1
        opsum += 0.5,"S-",n,"S+",n+1

        opsum += (1/x)*(N/8 - 0.25*ceil((n-1)/2) + l_0*(N-n)/2),"Z",n
        
        opsum += (0.5*ma*(-1)^(n-1)),"Z",n

    end

    opsum += (0.5*ma*(-1)^(N-1)),"Z",N

    opsum += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x) + (lambda*N/(8*x))),"Id",1

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

function environment_correlator(type, n, m, aD, inputs)

    if type == "constant"
        return aD
    elseif type == "delta"
        if n == m
            return aD
        else
            return 0.0
        end
    else # gaussian case
        sigma_over_a = inputs["sigma_over_a"]
        return aD*exp(-0.5*(1/sigma_over_a)^2*(n-m)^2)
    end

end

function get_dirac_vacuum_mps(sites; flip_sites = [])

    N = length(sites)
    state = [isodd(n) ? "1" : "0" for n = 1:N]
    state = []
    for n in 1:N
        if isodd(n)
            if n in flip_sites
                push!(state, "0")
            else
                push!(state, "1")
            end
        else
            if n in flip_sites
                push!(state, "1")
            else
                push!(state, "0") 
            end
        end
    end
    mps = MPS(sites, state)
   
    return mps

end

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
        replacetags!(U, "S=1/2,Site,n=$(i)", "S=1/2,Site,n=$(2*i-1)") # change the label of the second physical index from i to 2*i-1
        mps[2*i-1] = U

        # Fix V indices
        if i != N # if i == N there is no right link
            replacetags!(V, "Link,l=$(i)", "Link,l=$(2*i)") # same as the line above but for the right link
        end
        replacetags!(V, "Link,u", "Link,l=$(2*i-1)") # change tag of svd link to mps convention (here it is Link,u because of V = S*V above)
        replacetags!(V, "S=1/2,Site,n=$(i)", "S=1/2,Site,n=$(2*i)") # change the label of the second physical index from i to 2*i
        # V = reverse_dir(V, inds(V; :tags => "Site")[1]) # reverse the physical leg direction from in to out - this is mandatory for constructing MPO with autoMPO
        mps[2*i] = V

    end

    return noprime(mps)

end

# TODO: Define the SU(2) hamiltonan here
function get_double_aH_Hamiltonian(sites, x, l_0, ma, lambda, side)

    """
    This gives aH Hamiltonian

    side specifies "left" or "right" to imply H tensor product I or vice versa

    """

    N = div(length(sites), 2)

    opsum = OpSum()

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
            opsum += 0.25*(1/x)*(N-m+lambda),"Z",n_idx,"Z",m_idx

        end

        # Kinetic term
        opsum += 0.5,"S+",n_idx,"S-",n_idx+2
        opsum += 0.5,"S-",n_idx,"S+",n_idx+2

        opsum += (1/x)*(N/8 - 0.25*ceil((n-1)/2) + l_0*(N-n)/2),"Z",n_idx
        
        opsum += (0.5*ma*(-1)^(n-1)),"Z",n_idx

    end

    if side == "left"
        opsum += (0.5*ma*(-1)^(N-1)),"Z",2*N-1
        opsum += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x) + (lambda*N/(8*x))),"Id",1
    else
        opsum += (0.5*ma*(-1)^(N-1)),"Z",2*N
        opsum += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x) + (lambda*N/(8*x))),"Id",2
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

function measure_z(mps, site)

    sites = siteinds(mps)
    l = length(mps)

    z = op("Z", sites[site])
    mps = apply(z, mps)

    i = 1
    left, right = 2*i-1, 2*i
    res = mps[left]*mps[right]*dag(delta(sites[left], sites[right]))

    for i in 2:div(l,2)
        left, right = 2*i-1, 2*i
        res *= mps[left]*mps[right]*dag(delta(sites[left], sites[right]))
    end

    return res[1]

end

function measure_z_config(mps; left = true)

    n = length(mps)
    z_config = []
    if left
        for site in 1:2:n
            push!(z_config, measure_z(mps, site))
        end
    else
        for site in 2:2:n
            push!(z_config, measure_z(mps, site))
        end
    end

    return z_config

end

# TODO: Define the aLm tensor product aLndagger operator for our SU(2) model
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

# TODO: Define the aL_n_dagger tensor product aL_m operator for our SU(2) model
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

function get_Lindblad_opsum_without_l0_terms(sites, x, ma, lambda, aT, aD, env_corr_type, inputs, dissipator_sites)

    N = div(length(sites), 2)
    res = -1im*get_double_aH_Hamiltonian_without_l0_terms(sites, x, ma, lambda, "left")
    res += 1im*get_double_aH_Hamiltonian_without_l0_terms(sites, x, ma, lambda, "right")
    
    if aD != 0
        for n in dissipator_sites
            for m in dissipator_sites
                if env_corr_type == "delta" && n != m
                    continue
                end
                res += environment_correlator(env_corr_type, n, m, aD, inputs) * ( get_aLm_aLndag(2*n, 2*m-1, aT, sites) - 0.5 * get_aLndag_aLm(2*n-1, 2*m-1, aT, sites, "left") - 0.5 * get_aLndag_aLm(2*n, 2*m, aT, sites, "right") )
            end
        end
    end

    return res

end

function get_double_aH_Hamiltonian_without_l0_terms(sites, x, ma, lambda, side)

    """
    This gives aH Hamiltonian

    side specifies "left" or "right" to imply H tensor product I or vice versa

    """

    N = div(length(sites), 2)

    opsum = OpSum()

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
            opsum += 0.25*(1/x)*(N-m+lambda),"Z",n_idx,"Z",m_idx

        end

        # Kinetic term
        opsum += 0.5,"S+",n_idx,"S-",n_idx+2
        opsum += 0.5,"S-",n_idx,"S+",n_idx+2
        
        opsum += (0.5*ma*(-1)^(n-1)),"Z",n_idx

    end

    if side == "left"
        opsum += (0.5*ma*(-1)^(N-1)),"Z",2*N-1
    else
        opsum += (0.5*ma*(-1)^(N-1)),"Z",2*N
    end

    return opsum

end

function get_double_aH_Hamiltonian_just_l0_terms(sites, x, l_0, lambda, side)

    """
    This gives aH Hamiltonian

    side specifies "left" or "right" to imply H tensor product I or vice versa

    """

    N = div(length(sites), 2)

    opsum = OpSum()

    for n in 1:N-1

        if side == "left"
            n_idx = 2*n-1
        else
            n_idx = 2*n 
        end
        
        opsum += (1/x)*(N/8 - 0.25*ceil((n-1)/2) + l_0*(N-n)/2),"Z",n_idx
        
    end

    if side == "left"
        opsum += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x) + (lambda*N/(8*x))),"Id",1
    else
        opsum += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x) + (lambda*N/(8*x))),"Id",2
    end

    return opsum

end

function get_Lindblad_opsum_just_l0_terms(sites, x, l_0, lambda)

    res = -1im*get_double_aH_Hamiltonian_just_l0_terms(sites, x, l_0, lambda, "left")
    res += 1im*get_double_aH_Hamiltonian_just_l0_terms(sites, x, l_0, lambda, "right")
    
    return res

end

function get_applied_field(which_applied_field, inputs, t_over_a)

    l_0_1 = inputs["l_0_1"]
    
    if which_applied_field == "constant"
        return l_0_1
    else
        l_0_2 = inputs["l_0_2"]
        a_omega = inputs["a_omega"]
        if which_applied_field == "sauter"
            return l_0_1 + l_0_2/cosh(a_omega*t_over_a)^2
        elseif which_applied_field == "gaussian"
            return l_0_1 + l_0_2*exp(-(a_omega*t_over_a)^2)
        else # oscillatory case
            return l_0_1 + l_0_2*cos(a_omega*t_over_a)
        end
    end

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

# TODO: Define SU(2) hamiltonian here
function get_double_aH_Hamiltonian_individual_terms(N, x, l_0, side)

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
            opsum_electric_field_term += 0.25*(1/x)*(N-m),"Z",n_idx,"Z",m_idx

        end

        # Kinetic term
        opsum_kinetic_term += 0.5,"S+",n_idx,"S-",n_idx+2
        opsum_kinetic_term += 0.5,"S-",n_idx,"S+",n_idx+2

        opsum_electric_field_term += (1/x)*(N/8 - 0.25*ceil((n-1)/2) + l_0*(N-n)/2),"Z",n_idx
        
        opsum_mass_term += (0.5*(-1)^(n-1)),"Z",n_idx

    end

    if side == "left"
        opsum_mass_term += (0.5*(-1)^(N-1)),"Z",2*N-1
        opsum_electric_field_term += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x)),"Id",1
    else
        opsum_mass_term += (0.5*(-1)^(N-1)),"Z",2*N
        opsum_electric_field_term += ((l_0^2)*(N-1)/(2*x) + (l_0*N)/(4*x) + (N^2)/(16*x)),"Id",2
    end

    return opsum_kinetic_term, opsum_electric_field_term, opsum_mass_term

end
