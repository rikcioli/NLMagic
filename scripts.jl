using Base.Iterators
using Plots
using LaTeXStrings
using LinearAlgebra
using OptimKit
using ITensors, ITensorMPS
using IterTools
include("opt.jl")


const CX = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
const H = [1 1; 1 -1]/sqrt(2)
const Id = (1+0im)*[1. 0.; 0. 1.]
const X = [0. 1.; 1. 0.]
const Z = [1. 0.; 0. -1.]
const Y = [0. -1.0im; 1.0im 0.]
const T = [1. 0.; 0. exp(1im*pi/4)]
const paulis = [Id, X, Z, Y]

function Idn(n)
    I = zeros(ComplexF64,n,n)
    for i in 1:n
        I[i, i] = 1
    end
    return I
end

"Returns random N x N unitary matrix sampled with Haar measure"
function random_unitary(N::Int)
    x = (randn(N,N) + randn(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    
    return u
end 



function ptrace(rho::AbstractMatrix, dA::Int, dB::Int; sys = :B)
    rho4 = reshape(rho, dB, dA, dB, dA)  # kronecker product works inversely to reshaping
    if sys === :B
        # trace over B
        out = zeros(eltype(rho), dA, dA)
        @inbounds @views for b=1:dB
            out .+= rho4[b,:,b,:]
        end
        return out
    elseif sys === :A
        out = zeros(eltype(rho), dB, dB)
        @inbounds @views for a=1:dA
            out .+= rho4[:,a,:,a]
        end
        return out
    else
        error("sys must be :A or :B")
    end
end

function ptrace(rho::AbstractMatrix, dL::Int, dA::Int, dR::Int)
    @assert size(rho, 1) == size(rho, 2) == dL*dA*dR

    rho6 = reshape(rho, dR, dA, dL, dR, dA, dL)

    # sum over L and L’ and R and R’
    out = zeros(eltype(rho), dA, dA)
    @inbounds begin
        for l=1:dL, r=1:dR
            @views out .+= rho6[r, :, l, r, :, l]
        end
    end
    return out
end


function genPaulis(N)
    all_paulis = []
    for comb in Base.product(ntuple(_ -> 1:4, N)...)
        pauli_comb = [paulis[comb[i]] for i in 1:N]
        push!(all_paulis, reduce(kron, pauli_comb))
    end
    return all_paulis
end

function gradMagic(U::AbstractMatrix, psi, all_paulis)
    N = Int(log2(length(psi)))
    magic = 0.
    grad = zeros(4,4) 

    Upsi = U*psi
    for P in all_paulis
        braket = Upsi' * P * Upsi
        magic += abs2(braket)^2
        grad += 8*abs2(braket)*conj(braket)*P*Upsi*(psi')/(2^N)
    end

    magic = magic/(2^N)
    grad = -grad/magic

    magic = -log(magic)
    riemann_grad = project(U, grad)
    @show norm(riemann_grad)

    return magic, riemann_grad
end

function gradMagic(Uvec::Vector{Matrix{T}}, psi, all_paulis) where {T}
    N = Int(log2(length(psi)))
    magic = 0.
    n_unitaries = length(Uvec)
    sizes = [size(U)[1] for U in Uvec]
    grad = [zeros(ComplexF64,size,size) for size in sizes]

    Upsi = reduce(kron, Uvec)*psi
    for P in all_paulis
        braket = Upsi' * P * Upsi
        magic += abs2(braket)^2
        for i in eachindex(grad)
            Ucontr = [j==i ? Idn(sizes[j]) : Uvec[j]' for j in 1:n_unitaries]
            fullcontr = reduce(kron, Ucontr)*P*Upsi*(psi')/(2^N)
            if i==1
                redcontr = ptrace(fullcontr, sizes[1], div(2^N,sizes[1]); sys=:B)
            elseif i==n_unitaries
                redcontr = ptrace(fullcontr, div(2^N,sizes[end]), sizes[end]; sys=:A)
            else
                redcontr = ptrace(fullcontr, reduce(*, sizes[1:i-1]), sizes[i], reduce(*, sizes[i+1:end]))
            end
            grad[i] += 8*abs2(braket)*conj(braket)*redcontr
        end
    end

    magic = magic/(2^N)
    grad = -grad/magic

    magic = -log(magic)
    riemann_grad = project(Uvec, grad)
    #@show norm(riemann_grad)

    return magic, riemann_grad
end


const PAULI_LABELS = ["I", "X", "Y", "Z"]

function get_pauli_op(label, s)
    return op(label, s)
end

function pauli_string_mpo(pauli_labels::Vector{String}, sites)
    N = length(pauli_labels)
    return MPO([get_pauli_op(pauli_labels[n], sites[n]) for n in 1:N])
end

function pauli_expectation(psi::MPS, pauli_labels::Vector{String})
    P = pauli_string_mpo(pauli_labels, siteinds(psi))
    val = ITensorMPS.inner(psi', P, psi)  # Computes ⟨ψ|P|ψ⟩
    return real(val)
end

function psi_mq_mps(psi::MPS, q::Int=2)
    N = length(psi)
    d = 2^N
    acc = 0.0

    total = 4^N
    #prog = Progress(total, desc="Summing Pauli strings")

    for label_indices in IterTools.product(fill(1:4, N)...)
        pauli_labels = [PAULI_LABELS[i] for i in label_indices]
        ev = pauli_expectation(psi, pauli_labels)
        acc += (ev^(2q)) / d

        #next!(prog)
    end

    return -log2(acc)
end

function getState(h,N,psi0) 
    sites = siteinds(psi0) #conserve_szparity=true
    ampo = OpSum()
    for j = 1:N-1
        ampo .+=  -4.0,"Sz",j,"Sz",j+1 #exchanged x with z to match daniele's formula
        ampo .+=  -2.0*h,"Sx",j
    end
    ampo .+=  -4.0,"Sz",1,"Sz",N
    ampo .+=  -2.0*h,"Sx",N
    
    H = MPO(ampo,sites)
    
    #state = ["Up" for n=1:N]
    
    #psi0 = randomMPS(sites, 2)
    #symmetrize!(psi0)
    sweeps = Sweeps(20) # number of sweeps is 20
    maxdim!(sweeps, 50, 100, 200, 500) # gradually increase states kept (200 -10 prima)
    cutoff!(sweeps,1E-12)
    #noise!(sweeps, 1E-6, 1E-7, 1E-8, 0.0)
    energy,psi = dmrg(H,psi0,sweeps; outputlevel=1, ) 

    return psi, sites, energy
end 


function M2_NL(h)
    term1 = 2 * log2((h - 1) * h + 1)
    term2 = log2(81)
    sqrt_term = sqrt((h - 1) * h + 1)

    inner = 10 * sqrt_term + 1 +
        h * (
            -18 * sqrt_term + 1 +
            h * (
                h * (77 * h + 4 * sqrt_term - 154) -
                6 * (sqrt_term - 36)
            ) - 139
        ) + 71

    term3 = -log2(inner)

    return term1 + term2 + term3
end


function run()
    N = 3
    all_paulis = genPaulis(N)

    psi = zeros(ComplexF64, 2^N)
    psi[1] = 1
    #psi = kron(T*H, Id)*[1; 0; 0; 0]
    
    E = []
    m = []
    nl_m = []
    sites = siteinds("S=1/2",N )
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    psi0 = productMPS(sites,state)
    #psi0 = random_mps(ComplexF64, sites; linkdims=2)

    for h in 1.5:-0.1:0.0
        psi, sites = getState(h, N, psi0)
        psi0 = psi
        #H = [ expHermitian(op("Sy",sites,i), -2*pi/2*1im) for i=1:N] 
        #phi = apply(H,psi) 
        sites = removeqns(sites)
        psi = MPS([removeqns(psi[i]) for i in 1:N])
        println("h = ", h)

        #println("Entropy: ", psi_mq_mps(phi))
        push!(E, psi_mq_mps(psi, 2))
        #println("M(h): ", M(h))
        #push!(m, M(h))

        U0 = [random_unitary(4), random_unitary(2)]
        psi = reshape(Array{ComplexF64}(reduce(*, psi), sites), 2^N)
        fg = U -> gradMagic(U, psi, all_paulis)
        algorithm = LBFGS(5; maxiter = 3000, gradtol = 1E-8, verbosity = 2)
        Umin, NLmagic, _ = optimize(fg, U0, algorithm; retract = retract, inner = inner, transport! = transport!, isometrictransport=true);
        
        push!(nl_m, NLmagic)
    end

    plot = Plots.plot(0.0:0.1:1.5, reverse(E), label="Numerical M_2", marker = :circle, xlabel="h", ylabel="M_2", legend=:bottomright)
    #plot(0.0:0.1:1.5, m, label="Analytical M(h)", xlabel="h", ylabel="M", legend=:bottomright)
    Plots.plot!(plot, 0.0:0.1:1.5, reverse(nl_m), label="Numerical Non-local M(h) tripartite", marker =:circle)
    #plot!(0.0:0.1:1.5, [M2_NL(h) for h in 0.0:0.1:1.5], label="Analytical Non-local M(h)")

    

    # # example of saddle point
    # P(theta) = [1. 0.; 0. exp(theta*1im)] 
    # chi(theta) = kron(P(theta)*H, Id)*[1;0;0;0]
    # U0 = (1+0im)*kron(Id, Id)

    # magics = [gradMagic(U0, chi((t/100)*pi/2))[1] for t in 0:100]
    # plot = Plots.plot(0:100, magics, legend=:bottomright)
    return plot, E, nl_m
end


results = run()
Plots.plot!(results[1], 0.0:0.1:1.5, [M2_NL(h) for h in 0.0:0.1:1.5], label="Analytical Non-local M(h)")

              
# TEST RIEMANNIAN GRADIENT

# function genPoint()
#     Uvec = [mt.random_unitary(4) for _ in 1:2]
#     return Uvec, [U' for U in Uvec]
# end
# 
# function genTanVec(U)
#     V = [randn(ComplexF64, 4, 4) for _ in 1:2]
#     V = skew(V)
#     V = U .* V
#     V /= sqrt(inner(V,V))
# end
# 
# function testGrad(genPoint::Function, genTanVec::Function, computeCostGrad::Function, inner::Function, retract::Function)
#     U0, U0dag = genPoint()
#     V = genTanVec(U0)
#     func, grad = computeCostGrad(U0)
#     gradV = inner(grad, V) 
#     E = t -> abs(computeCostGrad(retract(U0, V, t)[1])[1] - func - t*gradV)
# 
#     tvals = exp10.(-8:0.1:0)
#     plot = Plots.plot(tvals, E.(tvals), yscale=:log10, xscale=:log10, legend=:bottomright)
#     Plots.plot!(plot, tvals, tvals .^2, yscale=:log10, xscale=:log10, label=L"O(t^2)")
#     Plots.plot!(plot, tvals, tvals, yscale=:log10, xscale=:log10, label=L"O(t)")
#     return plot
# end
# 
# psi = zeros(ComplexF64, 16)
# psi[1] = 1
# plot = testGrad(genPoint, genTanVec, U -> gradMagic(U, psi, genPaulis(4)), inner, retract)
# 