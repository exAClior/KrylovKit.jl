using LinearAlgebra
using KrylovKit
using InteractiveUtils
using Printf
using KrylovKit: Selector
using KrylovKit: BlockLanczos

function RandomUnitaryMatrix(N::Int)
    # from https://discourse.julialang.org/t/how-to-generate-a-random-unitary-matrix-perfectly-in-julia/34102
    x = (rand(N,N) + rand(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    return u
end


function degenerate_hamiltonian(dim, degeneracy)
    eigs = sort(randn(dim - degeneracy +1))
    eigs = Diagonal([eigs[1] * ones(degeneracy-1); eigs])
    U = RandomUnitaryMatrix(dim)
    return U' * eigs* U 
end


h = degenerate_hamiltonian(30,4)


@which eigsolve(h,rand(eltype(h),size(h,1)), 10, :SR, Lanczos())

@show sort(evals_kk)
@show sort(evals_la)

function eigsovle(A, X0, howmany, which, alg::BlockLanczos)
    A = h 
    n, p = size(X0)
    @assert alg.krylovdim * p <= n "Dimension of the Krylov subspace is too large"

    Xs = Matrix{eltype(A)}[X0]
    Ms = Matrix{eltype(A)}[]
    Bs = Matrix{eltype(A)}[]
    push!(Xs,(qr(rand(eltype(A), n,n)).Q)[:,1:p])
    push!(Ms, Xs[end]' * A * Xs[end])
    for k in 1:(r-1)
        R_k = A * Xs[k] - Xs[k] * Ms[k] - (k == 1 ? zeros(eltype(A),n,p) : Xs[k-1] * Bs[k-1]')
        X_kp1, B_k = qr(R_k)
        push!(Xs, X_kp1[:,1:p])
        push!(Bs, B_k)
        push!(Ms, X_kp1[:,1:p]' * A * X_kp1[:,1:p])
    end


    T_k = SchwartzAlgo(Ms,Bs) 

end

function SchwartzAlgo(Ms, Bs)
    # implement this https://doi.org/10.1007/BF02162505
    return T_k
end