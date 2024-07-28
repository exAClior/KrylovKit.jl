using LinearAlgebra
using KrylovKit
using InteractiveUtils
using KrylovKit: BlockLanczos
using SparseArrays

function randomUnitaryMatrix(N::Int)
    # from https://discourse.julialang.org/t/how-to-generate-a-random-unitary-matrix-perfectly-in-julia/34102
    x = (sprand(N, N, 0.05) + sprand(N, N,0.05) * im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR .== 0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    return u
end

function degenerate_hamiltonian(dim, degeneracy)
    eigs = sort(randn(dim - degeneracy + 1))
    eigs = Diagonal([eigs[1] * ones(degeneracy - 1); eigs])
    U = randomUnitaryMatrix(dim)
    return sparse(U' * eigs * U)
end

using Yao
2^7
h = degenerate_hamiltonian(2^7, 4)

p = 2^4 
X0 = sparse(qr(rand(eltype(h), size(h))).Q)[:, 1:p]

T̃ = eigsolve(h, X0, 10, :SR, BlockLanczos(; krylovdim=6))