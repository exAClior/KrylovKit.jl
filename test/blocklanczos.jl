using LinearAlgebra
using KrylovKit
using InteractiveUtils
using KrylovKit: BlockLanczos
using SparseArrays
using BenchmarkTools

function randomUnitaryMatrix(N::Int)
    # from https://discourse.julialang.org/t/how-to-generate-a-random-unitary-matrix-perfectly-in-julia/34102
    x = (sprand(N, N, 0.05) + sprand(N, N,0.05) * im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR .== 0] .= 1
    diagRm = spdiagm(diagR)
    u = f.Q * diagRm
    return u
end

function degenerate_hamiltonian(dim, degeneracy)
    eigs = sort(randn(dim - degeneracy + 1))
    eigs = spdiagm([eigs[1] * ones(degeneracy - 1); eigs])
    U = randomUnitaryMatrix(dim)
    return sparse(U' * eigs * U), eigs
end

using Yao
function toric_code_strings(m::Int, n::Int)
	li = LinearIndices((m, n))
	bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
	right(i, j) = li[mod1(i, m), mod1(j, n)]
	xstrings = Vector{Int}[]
	zstrings = Vector{Int}[]
	for i=1:m, j=1:n
		# face center
		push!(xstrings, [bottom(i, j-1), right(i, j), bottom(i, j), right(i-1, j)])
		# cross
		push!(zstrings, [right(i, j), bottom(i, j), right(i, j+1), bottom(i+1, j)])
	end
	return xstrings, zstrings
end

function toric_code_hamiltonian(m::Int, n::Int)
	xstrings, zstrings = toric_code_strings(m, n)
	sum([kron(2m*n, [i=>X for i in xs]...) for xs in xstrings[1:end-1]]) + sum([kron(2m*n, [i=>Z for i in zs]...) for zs in zstrings[1:end-1]])
end

using Random
Random.seed!(1234)
k = 10
p = Int(exp2(ceil(log2(k))))
krylovdim = 30
h = mat(toric_code_hamiltonian(3, 3))
X0 = (qr(sprandn(eltype(h), size(h,1),2^-3)).Q)[:, 1:p]
evals = eigsolve(-h, X0, k, :SR, BlockLanczos(; krylovdim=krylovdim));
evals2,_ = eigsolve(-h, randn(size(h,1)), k, :SR, Lanczos(;krylovdim=30));
evals # does to give me exactly the degeneracy but close
evals2



k = 10
p = Int(exp2(ceil(log2(k))))
krylovdim = 20
h, true_eigs = degenerate_hamiltonian(2^10, 4)
X0 = (qr(sprand(eltype(h), size(h,1),2^-4)).Q)[:, 1:p]

evals = eigsolve(h, X0, k, :SR, BlockLanczos(; krylovdim=krylovdim));
evals2,_ = eigsolve(h, randn(size(h,1)), 10, :SR, Lanczos(;krylovdim=30));
evals - sort(diag(true_eigs))[1:k]
evals2

# orthogonality of X_i https://www.cas.mcmaster.ca/~qiao/publications/spie05.pdf, https://apps.dtic.mil/sti/pdfs/ADA224011.pdf