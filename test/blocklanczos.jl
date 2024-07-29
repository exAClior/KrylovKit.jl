using LinearAlgebra
using KrylovKit
using InteractiveUtils
using KrylovKit: BlockLanczos
using SparseArrays
using BenchmarkTools
using KrylovKit: block_tridiagonalize, tridiag_sym_band_mtx

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

# using Yao
# function toric_code_strings(m::Int, n::Int)
# 	li = LinearIndices((m, n))
# 	bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
# 	right(i, j) = li[mod1(i, m), mod1(j, n)]
# 	xstrings = Vector{Int}[]
# 	zstrings = Vector{Int}[]
# 	for i=1:m, j=1:n
# 		# face center
# 		push!(xstrings, [bottom(i, j-1), right(i, j), bottom(i, j), right(i-1, j)])
# 		# cross
# 		push!(zstrings, [right(i, j), bottom(i, j), right(i, j+1), bottom(i+1, j)])
# 	end
# 	return xstrings, zstrings
# end

# function toric_code_hamiltonian(m::Int, n::Int)
# 	xstrings, zstrings = toric_code_strings(m, n)
# 	sum([kron(2m*n, [i=>X for i in xs]...) for xs in xstrings[1:end-1]]) + sum([kron(2m*n, [i=>Z for i in zs]...) for zs in zstrings[1:end-1]])
# end

# p = 2^4 
# h = toric_code_hamiltonian(3, 3)
# X0 = (qr(sprand(eltype(mat(h)), size(mat(h),1),2^-10)).Q)[:, 1:p]
# evals = eigsolve(-mat(h), X0, 10, :SR, BlockLanczos(; krylovdim=6))

p = 2^2 
h = degenerate_hamiltonian(2^6, 4)
X0 = (qr(sprand(eltype(h), size(h,1),2^-2)).Q)[:, 1:p] # how to generate random and orthogonal vectors
X0 = spzeros(eltype(h),size(h,1),p)
for i in 1:p
	X0[i,i] = 1.0
end

evals = eigsolve(h, X0, 10, :SR, BlockLanczos(; krylovdim=6))

band_h = block_tridiagonalize(h,X0)

real.(eigvals(Matrix(h)))

real.(eigvals(Matrix(band_h)))

evals