function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth))
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up

    # Initialize Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity - 2)
    numops = 1
    numiter = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    converged = 0
    local D, U, f
    while true
        β = normres(fact)
        K = length(fact)

        # diagonalize Krylov factorization
        if β <= tol
            if K < howmany
                @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`); setting `howmany = $K`."
                howmany = K
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany)
            U = copyto!(view(UU, 1:K, 1:K), I)
            f = view(HH, K + 1, 1:K)
            T = rayleighquotient(fact) # symtridiagonal

            # compute eigenvalues
            if K == 1
                D = [T[1, 1]]
                f[1] = β
                converged = Int(β <= tol)
            else
                if K < krylovdim
                    T = deepcopy(T)
                end
                D, U = tridiageigh!(T, U)
                by, rev = eigsort(which)
                p = sortperm(D; by=by, rev=rev)
                D, U = permuteeig!(D, U, p)
                mul!(f, view(U, K, :), β)
                converged = 0
                while converged < K && abs(f[converged + 1]) <= tol
                    converged += 1
                end
            end

            if converged >= howmany
                break
            elseif alg.verbosity > 1
                msg = "Lanczos eigsolve in iter $numiter, krylovdim = $K: "
                msg *= "$converged values converged, normres = ("
                msg *= @sprintf("%.2e", abs(f[1]))
                for i in 2:howmany
                    msg *= ", "
                    msg *= @sprintf("%.2e", abs(f[i]))
                end
                msg *= ")"
                @info msg
            end
        end

        if K < krylovdim# expand Krylov factorization
            fact = expand!(iter, fact; verbosity=alg.verbosity - 2)
            numops += 1
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            # Restore Lanczos form in the first keep columns
            H = fill!(view(HH, 1:(keep + 1), 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[keep + 1, j] = f[j]
            end
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(U, h')
            end
            @inbounds for j in 1:keep
                fact.αs[j] = H[j, j]
                fact.βs[j] = H[j + 1, j]
            end

            # Update B by applying U using Householder reflections
            B = basis(fact)
            B = basistransform!(B, view(U, :, 1:keep))
            r = residual(fact)
            B[keep + 1] = scale!!(r, 1 / β)

            # Shrink Lanczos factorization
            fact = shrink!(fact, keep)
            numiter += 1
        end
    end

    if converged > howmany
        howmany = converged
    end
    values = D[1:howmany]

    # Compute eigenvectors
    V = view(U, :, 1:howmany)

    # Compute convergence information
    vectors = let B = basis(fact)
        [B * v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [scale(r, last(v)) for v in cols(V)]
    end
    normresiduals = let f = f
        map(i -> abs(f[i]), 1:howmany)
    end

    if alg.verbosity > 0
        if converged < howmany
            @warn """Lanczos eigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Lanczos eigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

"""
    tridiag_sym_band_mtx(T̃::AbstractMatrix{T}, m::Int) where {T}

Tridiagonalize a band tridiagonal matrix T̃.

# Arguments
- `T̃::AbstractMatrix{T}`: The input tridiagonal symmetric band matrix from which to construct the tridiagonal matrix.
- `m::Int`: The bandwidth of the resulting tridiagonal symmetric band matrix.

# Returns
A tridiagonal symmetric matrix.
"""
function tridiag_sym_band_mtx(T̃::AbstractMatrix{T},m::Int) where {T}
    # implement this https://doi.org/10.1007/BF02162505
    n = size(T̃, 1)
    @inbounds for jj in 1:(n - 2) # eliminating elements in col jj
        for kk in min(m, n - jj):-1:2 # eliminating element in row kk
            grot, _ = givens(T̃, jj + kk - 1, jj + kk, jj)
            T̃ = grot * T̃ * grot'
            jj + kk + m > n && continue
            # eliminating element that deviates from tridiagonal symmetric band matrix
            for μ in 1:floor(Int, (n - kk - jj) / m)
                grot, _ = givens(T̃, jj + kk + μ * m - 1, jj + kk + μ * m,
                                 jj + kk + (μ - 1) * m - 1)
                T̃ = grot * T̃ * grot'
            end
            droptol!(T̃, 1e-16)
        end
    end
    # obtain symmetric tridiagonal matrix that has the same 
    # eigenvalues as the hermitian tridiaonal matrix 
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix#Similarity_to_symmetric_tridiagonal_matrix
    # this is valid, tested in smaller dimension
    return SymTridiagonal(Vector(real.(diag(T̃))), Vector(abs.(diag(T̃, 1))))
end

"""
    block_tridiagonalize(A::AbstractMatrix{T}, X1, r)

Block tridiagonalize a matrix `A` using the Block Lanczos algorithm.

# Arguments
- `A::AbstractMatrix{T}`: The matrix to be block tridiagonalized.
- `X1`: The initial block vector.
- `r::Int`: The number of iterations of Lanczos.

# Returns
- `T`: The block tridiagonal matrix.
"""
function block_tridiagonalize(A::AbstractMatrix{T},X1,r::Int) where T
    n,p = size(X1)

    Xprev = spzeros(eltype(X1), n,p)

    Ms = SparseMatrixCSC{eltype(X1),Int}[]
    Bs = SparseMatrixCSC{eltype(X1),Int}[] # could be made upper triangular

    push!(Ms, X1' * A * X1) # 

    # iterations to construct Krylov subspace and its QR decomposition iteratively 
    @inbounds for k in 1:(r - 1)
        @assert X1' * X1 ≈ I "X1 is not orthonormal"
        R_k = A * X1 - X1 * Ms[k] -
              (k == 1 ? spzeros(eltype(X1), n, p) : Xprev * Bs[k - 1]')
        X_kp1, B_k = qr(R_k)
        Xprev = X1
        X1 = Matrix(X_kp1) # maybe not first p? lost of orthogonality here
        push!(Bs, B_k)
        push!(Ms, X1' * A * X1)
    end

    # construct the block tridiagonal matrix from blocks vector
    T̃ = spzeros(eltype(X1), size(Ms[1]) .* length(Ms))
    m = size(Ms[1], 1)  # because M[i] and B[i] are upper triangular
    @inbounds for idx in eachindex(Ms)
        T̃[(1 + (idx - 1) * m):(idx * m), (1 + (idx - 1) * m):(idx * m)] = Ms[idx]
        if idx > 1
            T̃[(1 + (idx - 1) * m):(idx * m), (1 + (idx - 2) * m):((idx - 1) * m)] = Bs[idx - 1]
        end
        if idx < length(Ms)
            T̃[(1 + (idx - 1) * m):(idx * m), (1 + idx * m):((idx + 1) * m)] = Bs[idx]'
        end
    end
    return T̃
end

function eigsolve(A, X0, howmany::Int, which::Union{Symbol,Selector}, alg::BlockLanczos)
    n, p = size(X0)
    @assert alg.krylovdim * p <= n "Dimension of the Krylov subspace is too large"

    T̃ = block_tridiagonalize(A, X0, alg.krylovdim)

    T̃ = tridiag_sym_band_mtx(T̃, p)
    return sort(eigvals(T̃))[1:howmany]
end

function icgs(u::Vector{T}, Q::Matrix{T}, M::Union{Matrix{T}, Nothing}=nothing, maxiter::Int=3) where T
    """
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        u::Vector{T}: the column vector to be orthogonalized.
        Q::Matrix{T}: the search space.
        M::Union{Matrix{T}, Nothing}: the matrix, if provided, perform M-orthogonal.
        return_norm::Bool: return the norm of u.
        maxiter::Int: the maximum number of iterations.

    Return:
        Vector{T}, orthogonalized vector u.
    """
    @assert ndims(u) == 2
    uH, QH = u', Q'
    alpha = 0.5
    it = 1
    Mu = isnothing(M) ? u : M * u
    r_pre = norm(uH * Mu)
    for it in 1:maxiter
        u = u - Q * (QH * Mu)
        Mu = isnothing(M) ? u : M * u
        r1 = norm(uH * Mu)
        if r1 > alpha * r_pre
            break
        end
        r_pre = r1
    end
    if r1 <= alpha * r_pre
        @warn "loss of orthogonality @icgs."
    end
    return u, r1
end
