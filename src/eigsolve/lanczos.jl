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
        end
    end
    # obtain symmetric tridiagonal matrix that has the same 
    # eigenvalues as the hermitian tridiaonal matrix 
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix#Similarity_to_symmetric_tridiagonal_matrix
    # this is valid, tested in smaller dimension
    return SymTridiagonal(Vector(real.(diag(T̃))), Vector(abs.(diag(T̃, 1))))
end

_dot(::Nothing, b) = b
_dot(A::AbstractMatrix, b) = A*b

# https://github.com/SciML/ExponentialUtilities.jl/issues/32
function icgs!(A::AbstractMatrix, j, M=nothing; alpha=0.5, maxit=3)
    u = @view A[:,j]
    Q = @view A[:,1:j-1]
    Mu = _dot(M, u)
    r_pre = sqrt(abs(dot(u, Mu)))
    local r1
    for it = 1:maxit
        u .-= Q * (Q' * Mu)
        Mu = _dot(M, u)
        r1 = sqrt(abs(dot(u, Mu)))
        if r1 > alpha * r_pre
            break
        end
        r_pre = r1
    end
    normalize!(A[:,j])
    if r1 <= alpha * r_pre
        @warn "loss of orthogonality @icgs."
    end
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

    Xs = zeros(eltype(X1),n,(1+r)*p) 
    Xs[:,p+1:2*p] = X1
    Ms = Matrix{eltype(X1)}[]
    Bs = Matrix{eltype(X1)}[] # could be made upper triangular

    # iterations to construct Krylov subspace and its QR decomposition iteratively
    # each of this iteration should be moved to initialize and then expand of BlockLanczosIteration and Factorization
    for i in 1:(r - 1)
        U_i = A * Xs[:, (i * p + 1):((i + 1) * p)] - (i == 1 ? spzeros(eltype(X1), n, p) :
                                                      Xs[:, ((i - 1) * p + 1):(i * p)] * Bs[end]')
        M_i = Xs[:, (i * p + 1):((i + 1) * p)]' * U_i
        R_i = U_i - Xs[:, (i * p + 1):((i + 1) * p)] * M_i
        F = qr(R_i)
        X_i, B_i = Matrix(F.Q), F.R
        push!(Ms, M_i)
        push!(Bs, B_i)
        Xs[:, ((i + 1) * p + 1):((i + 2) * p)] = X_i
        for j in collect(((i + 1) * p + 1):((i + 2) * p))
            icgs!(Xs,j)
        end
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

function eigsolve(A, x₀, howmany::Int, which::Union{Symbol,Selector}, alg::BlockLanczos)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")


    n, p = size(x₀)
    @assert alg.krylovdim * p <= n "Dimension of the Krylov subspace is too large"

    iter = LanczosIterator(A, x₀, alg.orth)


    T̃ = block_tridiagonalize(A, x₀, alg.krylovdim)

    T̃ = tridiag_sym_band_mtx(T̃, p)
    return sort(eigvals(T̃))[1:howmany]
end
