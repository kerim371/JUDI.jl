export mse, studentst, fwa, icf, visco_misfit

using FFTW

"""
    mse(x, y)

Mean square error

    `5f0 * norm(x - y, 2)^2`

and its derivative w.r.t `x`

    `x-y`

"""
function mse(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Number}
    f = T(.5) * norm(x - y, 2)^2
    r = x - y
    return f, r
end

"""
studentst(x, y)

Student's T misfit 

    `.5 * (k+1) * log(1 + (x-y)^2 / k)`

and its derivative w.r.t x

    `(k + 1) * (x - y) / (k + (x - y)^2)`

"""
function studentst(x::AbstractArray{T}, y::AbstractArray{T}; k=T(2)) where {T<:Number}
    k = convert(T, k)
    f = sum(_studentst_loss.(x, y, k))
    r = (k + 1) .* (x - y) ./ (k .+ (x - y).^2)
    return f, r
end

_studentst_loss(x::T, y::T, k::T) where {T<:Number} = T(1/2) * (k + 1) * log(1 + (x-y)^2 / k)


"""
    fwa(x, y)

Frequency-weighted-amplitude (FWA) misfit inspired by Eq. (23-25) in the
visco-acoustic Q-FWI formulation. The implementation uses a trace-wise FFT
frequency weighting and returns `(objective, adjoint_source)`.
"""
function fwa(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Real}
    X = _as_traces(x)
    Y = _as_traces(y)
    size(X) == size(Y) || throw(DimensionMismatch("fwa expects x/y of identical size"))

    nt, nr = size(X)
    ω = _abs_omega(nt, one(T))

    r = similar(X)
    f = Base.zero(T)

    @inbounds for ir = 1:nr
        ux = @view X[:, ir]
        uy = @view Y[:, ir]

        Ux = fft(ux)
        Uy = fft(uy)

        Ax = real(ifft(ω .* abs.(Ux)))
        Ay = real(ifft(ω .* abs.(Uy)))
        ΔA = Ax .- Ay

        f += T(0.5) * sum(abs2, ΔA)

        RU = fft(ΔA) .* ω .* Ux ./ (abs.(Ux) .+ Base.eps(T))
        r[:, ir] .= real(ifft(RU))
    end

    return f, _reshape_like(r, x)
end


"""
    icf(x, y)

Instantaneous-centroid-frequency-inspired misfit for Q inversion.

To preserve compatibility with the existing architecture (time-domain adjoint
source expected by the propagator), this implementation uses a short-time,
frequency-weighted surrogate in the Fourier domain and returns
`(objective, adjoint_source)`.
"""
function icf(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Real}
    X = _as_traces(x)
    Y = _as_traces(y)
    size(X) == size(Y) || throw(DimensionMismatch("icf expects x/y of identical size"))

    nt, nr = size(X)
    ω = _abs_omega(nt, one(T))

    r = similar(X)
    f = Base.zero(T)

    @inbounds for ir = 1:nr
        ux = @view X[:, ir]
        uy = @view Y[:, ir]

        Ux = fft(ux)
        Uy = fft(uy)
        Ax2 = abs2.(Ux)
        Ay2 = abs2.(Uy)

        cux = sum(ω .* Ax2) / (sum(Ax2) + Base.eps(T))
        cuy = sum(ω .* Ay2) / (sum(Ay2) + Base.eps(T))
        Δc = cux - cuy

        f += T(0.5) * Δc^2

        # Adjoint source surrogate with centroid weighting.
        W = (ω .- cux) ./ (sum(Ax2) + Base.eps(T))
        r[:, ir] .= real(ifft(2 * Δc .* W .* Ux))
    end

    return f, _reshape_like(r, x)
end


"""
    visco_misfit(name::Symbol)

Factory for visco-oriented misfits:
`visco_misfit(:wf)`, `visco_misfit(:fwa)`, `visco_misfit(:icf)`.
"""
function visco_misfit(name::Symbol)
    name === :wf && return mse
    name === :fwa && return fwa
    name === :icf && return icf
    throw(ArgumentError("Unknown visco misfit $(name). Expected :wf, :fwa or :icf"))
end


_as_traces(x::AbstractVector) = reshape(x, :, 1)
_as_traces(x::AbstractMatrix) = x
_reshape_like(x::AbstractMatrix, ref::AbstractVector) = vec(x)
_reshape_like(x::AbstractMatrix, ref::AbstractMatrix) = x

function _abs_omega(nt::Int, dt::T) where {T<:Real}
    df = inv(T(nt) * dt)
    ω = Vector{T}(undef, nt)
    nh = fld(nt, 2)
    @inbounds for k = 1:nt
        fk = (k - 1) <= nh ? (k - 1) * df : (k - 1 - nt) * df
        ω[k] = abs(T(2) * T(π) * fk)
    end
    return ω
end
