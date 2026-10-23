# ============================================================================
# Gradient filters for post-processing the FWI gradient per shot.
# Each filter is a callable struct with signature (g::PhysicalParameter, model) -> PhysicalParameter
# ============================================================================

export AbstractGradientFilter, WindowedFK, WindowedMedian, CompositeFilter

"""
    AbstractGradientFilter

Supertype for all gradient filters. Subtypes must implement

    (f::MyFilter)(g::PhysicalParameter, model) -> PhysicalParameter
"""
abstract type AbstractGradientFilter end

# ---------------------------------------------------------------------------
# Windowed FK fan filter
# ---------------------------------------------------------------------------
"""
    WindowedFK(; win_z, win_x, max_slope, z_inc=win_z÷2, x_inc=win_x÷2, taper=0.2)

Sliding-window FK fan filter for gradient images. In each window the 2-D FFT
is computed and only components with `|kx/kz| <= max_slope` are kept (i.e.
events whose dip `|dz/dx|` in *grid samples* is below `max_slope`).
`taper` controls the softness of the edge (0 = hard cut).

The filter operates on `g.data` laid out as `(nx, nz)` (JUDI convention) and
returns a `PhysicalParameter` with the same metadata.
"""
struct WindowedFK <: AbstractGradientFilter
    win_z::Int
    win_x::Int
    z_inc::Int
    x_inc::Int
    max_slope::Float32
    taper::Float32
end

function WindowedFK(; win_z::Int, win_x::Int, max_slope::Real,
                    z_inc::Int=div(win_z, 2), x_inc::Int=div(win_x, 2),
                    taper::Real=0.2)
    win_z > 1 && win_x > 1 || throw(ArgumentError("win_z, win_x must be > 1"))
    z_inc >= 1 && x_inc >= 1 || throw(ArgumentError("z_inc, x_inc must be >= 1"))
    max_slope > 0 || throw(ArgumentError("max_slope must be > 0"))
    0 <= taper <= 1 || throw(ArgumentError("taper must be in [0, 1]"))
    return WindowedFK(win_z, win_x, z_inc, x_inc,
                      Float32(max_slope), Float32(taper))
end

# --- core: windowed FK on a plain matrix (nz, nx) ---------------------------
function _windowed_fk_matrix(img::AbstractMatrix{Float32}, f::WindowedFK)
    nz, nx = size(img)
    (nz >= f.win_z && nx >= f.win_x) ||
        throw(ArgumentError("image ($nz,$nx) smaller than window ($(f.win_z),$(f.win_x))"))

    out    = zeros(Float32, nz, nx)
    weight = zeros(Float32, nz, nx)

    # triangular window weights
    wz = Float32[ (f.win_z + 1)/2 - abs(i - (f.win_z + 1)/2) for i in 1:f.win_z ]
    wx = Float32[ (f.win_x + 1)/2 - abs(i - (f.win_x + 1)/2) for i in 1:f.win_x ]
    w2d = wz * wx'

    # frequency axes (cycles/sample), centred
    kz = Float32.(fftshift(fftfreq(f.win_z, 1)))
    kx = Float32.(fftshift(fftfreq(f.win_x, 1)))

    lo = f.max_slope * (1f0 - f.taper)
    hi = f.max_slope * (1f0 + f.taper)

    G = Matrix{ComplexF32}(undef, f.win_z, f.win_x)
    for zs in 1:f.z_inc:(nz - f.win_z + 1)
        for xs in 1:f.x_inc:(nx - f.win_x + 1)
            win = view(img, zs:zs+f.win_z-1, xs:xs+f.win_x-1)
            G .= fftshift(fft(Float32.(win)))

            # apply fan mask in-place
            @inbounds for iz in 1:f.win_z
                akz = abs(kz[iz])
                for ix in 1:f.win_x
                    akx = abs(kx[ix])
                    r = akz < eps(Float32) ? (akx < eps(Float32) ? 0f0 : Inf32) : akx / akz
                    m = r <= lo ? 1f0 : (r >= hi ? 0f0 : (hi - r) / (hi - lo))
                    G[iz, ix] *= m
                end
            end

            filt = real(ifft(ifftshift(G)))
            @views out[zs:zs+f.win_z-1, xs:xs+f.win_x-1] .+= w2d .* filt
            @views weight[zs:zs+f.win_z-1, xs:xs+f.win_x-1] .+= w2d
        end
    end
    return out ./ max.(weight, eps(Float32))
end

function (f::WindowedFK)(g::PhysicalParameter, model)
    g_nx_nz = Float32.(g.data)                  # (nx, nz) — JUDI layout
    g_nz_nx = permutedims(g_nx_nz, (2, 1))      # (nz, nx) — filter layout
    g_filt  = _windowed_fk_matrix(g_nz_nx, f)
    g_back  = permutedims(g_filt, (2, 1))       # back to (nx, nz)
    return PhysicalParameter(g_back, g.n, g.d, g.o)
end


# ---------------------------------------------------------------------------
# Windowed median filter
# ---------------------------------------------------------------------------
"""
    WindowedMedian(; win_z=3, win_x=3)

Applies a 2-D spatial median filter over a sliding window of size `(win_z, win_x)`.
Useful for removing high-frequency salt-and-pepper noise (speckles) from the gradient.
"""
struct WindowedMedian <: AbstractGradientFilter
    win_z::Int
    win_x::Int
end

function WindowedMedian(; win_z::Int=3, win_x::Int=3)
    win_z >= 1 && win_x >= 1 || throw(ArgumentError("win_z, win_x must be >= 1"))
    return WindowedMedian(win_z, win_x)
end

# Fast in-place median for small arrays (insertion sort)
function _fast_median!(v::AbstractVector{Float32})
    n = length(v)
    @inbounds for i in 2:n
        x = v[i]
        j = i - 1
        while j >= 1 && v[j] > x
            v[j+1] = v[j]
            j -= 1
        end
        v[j+1] = x
    end
    return v[(n+1)÷2]
end

function _windowed_median_matrix(img::AbstractMatrix{Float32}, f::WindowedMedian)
    nz, nx = size(img)
    out = zeros(Float32, nz, nx)
    hz = f.win_z ÷ 2
    hx = f.win_x ÷ 2
    
    # Pre-allocate buffer for the window
    buf = Vector{Float32}(undef, f.win_z * f.win_x)
    
    @inbounds for x in 1:nx
        for z in 1:nz
            idx = 1
            for wx in max(1, x - hx):min(nx, x + hx)
                for wz in max(1, z - hz):min(nz, z + hz)
                    buf[idx] = img[wz, wx]
                    idx += 1
                end
            end
            out[z, x] = _fast_median!(view(buf, 1:idx-1))
        end
    end
    return out
end

function (f::WindowedMedian)(g::PhysicalParameter, model)
    g_nx_nz = Float32.(g.data)                  # (nx, nz) — JUDI layout
    g_nz_nx = permutedims(g_nx_nz, (2, 1))      # (nz, nx) — filter layout
    g_filt  = _windowed_median_matrix(g_nz_nx, f)
    g_back  = permutedims(g_filt, (2, 1))       # back to (nx, nz)
    return PhysicalParameter(g_back, g.n, g.d, g.o)
end

# ---------------------------------------------------------------------------
# Composite filter (chain multiple filters sequentially)
# ---------------------------------------------------------------------------
"""
    CompositeFilter(f1, f2, ...)

Applies multiple gradient filters in sequence: `f_n( ... f_2(f_1(g)) ... )`.
Exapmple:
JUDI.Options(
    smooth_func = CompositeFilter(
        WindowedFK(win_z=40, win_x=40, max_slope=0.4f0, taper=0.2f0),
        WindowedMedian(win_z=13, win_x=13),
        )
    )
"""
struct CompositeFilter <: AbstractGradientFilter
    filters::Tuple{Vararg{AbstractGradientFilter}}
end

CompositeFilter(filters::AbstractGradientFilter...) = CompositeFilter(filters)

function (f::CompositeFilter)(g::PhysicalParameter, model)
    for filt in f.filters
        g = filt(g, model)
    end
    return g
end