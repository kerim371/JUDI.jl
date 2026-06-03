# Time-domain extended-source FWI (ES-FWI) for modeled/SEG-Y data.
#
# This is a cleaned-up version of a field/model-data SPG script.  The key ES-FWI
# difference from conventional FWI is the call
#
#   fwi_objective(...; extended_source=true, es_options=ESFWIOptions(...))
#
# which estimates the source extension from the current data residual instead of
# drawing random low-rank spatial weights.

using Distributed

const DIR_LOGS = joinpath(@__DIR__, "logs")
mkpath(DIR_LOGS)
cd(DIR_LOGS)


# ----------------------------- Parallel setup -----------------------------
const LOCAL_WORKERS = parse(Int, get(ENV, "JUDI_LOCAL_WORKERS", "8"))
if LOCAL_WORKERS > 0 && nworkers() < LOCAL_WORKERS
    addprocs(LOCAL_WORKERS - nworkers() + 1)
end

@sync for p in workers()
    @async remotecall_fetch(() -> myid(), p)
end
println("All workers alive")

# Set GPU affinity before loading JUDI/Devito on workers.
@everywhere begin
    const N_GPU = parse(Int, get(ENV, "JUDI_NGPU", "2"))
    ENV["CUDA_VISIBLE_DEVICES"] = string(myid() % N_GPU)
    println("Process ID: ", myid(), "\tCUDA_VISIBLE_DEVICES: ", ENV["CUDA_VISIBLE_DEVICES"])
end

@everywhere using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles
@everywhere using JUDI, SlimOptim, NLopt, HDF5, SegyIO, Plots, ImageFiltering, NPZ
@everywhere using SetIntersectionProjection
@everywhere using JUDI.FFTW, Zygote, Flux

# ----------------------------- Local helpers ------------------------------
rho_from_slowness(m) = 0.23f0 .* (sqrt.(1f0 ./ m) .* 1000f0) .^ 0.25f0

function ormsby_wavelet(; dt, t, f1, f2, f3, f4)
    nt = Int(round(t / dt)) + 1
    nf = div(nt, 2) + 1
    freqs = Float32.((0:nf-1) ./ (nt * dt))
    amp = zeros(Float32, nf)
    for (i, f) in pairs(freqs)
        amp[i] = f < f1 ? 0f0 :
                 f < f2 ? Float32((f - f1) / max(f2 - f1, eps(Float32))) :
                 f <= f3 ? 1f0 :
                 f < f4 ? Float32((f4 - f) / max(f4 - f3, eps(Float32))) : 0f0
    end
    w = real(irfft(ComplexF32.(amp), nt))
    return Float32.(circshift(w, div(nt, 2)))
end

function save_data(x, z, data; pltfile, title, colormap=:viridis, clim=nothing,
                   h5file=nothing, h5openflag="w", h5varname="data")
    plt = heatmap(x, z, data; yflip=true, title=title, color=colormap, clim=clim)
    savefig(plt, pltfile * ".png")
    if h5file !== nothing
        h5open(h5file, h5openflag) do fid
            haskey(fid, h5varname) && delete_object(fid, h5varname)
            write(fid, h5varname, data)
        end
    end
end

function save_fhistory(fhistory; h5file, h5openflag="r+", h5varname="fhistory")
    h5open(h5file, h5openflag) do fid
        haskey(fid, h5varname) && delete_object(fid, h5varname)
        write(fid, h5varname, fhistory)
    end
end

# ----------------------------- User settings ------------------------------
const MODELING_TYPE = "bulk"           # "slowness" or "bulk"
const FREE_SURFACE = true
const LIMIT_M = true
const BUFFER_SIZE = 1000f0
const NB = 40

const EXTENDED_SOURCE = true
const ES_OPTIONS = ESFWIOptions(; mu=1f-3,
                                hessian_mode=get(ENV, "JUDI_ES_HESSIAN", "scalar"),
                                filter_eps=1f-3)

const SIGNAL_TYPE = "ormsby"           # "ormsby", "ricker", or "klauder"
const RICKER_FRQ = 0.008f0             # kHz
const ORMSBY = (0.0f0, 0.002f0, 0.050f0, 0.075f0)  # kHz
const KLAUDER = (fmin=0.002f0, fmax=0.02f0, slength=8.0f0,
                 taper=0.5f0, tshift=nothing, minphase=true)

const FRQ0 = 0.0f0                     # kHz
const FRQ1 = 0.005f0                   # kHz
const SEABED = 0.01f0                  # km
const VWATER = 1.5f0                   # km/s
const RHOWATER = 1.02f0                # g/cm^3

const PRESTK_DIR = joinpath(@__DIR__, "..", "DATA", "shots_ormsby_0.0-0.002-0.05-0.075Hz_1000.0ms_UNDER_fs")
const PRESTK_FILE = "shot"
const MODEL_FILE = joinpath(@__DIR__, "..", "DATA", "model_il892_1m_smooth.h5")
const DIR_OUT = joinpath(@__DIR__, "..", "DATA",
                         "es_fwi_spg_$(MODELING_TYPE)",
                         "$(FRQ0)Hz_$(FRQ1)Hz_nb$(NB)_buffer$(Int(round(BUFFER_SIZE)))m_$(ES_OPTIONS.hessian_mode)")
const MODEL_FILE_OUT = "model"
mkpath(DIR_OUT)

const SEGY_DEPTH_KEY_SRC = "SourceSurfaceElevation"
const SEGY_DEPTH_KEY_REC = "RecGroupElevation"

# ------------------------------ Data/model --------------------------------
container = segy_scan(PRESTK_DIR, PRESTK_FILE,
                      ["SourceX", "SourceY", "GroupX", "GroupY",
                       "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key=SEGY_DEPTH_KEY_REC)
src_geometry = Geometry(container; key="source", segy_depth_key=SEGY_DEPTH_KEY_SRC)

fid = h5open(MODEL_FILE, "r")
n = Tuple(Int64(i) for i in read(fid, "n"))
d = Tuple(Float32(i) for i in read(fid, "d"))
o = Tuple(Float32(i) for i in read(fid, "o"))
m0 = Float32.(read(fid, "m"))
close(fid)

# Clip starting model to sane velocity bounds.
const VMIN = 0.8f0
const VMAX = 7.0f0
const MMIN = (1f0 / VMAX)^2
const MMAX = (1f0 / VMIN)^2
m0[m0 .< MMIN] .= MMIN
m0[m0 .> MMAX] .= MMAX

# Optional densification of the starting model.
const DENSE_FACTOR = 0.1f0
i_dense = 1f0:1f0/DENSE_FACTOR:size(m0, 1)
j_dense = 1f0:1f0/DENSE_FACTOR:size(m0, 2)
m0 = interpolate(m0, BSpline(Linear()))(i_dense, j_dense)
n = size(m0)
d = Tuple(Float32(di / DENSE_FACTOR) for di in d)

if MODELING_TYPE == "slowness"
    model0 = Model(n, d, o, m0, nb=NB)
elseif MODELING_TYPE == "bulk"
    rho0 = rho_from_slowness(m0)
    model0 = Model(n, d, o, m0, rho=rho0, nb=NB)
else
    error("Unknown MODELING_TYPE=$(MODELING_TYPE)")
end

x = collect((o[1]:d[1]:o[1] + (n[1]-1)*d[1]) ./ 1000f0)
z = collect((o[2]:d[2]:o[2] + (n[2]-1)*d[2]) ./ 1000f0)
seabed_ind = [findfirst(zz -> zz > SEABED, z) for _ in eachindex(x)]

@info "modeling_type=$(MODELING_TYPE), n=$(n), d=$(d), o=$(o)"
@info "extended_source=$(EXTENDED_SOURCE), es_options=$(ES_OPTIONS)"

# ------------------------------ Wavelet -----------------------------------
dt = get_dt(src_geometry, 1)
fs = 1000f0 / src_geometry.dt[1]

function build_wavelet(src_geometry)
    nt = src_geometry.nt[1]
    if SIGNAL_TYPE == "ricker"
        return ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], RICKER_FRQ)
    elseif SIGNAL_TYPE == "ormsby"
        w = Matrix{Float32}(undef, nt, 1)
        f1, f2, f3, f4 = ORMSBY
        w[:, 1] = ormsby_wavelet(dt=src_geometry.dt[1]/1000f0, t=src_geometry.t[1]/1000f0,
                                  f1=f1*1000f0, f2=f2*1000f0, f3=f3*1000f0, f4=f4*1000f0)
        return w
    elseif SIGNAL_TYPE == "klauder"
        error("Klauder wavelet generation is intentionally left to a site-local utility; set SIGNAL_TYPE="ormsby" or "ricker" in this template.")
    else
        error("Unknown SIGNAL_TYPE=$(SIGNAL_TYPE)")
    end
end

wavelet = build_wavelet(src_geometry)
responsetype = FRQ0 == 0 ? Lowpass(FRQ1*1000f0; fs=fs) :
               isinf(FRQ1) ? Highpass(FRQ0*1000f0; fs=fs) :
               Bandpass(FRQ0*1000f0, FRQ1*1000f0; fs=fs)
wavelet[:, 1] = filt(digitalfilter(responsetype, Butterworth(5)), wavelet[:, 1])
q = judiVector(src_geometry, wavelet)

# -------------------------- Preconditioners/options ------------------------
Ml_ref = judiDataMute(q.geometry, d_obs.geometry, vp=8000, t0=0.3f0,
                      mode=:reflection, taperwidth=20)
Ml_tur = judiDataMute(q.geometry, d_obs.geometry, vp=500, t0=-0.1f0,
                      mode=:turning, taperwidth=20)
Ml_freq = judiFilter(d_obs.geometry, FRQ0*1000f0, FRQ1*1000f0)

mminArr = fill(MMIN, size(model0))
mmaxArr = fill(MMAX, size(model0))
model0.m[model0.m .< mminArr] .= MMIN
model0.m[model0.m .> mmaxArr] .= MMAX

jopt = JUDI.Options(IC="fwi",
                    limit_m=LIMIT_M,
                    buffer_size=BUFFER_SIZE,
                    optimal_checkpointing=false,
                    free_surface=FREE_SURFACE,
                    space_order=16)

Dm = judiDepthScaling(model0)
Il = inv(judiIllumination(model0))

# ------------------------------ Misfit -------------------------------------
@everywhere n2(x) = x / norm(x, 2)
@everywhere function Hilbert(x)
    n = size(x, 1)
    σ = sign.(-n/2+1:n/2)
    return imag(ifft(fftshift(σ .* fftshift(fft(x, 1), 1), 1), 1))
end
@everywhere HLoss(dsyn, dobs) = sum(abs2.((dsyn - dobs) .+ 1im .* Hilbert(dsyn - dobs)))
@everywhere function envelope(dsyn, dobs)
    ϕ = HLoss(n2(dsyn), n2(dobs))
    g = Zygote.gradient(xs -> HLoss(n2(xs), n2(dobs)), dsyn)
    return ϕ, real.(g[1])
end

# ----------------------------- Optimization --------------------------------
const NITERATIONS = 30
const SHOT_FROM = 1
const SHOT_STEP = 4
const SHOT_TO = d_obs.nsrc
const MUTE_REFLECTIONS = false
const MUTE_TURNING = false

count = Ref(0)
fhistory = Float32[]

function objective_function(m_update_vec)
    count[] += 1
    m_update = reshape(m_update_vec, size(model0))
    m_update .= clamp.(m_update, mminArr, mmaxArr)

    model0.m .= Float32.(m_update)
    if MODELING_TYPE == "bulk"
        model0.rho .= Float32.(reshape(rho_from_slowness(model0.m), size(model0)))
    end

    indsrc = SHOT_FROM:SHOT_STEP:SHOT_TO
    data_precon = MUTE_REFLECTIONS ? Ml_tur[indsrc] * Ml_freq[indsrc] :
                  MUTE_TURNING ? Ml_ref[indsrc] * Ml_freq[indsrc] :
                  Ml_freq[indsrc]

    fval, gradient = fwi_objective(model0, q[indsrc], d_obs[indsrc];
                                   options=jopt,
                                   data_precon=data_precon,
                                   misfit=envelope,
                                   extended_source=EXTENDED_SOURCE,
                                   es_options=ES_OPTIONS)

    gradient = reshape(Dm * Il * gradient, size(model0))
    push!(fhistory, Float32(fval[]))

    println("iteration: ", count[], "\tfval: ", fval[], "\tnorm: ", norm(gradient))

    save_data(x, z, adjoint(reshape(model0.m.data, size(model0)));
              pltfile=joinpath(DIR_OUT, "ES-FWI slowness $(count[])") ,
              title="ES-FWI slowness^2 $(MODELING_TYPE): $(FRQ0*1000)-$(FRQ1*1000)Hz, iter $(count[])",
              colormap=cgrad(:Spectral, rev=true),
              h5file=joinpath(DIR_OUT, MODEL_FILE_OUT * " " * string(count[]) * ".h5"),
              h5openflag="w", h5varname="m")
    save_data(x, z, sqrt.(1f0 ./ adjoint(reshape(model0.m.data, size(model0))));
              pltfile=joinpath(DIR_OUT, "ES-FWI velocity $(count[])") ,
              title="ES-FWI velocity $(MODELING_TYPE): $(FRQ0*1000)-$(FRQ1*1000)Hz, iter $(count[])",
              colormap=cgrad(:Spectral, rev=true),
              h5file=joinpath(DIR_OUT, MODEL_FILE_OUT * " " * string(count[]) * ".h5"),
              h5openflag="r+", h5varname="v")
    save_data(x, z, adjoint(reshape(gradient.data, size(model0)));
              pltfile=joinpath(DIR_OUT, "ES-FWI gradient $(count[])") ,
              title="ES-FWI gradient $(MODELING_TYPE): $(FRQ0*1000)-$(FRQ1*1000)Hz, iter $(count[])",
              clim=(-maximum(abs, gradient.data)/5f0, maximum(abs, gradient.data)/5f0),
              colormap=:bluesreds,
              h5file=joinpath(DIR_OUT, MODEL_FILE_OUT * " " * string(count[]) * ".h5"),
              h5openflag="r+", h5varname="grad")
    save_fhistory(fhistory;
                  h5file=joinpath(DIR_OUT, MODEL_FILE_OUT * " " * string(count[]) * ".h5"),
                  h5openflag="r+", h5varname="fhistory")

    return fval[], gradient
end

proj(x) = reshape(median([vec(mminArr) vec(x) vec(mmaxArr)]; dims=2), size(model0))

@info "STARTED ES-FWI COMPUTATIONS"
spgopt = spg_options(verbose=3,
                     maxIter=NITERATIONS,
                     memory=3,
                     suffDec=1f-3,
                     iniStep=1f0,
                     maxLinesearchIter=12,
                     useSpectral=true,
                     feasibleInit=true)
sol = spg(objective_function, model0.m.data, proj, spgopt)

for p in workers()
    rmprocs(p)
end
