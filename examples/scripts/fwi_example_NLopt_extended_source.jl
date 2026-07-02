# 2D FWI on Overthrust model with L-BFGS using NLopt library
# Extended source FWI with source weight estimation via LSQR
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using Statistics, Random, LinearAlgebra, Printf
using JUDI, HDF5, NLopt, SegyIO

# Load starting model
n,d,o,m0,m_true = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model.h5","r"), "n", "d", "o", "m0", "m")

# === Задай свои скорости (км/с) ===
v_start = 1.5f0   # скорость на 22-м столбце
v_end   = 4.5f0   # скорость на последнем столбце

col_start = 21
nz = size(m0, 2)

# В каждом столбце j >= col_start задаём линейную скорость по горизонтали
for j in col_start:nz
    v_col = v_start + (v_end - v_start) * (j - col_start) / (nz - col_start)
    m0[:, j] .= 1f0 ./ (v_col^2)          # медленность² (с/км)²
end

model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./m0)
vmin = ones(Float32, model0.n) .* 1.3f0
vmax = ones(Float32, model0.n) .* 6.5f0

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read("$(JUDI.JUDI_DATA)/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0)    # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)

############################### Extended Source FWI ###########################################

# ES FWI options
    # extended_source::Bool
    # es_lambda::Float32
    # es_damp::Float32
    # es_atol::Float32
    # es_btol::Float32
    # es_conlim::Float32
    # es_maxiter::Int64
    # es_verbose::Bool
opt_es = Options(extended_source=true, 
    es_lambda=100f0, 
    es_damp=0f0,
    es_atol=1e-9,
    es_btol=1e-9,
    es_maxiter=5, 
    es_verbose=true)
# opt_es = Options(extended_source=false)

# Save initial model
@info "Saving initial model"
h5open("fwi_es_initial.h5", "w") do file
    write(file, "n", collect(n))
    write(file, "d", collect(d))
    write(file, "o", collect(o))
    write(file, "m", m0)
    write(file, "m_true", m_true)
    write(file, "v", sqrt.(1f0 ./ m0))
    write(file, "v_true", sqrt.(1f0 ./ m_true))
end

# NLopt objective function
count = 0
fhistory = Float32[]
batchsize = 16
println("No.  ", "fval         ", "norm(gradient)")
function f!(x,grad)

    # Update model
    model0.m .= convert(Array{Float32, 2}, reshape(x, model0.n))

    # Random batch of shots
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i]; options=opt_es)

    # Reset gradient in water column to zero
    gradient = reshape(gradient, model0.n)
    gradient[:, 1:21] .= 0f0
    grad[1:end] = vec(gradient)

    global count; count += 1
    global fhistory
    push!(fhistory, fval)
    println(count, "    ", fval, "    ", norm(grad))

    # Save iteration snapshot
    fname = @sprintf("fwi_es_iter_%03d.h5", count)
    h5open(fname, "w") do file
        write(file, "n", collect(n))
        write(file, "d", collect(d))
        write(file, "o", collect(o))
        write(file, "m", model0.m.data)
        write(file, "m_true", m_true)
        write(file, "v", sqrt.(1f0 ./ model0.m.data))
        write(file, "v_true", sqrt.(1f0 ./ m_true))
        write(file, "gradient", gradient)
        write(file, "fval", fval)
        write(file, "fhistory", fhistory)
    end

    return convert(Float64, fval)
end

# Optimization parameters
opt = Opt(:LD_LBFGS, prod(model0.n))
lower_bounds!(opt, mmin); upper_bounds!(opt, mmax)
min_objective!(opt, f!)
maxeval!(opt, parse(Int, get(ENV, "NITER", "10")))
(minf, minx, ret) = optimize(opt, vec(model0.m.data))

@info "Optimization finished with result: $ret, final objective: $minf"