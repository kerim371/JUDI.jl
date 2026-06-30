# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI, SegyIO, LinearAlgebra, PythonPlot, IterativeSolvers, JOLI

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 2000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Source wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(time, dt, f0)

###################################################################################################

# Write shots as segy files to disk
opt = Options()

# Setup operators
Pr = judiProjection(recGeometry)
F = judiModeling(model; options=opt)

# Random weights (size of the model)
w_tmp = zeros(Float32, model.n)
w_tmp[50,30] = 1f0
w_tmp[80,80] = 1f0
w = judiWeights(w_tmp)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(dt, wavelet)

# Model observed data w/ extended source
lambda = 1f2
I = joDirac(prod(model.n), DDT=Float32, RDT=Float32)
F = Pr*F*adjoint(Pw)
F̄ = [F; lambda*I]

# Simultaneous observed data
d_sim = F*w

# # Adjoint operation
w_adj = adjoint(F)*d_sim

# # LSQR
w_inv = 0f0 .* w
w_inv_no_damp = 0f0 .* w
lsqr!(w_inv, F̄, [d_sim; lambda*w]; maxiter=5, verbose=true, damp=1e2, atol = 1e-9, btol = 1e-9)
lsqr!(w_inv_no_damp, F, d_sim; maxiter=5, verbose=true, damp=1e2, atol = 1e-9, btol = 1e-9)

d_pred = F*w_inv;
d_pred_no_damp = F*w_inv_no_damp;

# Plot results
# Сравнение данных
fig1 = figure()
subplot(1,3,1)
imshow(d_sim.data[1], vmin=-1e1, vmax=1e1, cmap="gray"); title("Observed data")
subplot(1,3,2)
imshow(d_pred.data[1], vmin=-1e1, vmax=1e1, cmap="gray"); title("Predicted data")
subplot(1,3,3)
imshow(d_pred_no_damp.data[1], vmin=-1e1, vmax=1e1, cmap="gray"); title("Predicted data no damp")
savefig("data_comparison.png", dpi=150)
close(fig1)



# Веса и восстановления
fig2 = figure()
subplot(2,2,1)
imshow(w.weights[1], vmin=0, vmax=1, cmap="gray"); title("True Weights")
subplot(2,2,2)
imshow(w_adj.weights[1], vmin=minimum(w_adj)/10f0, vmax=maximum(w_adj)/10f0, cmap="gray"); title("Adjoint")
subplot(2,2,3)
imshow(w_inv.weights[1], vmin=minimum(w_inv)/10f0, vmax=maximum(w_inv)/10f0, cmap="gray"); title("Damped LSQR")
subplot(2,2,4)
imshow(w_inv_no_damp.weights[1], vmin=minimum(w_inv)/10f0, vmax=maximum(w_inv)/10f0, cmap="gray"); title("LSQR no damp")
savefig("weights_comparison.png", dpi=150)
close(fig2)
