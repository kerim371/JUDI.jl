export fwi_objective, lsrtm_objective, fwi_objective!, lsrtm_objective!

# Type of accepted input
Dtypes = Union{<:judiVector, NTuple{N, <:judiVector} where N, Vector{<:judiVector}, <:LazyMul}
MTypes = Union{<:AbstractModel, NTuple{N, <:AbstractModel} where N, Vector{<:AbstractModel}}
dmTypes = Union{dmType, NTuple{N, dmType} where N, Vector{dmType}}

function _multi_src_fg(model_full::AbstractModel, source::Dtypes, dObs::Dtypes, dm, options::JUDIOptions;
                      nlind::Bool=false, lin::Bool=false, misfit::Function=mse, illum::Bool=false,
                      data_precon=nothing, model_precon=LinearAlgebra.I)
    GC.gc(true)
    PythonCall.GC.gc()
    devito.clear_cache()

    # assert this is for single source LSRTM
    @assert source.nsrc == 1 "Multiple sources are used in a single-source fwi_objective"
    @assert dObs.nsrc == 1 "Multiple-source data is used in a single-source fwi_objective"    

    # Load full geometry for out-of-core geometry containers
    d_geometry = Geometry(dObs.geometry)
    s_geometry = Geometry(source.geometry)
    
    # If model preconditioner is provided, apply it
    dm = isnothing(dm) ? dm : model_precon * dm

    # Limit model to area with sources/receivers
    if options.limit_m == true
        @juditime "Limit model to geometry" begin
            model = deepcopy(model_full)
            model, dm = limit_model_to_receiver_area(s_geometry, d_geometry, model, options.buffer_size; pert=dm)
        end
    else
        model = model_full
    end

    # Set up Python model
    @juditime "Devito Model" begin
        modelPy = devito_model(model, options, dm)
        dtComp = pyconvert(Float32, modelPy.critical_dt)
    end

    # Extrapolate input data to computational grid
    qIn = time_resample(make_input(source), s_geometry, dtComp)
    dObserved = time_resample(make_input(dObs), d_geometry, dtComp)
    qIn, dObserved = _maybe_pad_t0(qIn, s_geometry, dObserved, d_geometry, dtComp)

    # Extended source FWI: estimate source weights via LSQR
    if options.extended_source
        @juditime "Extended source LSQR" begin
            # Use raw wavelet from source (not resampled) so that
            # devito_interface can resample it to the correct dt internally.
            wavelet_raw = make_input(source)  # (nt_src, 1) or (nt_src,)
            if ndims(wavelet_raw) == 1
                wavelet_raw = reshape(wavelet_raw, length(wavelet_raw), 1)
            end
            Pw = judiWavelet(s_geometry.dt[1], wavelet_raw)

            # Build extended source operator as in extended_source_lsqr.jl:
            #   F_ext = Pr * F(model_only) * Pw'
            Fwd = judiModeling(model; options=options)
            Pr = judiProjection(d_geometry)

            # Extended source operator: F_ext = Pr * F * Pw'
            F_ext = Pr * Fwd * adjoint(Pw)

            # Initialize weights (zeros on cropped model)
            w = judiWeights(zeros(Float32, model.n))

            # Build regularized system: [F_ext; es_lambda * I] * w = [d_obs; 0]
            I_op = joDirac(prod(model.n), DDT=Float32, RDT=Float32)
            A_ext = [F_ext; options.es_lambda * I_op]
            # Use original dObs (judiVector with correct geometry) for vcat with judiWeights
            d_obs_w = judiWeights(zeros(Float32, model.n))
            b_ext = [get_data(dObs); d_obs_w]
            println("typeof(dObs): $(typeof(dObs))")
            println("typeof(dObs[1]): $(typeof(dObs[1]))")
            println("size(dObs): $(size(dObs))")
            println("typeof(get_data(dObs)): $(typeof(get_data(dObs)))")
            println("typeof(make_input(dObs)): $(typeof(make_input(dObs)))")

            # LSQR for source weights
            lsqr!(w, A_ext, b_ext; damp=options.es_damp, atol=options.es_atol,
                  btol=options.es_btol, conlim=options.es_conlim,
                  maxiter=options.es_maxiter, verbose=options.es_verbose)

            # Extract weight array (padded for Devito)
            weights_array = vec(w.weights[1])
        end
    end

    # Set up coordinates
    @juditime "Sparse coords setup" begin
        src_coords = options.extended_source ? nothing : setup_grid(s_geometry, size(model))
        rec_coords = setup_grid(d_geometry, size(model))
    end

    # Setup misfit function
    if !isnothing(data_precon)
        new_t = StepRangeLen(0f0, Float32(dtComp), Int64(size(dObserved, 1)))
        Pcomp  = time_resample(data_precon, new_t)
        runtime_misfit = (x, y) -> misfit(Pcomp*x, Pcomp*y)
    else
        runtime_misfit = misfit
    end

    mfunc = pyjl(runtime_misfit)

    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies

    @juditime "Python call to J_adjoint" begin
        if options.extended_source
            shape = pyconvert(Tuple, modelPy.shape)
            weights_padded = pad_array(reshape(weights_array, shape), modelPy.padsizes; mode=:zeros)
            argout = wrapcall_data(ac.J_adjoint, modelPy, nothing, qIn, rec_coords,
                                    dObserved, ws=weights_padded,
                                    t_sub=options.subsampling_factor,
                                    checkpointing=options.optimal_checkpointing,
                                    freq_list=freqs, ic=options.IC, is_residual=false,
                                    born_fwd=lin, nlind=nlind,
                                    dft_sub=options.dft_subsampling_factor[1],
                                    f0=options.f0, return_obj=true,
                                    misfit=mfunc, illum=illum)
        else
            argout = wrapcall_data(ac.J_adjoint, modelPy, src_coords, qIn, rec_coords,
                                    dObserved, t_sub=options.subsampling_factor,
                                    checkpointing=options.optimal_checkpointing,
                                    freq_list=freqs, ic=options.IC, is_residual=false,
                                    born_fwd=lin, nlind=nlind,
                                    dft_sub=options.dft_subsampling_factor[1],
                                    f0=options.f0, return_obj=true,
                                    misfit=mfunc, illum=illum)
        end
    end

    @juditime "Remove padding from gradient" begin
        grad = PhysicalParameter(remove_padding(argout[2], modelPy.padsizes; true_adjoint=options.sum_padding), spacing(model), origin(model))
    end

    fval = Ref{Float32}(argout[1])
    if illum
        @juditime "Process illumination" begin
            illumu = PhysicalParameter(remove_padding(argout[3], modelPy.padsizes; true_adjoint=false), spacing(model), origin(model))
            illumv = PhysicalParameter(remove_padding(argout[4], modelPy.padsizes; true_adjoint=false), spacing(model), origin(model))
        end
        return fval, grad, illumu, illumv
    end
    return fval, grad
end

multi_src_fg = retry(_multi_src_fg)


# Find number of experiments
get_nexp(x) = 1
for T in [judiVector, AbstractModel, judiWeights, judiWavefield, PhysicalParameter, Vector{Float32}]
    @eval get_nexp(v::Vector{<:$T}) = length(v)
    @eval get_nexp(v::Tuple{N, <:$T}) where N = length(v)
end   

get_exp(x, i) = x
get_exp(x::Tuple{}, i::Any) = x[i]
for T in [judiVector, AbstractModel, judiWeights, judiWavefield, Array{Float32}, PhysicalParameter]
    @eval get_exp(v::Vector{<:$T}, i) = v[i]
    @eval get_exp(v::NTuple{N, <:$T}, i) where N = v[i]
end

function check_args(args...)
    n = [get_nexp(a) for a in args]
    nexp = maximum(n)
    check = all(ni -> (ni==nexp || ni==1), n)
    check || throw(ArgumentError("Incompatible number of experiements"))
    return nexp
end

################################################################################################
####################### User Interface #########################################################
################################################################################################

function fwi_objective(model::MTypes, q::Dtypes, dobs::Dtypes; options=Options(), kw...)
    n_exp = check_args(model, q, dobs)
    G = n_exp == 1 ? similar(model.m, model) : [similar(get_exp(model, i).m, get_exp(model, i)) for i=1:n_exp]
    f = fwi_objective!(G, model, q, dobs; options=options, kw...)
    f, G
end

function lsrtm_objective(model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmTypes; options=Options(), nlind=false, kw...)
    n_exp = check_args(model, q, dobs, dm)
    G = n_exp == 1 ? similar(model.m, model) : [similar(get_exp(model, i).m, get_exp(model, i)) for i=1:n_exp]
    f = lsrtm_objective!(G, model, q, dobs, dm; options=options, nlind=nlind, kw...)
    f, G
end

function fwi_objective!(G, model::MTypes, q::Dtypes, dobs::Dtypes; options=Options(), kw...)
    n_exp = check_args(G, model, dobs, q)
    return multi_exp_fg!(Val(n_exp), G, model, q, dobs, nothing; options=options, nlind=false, lin=false, kw...)
end

function lsrtm_objective!(G, model::MTypes, q::Dtypes, dobs::Dtypes, dm::dmTypes; options=Options(), nlind=false, kw...)
    n_exp = check_args(G, model, q, dobs, dm)
    return multi_exp_fg!(Val(n_exp), G, model, q, dobs, dm; options=options, nlind=nlind, lin=true, kw...)
end

multi_exp_fg!(n::Val{1}, ar...; kw...) = multi_src_fg!(ar...; kw...)

function multi_exp_fg!(n::Val{N}, ar...; kw...) where N
    f = zeros(Float32, N)
    @sync for i=1:N
        ai = (get_exp(a, i) for a in ar)
        @async f[i] = multi_src_fg!(ai...; kw...)
    end
    sum(f)
end