module ParticleDetection
    using KernelOps
    @inline function op_mean(A,Is,I)
        m=zero(eltype(A))
        @inbounds for i in Is
            m+=A[i]
        end
        m/length(Is)
    end

    struct op_gaussian{T,N} <: Function
        sn::Int
        w::Array{T,N}
        function op_gaussian(A,sn)
            w=similar(eltype(A)[],ntuple(i->sn*2+1,ndims(A)))
            for I in CartesianIndices(w)
                deltaX=Tuple(I).-sn.-1
                delta=sum(pow2,deltaX)/(sn*sn)
                w[I]=exp(-2*delta)
            end
            new{eltype(A),ndims(A)}(sn,w)
        end
    end
    @inline function (p::op_gaussian)(A::AbstractArray{T}, Is, I) where T
        ws=zero(T)
        s=ws
        rc=CartesianIndex(ntuple(i->p.sn+1,ndims(A)))
        @inbounds for i in Is
            ni=i-I+rc
            s+=A[i]*p.w[ni]
            ws+=p.w[ni]
        end
        s/ws
    end

    @inline function op_max(A,Is,I)
        m=A[I]
        @inbounds @simd for i in Is
            m=max(m,A[i])
        end
        m
    end

    @inline function pow2(x)
        x*x
    end
    export bp_filter

    function bp_filter(image,sn,so)
        fobject=KernelOp(op_mean,image,(so,so))
        fnoise=KernelOp(op_gaussian(image,sn),image,(sn,sn))
        Z=zero(eltype(image))
        @. max(fnoise-fobject,Z)
    end
    export local_maxima
    function local_maxima(image,abs_th=zero(eltype(image)))
        abs_th=convert(eltype(image),abs_th)
        ONE=one(eltype(image))
        ZERO=zero(eltype(image))
        max_im=KernelOp(op_max,image,(1,1))
        (max_im .== image) .& (image .> abs_th)
    end


end # module

#%%
