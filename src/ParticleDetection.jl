module ParticleDetection
    using KernelOps
    @inline function op_mean(A,Is,I)
        m=zero(eltype(A))
        @inbounds @simd for i in Is
            m+=A[i]
        end
        m/length(Is)
    end

    struct op_gaussian <: Function
        sn::Int
    end
    @inline function (p::op_gaussian)(A::AbstractArray{T}, Is, I) where T
        ws=zero(T)
        s=ws
        K1=convert(T,pow2(p.sn)::Int)::T
        @inbounds for i in Is
            deltaX=convert.(T,Tuple(i-I))
            delta=sum(pow2,deltaX)::T/K1::T
            w=exp(-T(2)*delta::T)
            s+=A[i]*w
            ws+=w
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
        fnoise=KernelOp(op_gaussian(sn),image,(sn,sn))
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
