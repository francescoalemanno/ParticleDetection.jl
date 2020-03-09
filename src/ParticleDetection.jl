module ParticleDetection
    using KernelOps
    @inline function op_mean(A,Is,I)
        m=zero(eltype(A))
        @inbounds @simd for i in Is
            m+=A[i]
        end
        m/length(Is)
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
        fnoise=KernelOp(image,(sn,sn)) do A, Is, I
            ws=zero(eltype(A))
            s=ws
            @inbounds for i in Is
                deltaX=Tuple(i-I)
                delta=sum(pow2,deltaX)/pow2(sn)
                w=exp(-2delta)
                s+=A[i]*w
                ws+=w
            end
            s/ws
        end
        bpass=KernelOp(collect(fnoise),(0,0)) do A,Is,I
            @inbounds max(A[I]-fobject[I],zero(eltype(image)))
        end
        bpass
    end
    export local_maxima
    function local_maxima(image,abs_th=zero(eltype(image)))
        abs_th=convert(eltype(image),abs_th)
        ONE=one(eltype(image))
        ZERO=zero(eltype(image))
        KernelOp(collect(image),(1,1)) do A,Is,I
            @inbounds om=m=A[I]
            @inbounds @simd for i in Is
                m=max(m,A[i])
            end
            (m==om && m>abs_th) ? ONE : ZERO
        end
    end


end # module

#%%
