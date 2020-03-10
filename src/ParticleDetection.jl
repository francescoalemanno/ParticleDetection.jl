module ParticleDetection
    using KernelOps
    @inline function op_mean(A,Is,I)
        m=zero(eltype(A))
        @simd for i in Is
            @inbounds m+=A[i]
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
        @inbounds m=A[I]
        @simd for i in Is
            @inbounds m=max(m,A[i])
        end
        m
    end

    @inline function pow2(x)
        x*x
    end

    export bp_filter

    function bp_filter(image,sn,so)
        T=eltype(image)
        G=op_gaussian(image,sn)
        Z=zero(T)
        fobject=KernelOp(op_mean,image,(so,so))
        fnoise=KernelOp(G,image,(sn,sn))
        pre=similar(T[],size(image))
        Threads.@threads for i in CartesianIndices(pre)
            @inbounds pre[i]=max(fnoise[i]-fobject[i],Z)
        end
        pre
    end

    export local_maxima
    
    function local_maxima(image,abs_th=zero(eltype(image)))
        abs_th=convert(eltype(image),abs_th)
        max_im=KernelOp(op_max,image,(1,1))
        mask=falses(size(image)...)
        Threads.@threads for i in CartesianIndices(mask)
            @inbounds a=max_im[i]
            @inbounds b=image[i]
            @fastmath c=((a == b) && (b > abs_th))
            mask[i]=c
        end
        mask
    end
end # module
