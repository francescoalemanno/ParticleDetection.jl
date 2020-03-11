module ParticleDetection
    using KernelOps
    @inline function op_mean(A,Is,I)
        m=zero(eltype(A))
        @simd for i in Is
            @inbounds m+=A[i]
        end
        m/length(Is)
    end
    @inline function rep_tuple(val,A::AbstractArray{T,N}) where {T,N}
        ntuple(N) do x
            val
        end
    end
    struct op_gaussian{T,N} <: Function
        sn::Int
        w::Array{T,N}
        function op_gaussian(A,sn)
            w=similar(eltype(A)[],rep_tuple(sn*2+1,A))
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
        rc=CartesianIndex(rep_tuple(p.sn+1,A))
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

    @inline function bp_filter(image,sn,so)
        T=eltype(image)
        G=op_gaussian(image,sn)
        Z=zero(T)
        fobject=KernelOp(op_mean,image,rep_tuple(so,image))
        fnoise=KernelOp(G,image,rep_tuple(sn,image))
        pre=similar(T[],size(image))
        Threads.@threads for i in CartesianIndices(pre)
            @inbounds pre[i]=max(fnoise[i]-fobject[i],Z)
        end
        pre
    end

    @inline function local_maxima(image,abs_th=zero(eltype(image)))
        abs_th=convert(eltype(image),abs_th)
        max_im=KernelOp(op_max,image,rep_tuple(1,image))
        mask=falses(size(image)...)
        Threads.@threads for i in CartesianIndices(mask)
            @inbounds a=max_im[i]
            @inbounds b=image[i]
            @fastmath c=((a == b) && (b > abs_th))
            mask[i]=c
        end
        mask
    end
    export detect_particles
"""
detect_particles(image, noise_scale, object_scale, abs_threshold)
"""
    function detect_particles(image::AbstractArray{T},noise_scale::Int,object_scale::Int,abs_threshold::T) where T
        lm=local_maxima(bp_filter(image,noise_scale,object_scale),abs_threshold)
        [Tuple(p) for p in CartesianIndices(lm) if lm[p]]
    end
end # module
