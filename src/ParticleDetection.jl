module ParticleDetection

end # module

#%%


struct KernelOp{T,N,S} <: AbstractArray{T,N}
    parent::S
    op::Function
    opdims::CartesianIndex{N}
    lowbound::CartesianIndex{N}
    upbound::CartesianIndex{N}
    @inline function KernelOp{T,N,S}(A::S,operation::Function,dims::CartesianIndex{N}) where {T,N,S <: AbstractArray{T,N}}
        l=minimum(CartesianIndices(A))
        h=maximum(CartesianIndices(A))
        new(A,operation,dims,l,h)
    end
end
@inline function KernelOp(A::AbstractArray{T,N},op::Function,dims::NTuple{N,Int}) where {T,N}
    KernelOp{eltype(A),ndims(A),typeof(A)}(A,op,CartesianIndex(dims))
end
import Base.size, Base.getindex, Base.IndexStyle

@inline IndexStyle(::KernelOp) = IndexCartesian()
@inline function size_parent(A::KernelOp)
    size(A.parent)
end
@inline function size(A::KernelOp)
    return size_parent(A)
end
@inline function getindex(A::KernelOp{T,N,S},Index::Vararg{Int,N}) where {T,N,S}
    I=CartesianIndex(Index)
    @boundscheck checkbounds(A,I)
    @inbounds begin
        lowI=max(I-A.opdims,A.lowbound)
        upI=min(I+A.opdims,A.upbound)
        return A.op(A.parent,lowI:upI,I)
    end
end

@inline function op_mean(A,r,c)
    m=zero(eltype(A))
    @inbounds begin
        @simd for i in r
            m+=A[i]
        end
    end
    m/length(r)
end
@inline function op_max(A,r,c)
    m=A[r[1]]
    @inbounds begin
        @simd for i in r
            m=max(m,A[i])
        end
    end
    m
end


M=rand(1000,1000)
f1=KernelOp(M,op_mean,(1,1))
f2=KernelOp(f1,op_max,(2,2))

@time maximum(f2)


using Serialization

using PyPlot

ima=deserialize("test/particles.bin")

pygui(true)

imshow(ima)
scatter(1:10,1:10)
