
@inline function tnorm2(x::NTuple{N,T},type::Type{T}) where {N,T}
    x[1]*x[1]+tnorm2(Base.tail(x),T)
end

@inline function tnorm2(x::NTuple{0},type::Type{T}) where {T}
    zero(T)
end

@inline function tnorm2(x::NTuple{N,T}) where {N,T}
    tnorm2(x,T)
end

function tnorm2(x)
    tnorm2(promote(x...))
end

function select_seeds(ima,P)
    S=size(ima)
    l=@. S>>1
    h=@. (S+l)>>1
    seeds=Tuple{ntuple(i->Int,length(S))...}[]
    for p in P
        all(l.<=p.<=h) || continue;
        push!(seeds,p)
    end
    seeds
end

function integrate_disk(ima,P,r::Int)
    r_half=r>>1
    op_size=ntuple(i->(r+r_half),ndims(ima))
    kop=KernelOp(ima,op_size) do A,Is,ci_I
        Iim=zero(eltype(A))
        I=Tuple(ci_I)
        N=0
        for ci_Itrasl in Is
            Itrasl=Tuple(ci_Itrasl)
            dist=tnorm2(Itrasl.-I)
            if r*r < dist< 2*r*r
                Iim += A[ci_Itrasl]
                N+=1
            end
        end
        Iim
    end
    [kop[p...] for p in P]
end

function scan_radiuses(ima,P,rs::Int)
    min_r=max(rs>>2,1)
    max_r=(rs<<2)+min_r
    T=eltype(ima)
    r_range=min_r:max_r
    sums=zeros(T,length(r_range))
    i=1
    for r in r_range
        sums[i] = sum(integrate_disk(ima,P,r))
        i+=1
    end
    r_range,sums/length(P)
end

function kern_find_best_radius(ima,P,rs::Int)
    r_range,intensity=scan_radiuses(ima,P,rs)
    r_pos=rs-minimum(r_range)+1
    Δ=intensity[r_pos]*0.2 # imagine an error of 20% on the intensity
    Ns=similar(intensity)
    for i in eachindex(intensity)
        I=intensity[i]
        N=sum(@. I-Δ <= intensity <= I+Δ )
        Ns[i]=N
    end
    Nmax_greedy=maximum(Ns)
    mask=(Ns).>=(Nmax_greedy-1) # there is a noise in Ns hence the -1
    best_radiuses=r_range[mask]
    round(Int,sum(best_radiuses)/length(best_radiuses))
end

function find_best_radius(ima,P,rs::Int)
    r=rs
    for i in 1:10
        nr=kern_find_best_radius(ima,P,r)
        nr==r && break
        r=nr
    end
    r
end
