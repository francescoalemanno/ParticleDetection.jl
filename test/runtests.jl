using ParticleDetection
using Test
using Serialization

@testset "Detect Particles" begin
    ima=deserialize(joinpath(@__DIR__, "particles.bin"))[1:1000,1:1000]
    @time lm=local_maxima(bp_filter(ima,5,20),0.1)
    P=[Tuple(p) for p in CartesianIndices(lm) if lm[p]>0.5]
    @test P==[(818, 309), (785, 412), (839, 446), (864, 449),
              (774, 465), (829, 487), (774, 504), (871, 510),
              (801, 523), (812, 541), (834, 541), (806, 549),
              (863, 555), (849, 563), (896, 571), (916, 574),
              (850, 576), (863, 609), (997, 675), (975, 682),
              (985, 684), (923, 687), (928, 695), (995, 711),
              (909, 783), (931, 803), (917, 805), (894, 809),
              (944, 847), (963, 854), (344, 867), (345, 873),
              (918, 874), (930, 951), (702, 971)]
end
