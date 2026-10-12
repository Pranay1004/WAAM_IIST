using Test, WAAMFlow

@testset "WAAMFlow" begin
    include("verify_core.jl")
    include("verify_energy.jl")
    include("verify_poisson.jl")
    include("verify_vof.jl")
end
