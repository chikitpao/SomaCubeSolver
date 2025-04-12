# soma_cube_solver.jl
# Soma cube solver
# Author: Chi-Kit Pao

using Plots; gr()

TEST_PATH = "test/"
RESULT_PATH = "result/"

function test()
    A = collect(reshape(1:27,3,3,3))

    # Output:
    # Array{Int64, 3}
    # [1 4 7; 2 5 8; 3 6 9;;; 10 13 16; 11 14 17; 12 15 18;;; 19 22 25; 20 23 26; 21 24 27]
    # [3 2 1; 6 5 4; 9 8 7;;; 12 11 10; 15 14 13; 18 17 16;;; 21 20 19; 24 23 22; 27 26 25]
    # [3 6 9; 12 15 18; 21 24 27;;; 2 5 8; 11 14 17; 20 23 26;;; 1 4 7; 10 13 16; 19 22 25]
    println(typeof(A))
    println(A)
    println(mapslices(rotr90,A,dims=[1,2]))
    println(mapslices(rotr90,A,dims=[1,3]))
    # Axis 1 -> row
    # Axis 2 -> column
    # Axis 3 -> "layer"
end

function build_polycube(raw_polycube::Matrix{Int64})::Array{Int64, 3}
    result = zeros(Int64, 3, 3, 3)
    # Output: Array{Int64, 3}
    # println(typeof(result))
    for i ∈ 0:2
        result[:, :, (i+1)] = map(x -> convert(Int64, x & (1 << i) != 0), raw_polycube)
    end
    return result
end

struct Template
    points_x::Vector{Int8}
    points_y::Vector{Int8}
    points_z::Vector{Int8}
    connections::Vector{Tuple{Int64, Int64, Int64}}  # Must be Int64
    edges_x::Vector{Int8}
    edges_y::Vector{Int8}
    edges_z::Vector{Int8}
end

function plot_cube(template::Template, posx::Int64, posy::Int64, posz::Int64, id::Int64)
    dx = posx - 1
    dy = posy - 1
    dz = posz - 1
    scatter!((0.5 + dx, 0.5 + dy, 0.5 + dz); c=:red, ms=3, msw=0.1)
    px = map(x -> x + dx, template.points_x)
    py = map(x -> x + dy, template.points_y)
    pz = map(x -> x + dz, template.points_z)
    colors = palette(:tab10)
    color = colors[id]
    mesh3d!(px,py,pz; template.connections, proj_type=:persp, fc=color, lc=color, fa=0.1, lw=0, legend=false)
    ex = map(x -> x + dx, template.edges_x)
    ey = map(x -> x + dy, template.edges_y)
    ez = map(x -> x + dz, template.edges_z)
    plot!(ex,ey,ez; lc=:black, lw=0.5, lims=(0.0,3.25))
end

function plot_polycube(template::Template, polycube::Array{Int64, 3}, id::Int64, file_name::String, stand_alone::Bool=true)
    if stand_alone
        plot()
    end
    for x ∈ 1:3
        for y ∈ 1:3
            for z ∈ 1:3
                if polycube[x, y, z] == 1
                    plot_cube(template, x, y, z, id)
                end
            end
        end
    end
    if stand_alone
        savefig(file_name)
    end
end

function plot_variations(template::Template, polycube::Array{Int64, 3}, id::Int64)
    pc = polycube
    i = id
    # Other 3 orientations of "up" position
    plot_polycube(template, mapslices(rotl90,pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_up_left.png")
    plot_polycube(template, mapslices(rot180,pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_up_180.png")
    plot_polycube(template, mapslices(rotr90,pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_up_right.png")

    # "Left" position
    left_pc = mapslices(rotl90,pc,dims=[1,3])
    plot_polycube(template, left_pc, i, TEST_PATH * "Figure$i" * "left.png")
    # Other 3 orientations
    plot_polycube(template, mapslices(rotl90,left_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_left_left.png")
    plot_polycube(template, mapslices(rot180,left_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_left_180.png")
    plot_polycube(template, mapslices(rotr90,left_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_left_right.png")

    # "Right" position
    right_pc = mapslices(rotr90,pc,dims=[1,3])
    plot_polycube(template, right_pc, i, TEST_PATH * "Figure$i" * "right.png")
    # Other 3 orientations
    plot_polycube(template, mapslices(rotl90,right_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_right_left.png")
    plot_polycube(template, mapslices(rot180,right_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_right_180.png")
    plot_polycube(template, mapslices(rotr90,right_pc,dims=[2,3]), i, TEST_PATH * "Figure$i" * "_right_right.png")

    # # "Down" position
    down_pc = mapslices(rot180,pc,dims=[1,3])
    plot_polycube(template, down_pc, i, TEST_PATH * "Figure$i" * "down.png")
    # Other 3 orientations
    plot_polycube(template, mapslices(rotl90,down_pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_down_left.png")
    plot_polycube(template, mapslices(rot180,down_pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_down_180.png")
    plot_polycube(template, mapslices(rotr90,down_pc,dims=[1,2]), i, TEST_PATH * "Figure$i" * "_down_right.png")

    # "Forward position"
    forward_pc = mapslices(rotr90,pc,dims=[2,3])
    plot_polycube(template, forward_pc, i, TEST_PATH * "Figure$i" * "_forward.png")
    # Other 3 orientations
    plot_polycube(template, mapslices(rotl90,forward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_forward_left.png")
    plot_polycube(template, mapslices(rot180,forward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_forward_180.png")
    plot_polycube(template, mapslices(rotr90,forward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_forward_right.png")

    # "Backward position"
    backward_pc = mapslices(rotl90,pc,dims=[2,3])
    plot_polycube(template, backward_pc, i, TEST_PATH * "Figure$i" * "_backward.png")
    # Other 3 orientations
    plot_polycube(template, mapslices(rotl90,backward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_backward_left.png")
    plot_polycube(template, mapslices(rot180,backward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_backward_180.png")
    plot_polycube(template, mapslices(rotr90,backward_pc,dims=[1,3]), i, TEST_PATH * "Figure$i" * "_backward_right.png")
end

function get_rotations(polycube::Array{Int64, 3})::Set{Array{Int64, 3}}
    result = Set{Array{Int64, 3}}()
    pc = polycube
    # "Up" position
    push!(result, polycube)
    # Other 3 orientations of "up" position
    push!(result, mapslices(rotl90,pc,dims=[1,2]))
    push!(result, mapslices(rot180,pc,dims=[1,2]))
    push!(result, mapslices(rotr90,pc,dims=[1,2]))

    # "Left" position
    left_pc = mapslices(rotl90,pc,dims=[1,3])
    push!(result, left_pc)
    # Other 3 orientations
    push!(result, mapslices(rotl90,left_pc,dims=[2,3]))
    push!(result, mapslices(rot180,left_pc,dims=[2,3]))
    push!(result, mapslices(rotr90,left_pc,dims=[2,3]))

    # "Right" position
    right_pc = mapslices(rotr90,pc,dims=[1,3])
    push!(result, right_pc)
    # Other 3 orientations
    push!(result, mapslices(rotl90,right_pc,dims=[2,3]))
    push!(result, mapslices(rot180,right_pc,dims=[2,3]))
    push!(result, mapslices(rotr90,right_pc,dims=[2,3]))

    # # "Down" position
    down_pc = mapslices(rot180,pc,dims=[1,3])
    push!(result, down_pc)
    # Other 3 orientations
    push!(result, mapslices(rotl90,down_pc,dims=[1,2]))
    push!(result, mapslices(rot180,down_pc,dims=[1,2]))
    push!(result, mapslices(rotr90,down_pc,dims=[1,2]))

    # "Forward position"
    forward_pc = mapslices(rotr90,pc,dims=[2,3])
    push!(result, forward_pc)
    # Other 3 orientations
    push!(result, mapslices(rotl90,forward_pc,dims=[1,3]))
    push!(result, mapslices(rot180,forward_pc,dims=[1,3]))
    push!(result, mapslices(rotr90,forward_pc,dims=[1,3]))

    # "Backward position"
    backward_pc = mapslices(rotl90,pc,dims=[2,3])
    push!(result, backward_pc)
    # Other 3 orientations
    push!(result, mapslices(rotl90,backward_pc,dims=[1,3]))
    push!(result, mapslices(rot180,backward_pc,dims=[1,3]))
    push!(result, mapslices(rotr90,backward_pc,dims=[1,3]))

    return result
end

function get_tranformations(polycube::Array{Int64, 3})::Set{Array{Int64, 3}}
    result = Set{Array{Int64, 3}}()
    rotations = get_rotations(polycube)
    for rotation ∈ rotations
        x_sums = reshape(mapslices(sum, rotation, dims=[2,3]), (1,3))
        x_back = length(collect(Iterators.takewhile(==(0), x_sums)))
        x_zero_count = count(==(0), x_sums)

        y_sums = reshape(mapslices(sum, rotation, dims=[1,3]), (1,3))
        y_back = length(collect(Iterators.takewhile(==(0), y_sums)))
        y_zero_count = count(==(0), y_sums)

        z_sums = reshape(mapslices(sum, rotation, dims=[1,2]), (1,3))
        z_back = length(collect(Iterators.takewhile(==(0), z_sums)))
        z_zero_count = count(==(0), z_sums)

        # Shift to the lowest possible position
        rotation_mod = zeros(Int64, size(rotation))
        circshift!(rotation_mod, rotation, (-x_back, -y_back, -z_back))

        # Try all possible translations
        for dx = 0:x_zero_count
            for dy = 0:y_zero_count
                for dz = 0:z_zero_count
                    push!(result, circshift(rotation_mod, (dx, dy, dz)))
                end
            end
        end
    end
    return result
end

function calculate_solutions(temp_matrix::Array{Int64, 3},
    transformations::Vector{Set{Array{Int64, 3}}})::Vector{Array{Int64, 3}}

    if isempty(transformations)
        return []
    end
    for t ∈ transformations[1]
        # Backtracking
        temp = temp_matrix + t
        if maximum(temp) > 1
            continue
        end

        if length(transformations) == 1
            return [t]
        else
            temp_result = calculate_solutions(temp, transformations[2:end])
            if !isempty(temp_result)
                return append!([t], temp_result)
            end
        end
    end
    return []
end

function main()
    with_tests::Bool = false
    
    for arg in ARGS
        if arg == "--with-tests"
            with_tests = true
        end
    end
    
    if with_tests
        test()
    end

    # Soma polycubes are described in top view.
    # Bottom -> Bit 0
    # Middle -> Bit 1
    # Top -> Bit 2
    raw_polycubes = [
        [1 1 0
         3 0 0
         1 0 0],
        [1 0 0
         1 1 0
         1 0 0],
        [1 3 0
         0 1 1
         0 0 0],
        [1 0 0
         1 3 0
         1 0 0],
        [1 3 0
         1 0 0
         0 0 0],
        [1 1 0
         0 1 0
         0 1 0],
    ]

    # Output: typeof(raw_polycubes): Vector{Matrix{Int64}}
    # println("typeof(raw_polycubes): $(typeof(raw_polycubes))")

    # Create directories for output files.
    if with_tests && !isdir(TEST_PATH) && !isfile(TEST_PATH)
        mkdir(TEST_PATH)
    end
    if !isdir(RESULT_PATH) && !isfile(RESULT_PATH)
        mkdir(RESULT_PATH)
    end

    polycubes = [build_polycube(rp) for rp ∈ raw_polycubes]
    # Output: Vector{Array{Int64, 3}}
    println("typeof(polycubes): $(typeof(polycubes))")

    template = Template(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 1],
        [(1,2,3), (4,2,3), (4,7,8), (7,5,6), (2,4,7), (1,6,2), (2,7,6), (7,8,5), (4,8,5), (4,5,3), (1,6,3), (6,3,5)],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    transformations = Vector{Set{Array{Int64, 3}}}()
    for (i, pc) ∈ enumerate(polycubes)
        if with_tests
            # "Up" position
            plot_polycube(template, pc, i, TEST_PATH * "Figure$i" * "_up.png")
            if i == 1
                plot_variations(template, pc, i)
            end
        end
        push!(transformations, get_tranformations(pc))
    end

    # Output:
    # transformation count: 600
    # [96, 72, 96, 96, 96, 144]
    println("transformation count: $(sum(map(length, transformations)))")
    println("$(map(length, transformations))")

    result = calculate_solutions(zeros(Int64, 3, 3, 3), transformations)
    println("Result:")
    println(result)
    println(sum([i * r for (i, r) ∈ enumerate(result)]))

    plot()
    for (i, pc) ∈ enumerate(result)
        plot_polycube(template, pc, i, "", false)
    end
    #gui()
    savefig(RESULT_PATH * "Result.png")
    for (i, pc) ∈ enumerate(result)
        plot_polycube(template, pc, i, RESULT_PATH * "Result$i" * ".png")
    end
end

@time main()


# Result:
# [[1 0 0; 0 0 0; 0 0 0;;; 1 0 0; 1 0 0; 0 0 0;;; 1 1 0; 0 0 0; 0 0 0], [0 1 0; 1 1 0; 0 1 0;;; 0 0 0; 0 0 0; 0 0 0;;; 0 0 0; 0 0 0; 0 0 0], [0 0 0; 0 0 0; 0 0 1;;; 0 0 0; 0 0 1; 0 1 1;;; 0 0 0; 0 0 0; 0 1 0], [0 0 0; 0 0 0; 0 0 0;;; 0 0 0; 0 1 0; 0 0 0;;; 0 0 1; 0 1 1; 0 0 1], [0 0 1; 0 0 1; 0 0 0;;; 0 1 1; 0 0 0; 0 0 0;;; 0 0 0; 0 0 0; 0 0 0], [0 0 0; 0 0 0; 1 0 0;;; 0 0 0; 0 0 0; 1 0 0;;; 0 0 0; 1 0 0; 1 0 0]]
# [1 2 5; 2 2 5; 6 2 3;;; 1 5 5; 1 4 3; 6 3 3;;; 1 1 4; 6 4 4; 6 3 4]
# 4.342548 seconds (7.89 M allocations: 435.977 MiB, 3.10% gc time, 78.25% compilation time)

# # Output of "time julia soma_cube_solver.jl":
# real	0m10,235s
# user	0m10,686s
# sys	0m0,268s
