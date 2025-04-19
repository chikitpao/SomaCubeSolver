# soma_cube_solver.py
# Soma cube solver
# Author: Chi-Kit Pao

from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


TEST_PATH = "test/"
RESULT_PATH = "result/"

def test():
    A = np.arange(27).reshape((3, 3, 3))

    # Output:

    # <class 'numpy.ndarray'>

    # [[[ 0  1  2]
    #   [ 3  4  5]
    #   [ 6  7  8]]
    #
    #  [[ 9 10 11]
    #   [12 13 14]
    #   [15 16 17]]
    #
    #  [[18 19 20]
    #   [21 22 23]
    #   [24 25 26]]]

    #  [[[ 6  3  0]
    #   [ 7  4  1]
    #   [ 8  5  2]]
    #
    #  [[15 12  9]
    #   [16 13 10]
    #   [17 14 11]]
    #
    #  [[24 21 18]
    #   [25 22 19]
    #   [26 23 20]]]

    # [[[18  9  0]
    #   [21 12  3]
    #   [24 15  6]]
    #
    #  [[19 10  1]
    #   [22 13  4]
    #   [25 16  7]]
    #
    #  [[20 11  2]
    #   [23 14  5]
    #   [26 17  8]]]

    print(type(A))
    print(A)
    print(np.rot90(A, k=-1, axes=(1,2)))  # k = -1 => clockwise
    print(np.rot90(A, k=-1, axes=(0,2)))  # k = -1 => clockwise
    # Axis 0 -> "layer"
    # Axis 1 -> row
    # Axis 2 -> column

def build_polycube(raw_polycube:list) -> np.ndarray:
    result = np.zeros((3, 3, 3), int)
    # Output: type(result)=<class 'numpy.ndarray'>
    # print(f"{type(result)=}")
    for row, line in enumerate(raw_polycube):
        for col, v in enumerate(line):
            for layer in range(3):
                result[layer, row, col] = int((v & (1 << layer)) != 0)
    return result

def plot_cube(posx:int, posy:int, posz:int, id:int, ax):
    # Output: type(ax)=<class 'matplotlib.axes._subplots.Axes3DSubplot'>
    # print(f"{type(ax)=}")

    r = [0,1]
    X, Y = np.meshgrid(r, r)
    one = np.array([[1, 1]])
    X_ = X+posx
    Y_ = Y+posy
    Z1_ = X+posz
    Z2_ = Y+posz
    alpha_ = 0.1
    wire_color = 'k'
    linewidth_ = 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    color_ = colors[id]

    # bottom and top
    ax.plot_surface(X_, Y_, one*posz, alpha=alpha_, color=color_)
    ax.plot_wireframe(X_, Y_, one*posz, color=wire_color, linewidth=linewidth_)
    ax.plot_surface(X_, Y_, one*(posz+1), alpha=alpha_, color=color_)
    ax.plot_wireframe(X_, Y_, one*(posz+1), color=wire_color, linewidth=linewidth_)
    # left and right
    ax.plot_surface(one*posx, Y_, Z1_, alpha=alpha_, color=color_)
    ax.plot_wireframe(one*posx, Y_, Z1_, color=wire_color, linewidth=linewidth_)
    ax.plot_surface(one*(posx+1), Y_, Z1_, alpha=alpha_, color=color_)
    ax.plot_wireframe(one*(posx+1), Y_, Z1_, color=wire_color, linewidth=linewidth_)
    # front and back
    ax.plot_surface(X_, one*posy, Z2_, alpha=alpha_, color=color_)
    ax.plot_wireframe(X_, one*posy, Z2_, color=wire_color, linewidth=linewidth_)
    ax.plot_surface(X_, one*(posy+1), Z2_, alpha=alpha_, color=color_)
    ax.plot_wireframe(X_, one*(posy+1), Z2_, color=wire_color, linewidth=linewidth_)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def add_subplot(fig):
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim3d([0.0, 3.25])
    ax.set_ylim3d([0.0, 3.25])
    ax.set_zlim3d([0.0, 3.25])
    return ax

def plot_polycube(polycube:np.ndarray, id:int, file_name:str, ax=None):
    stand_alone = not ax
    if stand_alone:
        fig = plt.figure()
        ax = add_subplot(fig)

    for x in range(3):
        for y in range(3):
            for z in range(3):
                if polycube[z, x, y] == 1:
                    plot_cube(x, y, z, id, ax)

    if stand_alone:
        plt.savefig(file_name)
        plt.close(fig)

def plot_variations(polycube:np.ndarray, id:int):
    pc = polycube
    i = id
    # Other 3 orientations of "up" position

    plot_polycube(np.rot90(pc, k=1, axes=(0,1)), i, TEST_PATH + f"Figure{i}_up_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(0,1)), i, TEST_PATH + f"Figure{i}_up_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(0,1)), i, TEST_PATH + f"Figure{i}_up_right.png")

    # "Left" position
    left_pc = np.rot90(pc, k=1, axes=(0,2))
    plot_polycube(left_pc, i, TEST_PATH + f"Figure{i}left.png")
    # Other 3 orientations
    plot_polycube(np.rot90(pc, k=1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_left_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(1,2)), i, TEST_PATH + f"Figure{i}_left_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_left_right.png")

    # "Right" position
    right_pc = np.rot90(pc, k=-1, axes=(0,2))
    plot_polycube(right_pc, i, TEST_PATH + f"Figure{i}right.png")
    # Other 3 orientations
    plot_polycube(np.rot90(pc, k=1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_right_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(1,2)), i, TEST_PATH + f"Figure{i}_right_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_right_right.png")

    # # "Down" position
    down_pc = np.rot90(pc, k=2, axes=(0,2))
    plot_polycube(down_pc, i, TEST_PATH + f"Figure{i}down.png")
    # Other 3 orientations
    plot_polycube(np.rot90(pc, k=1, axes=(0,1)), i, TEST_PATH + f"Figure{i}_down_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(0,1)), i, TEST_PATH + f"Figure{i}_down_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(0,1)), i, TEST_PATH + f"Figure{i}_down_right.png")

    # "Forward position"
    forward_pc = np.rot90(pc, k=-1, axes=(1,2))
    plot_polycube(forward_pc, i, TEST_PATH + f"Figure{i}_forward.png")
    # Other 3 orientations
    plot_polycube(np.rot90(pc, k=1, axes=(0,2)), i, TEST_PATH + f"Figure{i}_forward_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(0,2)), i, TEST_PATH + f"Figure{i}_forward_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(0,2)), i, TEST_PATH + f"Figure{i}_forward_right.png")

    # "Backward position"
    backward_pc = np.rot90(pc, k=1, axes=(1,2))
    plot_polycube(backward_pc, i, TEST_PATH + f"Figure{i}_backward.png")
    # Other 3 orientations
    plot_polycube(np.rot90(pc, k=1, axes=(0,2)), i, TEST_PATH + f"Figure{i}_backward_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(0,2)), i, TEST_PATH + f"Figure{i}_backward_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(0,2)), i, TEST_PATH + f"Figure{i}_backward_right.png")

def get_rotations(polycube:np.ndarray) -> list[np.ndarray]:
    # REMARK: 'numpy.ndarray' is not hashable, so put it into a
    # set will cause error:
    # TypeError: unhashable type: 'numpy.ndarray'
    result = list()
    pc = polycube
    # "Up" position
    result.append(polycube)
    # Other 3 orientations of "up" position
    result.append(np.rot90(pc, k=1, axes=(0,1)))
    result.append(np.rot90(pc, k=2, axes=(0,1)))
    result.append(np.rot90(pc, k=-1, axes=(0,1)))

    # "Left" position
    left_pc = np.rot90(pc, k=1, axes=(0,2))
    result.append(left_pc)
    # Other 3 orientations
    result.append(np.rot90(left_pc, k=1, axes=(1,2)))
    result.append(np.rot90(left_pc, k=2, axes=(1,2)))
    result.append(np.rot90(left_pc, k=-1, axes=(1,2)))

    # "Right" position
    right_pc = np.rot90(pc, k=-1, axes=(0,2))
    result.append(right_pc)
    # Other 3 orientations
    result.append(np.rot90(right_pc, k=1, axes=(1,2)))
    result.append(np.rot90(right_pc, k=2, axes=(1,2)))
    result.append(np.rot90(right_pc, k=-1, axes=(1,2)))

    # # "Down" position
    down_pc = np.rot90(pc, k=2, axes=(0,2))
    result.append(down_pc)
    # Other 3 orientations
    result.append(np.rot90(down_pc, k=1, axes=(0,1)))
    result.append(np.rot90(down_pc, k=2, axes=(0,1)))
    result.append(np.rot90(down_pc, k=-1, axes=(0,1)))

    # "Forward position"
    forward_pc = np.rot90(pc, k=-1, axes=(1,2))
    result.append(forward_pc)
    # Other 3 orientations
    result.append(np.rot90(forward_pc, k=1, axes=(0,2)))
    result.append(np.rot90(forward_pc, k=2, axes=(0,2)))
    result.append(np.rot90(forward_pc, k=-1, axes=(0,2)))

    # "Backward position"
    backward_pc = np.rot90(pc, k=1, axes=(1,2))
    result.append(backward_pc)
    # Other 3 orientations
    result.append(np.rot90(backward_pc, k=1, axes=(0,2)))
    result.append(np.rot90(backward_pc, k=2, axes=(0,2)))
    result.append(np.rot90(backward_pc, k=-1, axes=(0,2)))

    return result

def get_tranformations(polycube:np.ndarray) -> list[np.ndarray]:
    # REMARK: 'numpy.ndarray' is not hashable, so put it into a
    # set will cause error:
    # TypeError: unhashable type: 'numpy.ndarray'
    temp_result = []
    rotations = get_rotations(polycube)
    for rotation in rotations:
        x_sums = [rotation[:, i, :].sum() for i in range(3)]
        x_back = len(list(itertools.takewhile(lambda x: x == 0, x_sums)))
        x_zero_count = len(list(filter(lambda x: x == 0, x_sums)))

        y_sums = [rotation[:, :, i].sum() for i in range(3)]
        y_back = len(list(itertools.takewhile(lambda x: x == 0, y_sums)))
        y_zero_count = len(list(filter(lambda x: x == 0, y_sums)))

        z_sums = [rotation[i, :, :].sum() for i in range(3)]
        z_back = len(list(itertools.takewhile(lambda x: x == 0, z_sums)))
        z_zero_count = len(list(filter(lambda x: x == 0, z_sums)))

        # Shift to the lowest possible position
        rotation_mod = np.roll(rotation, (-z_back, -x_back, -y_back), (0, 1, 2))

        # Try all possible translations
        # REMARK: Translations will not cause duplicates here.
        for dx in range(x_zero_count+1):
            for dy in range(y_zero_count+1):
                for dz in range(z_zero_count+1):
                    temp_result.append(np.roll(rotation_mod, (dz, dx, dy), (0, 1, 2)))

    # Remove duplicates
    result = []
    for candidate in temp_result:
        if all(not np.array_equal(M, candidate) for M in result):
            result.append(candidate)

    return result

def calculate_solutions(all_solutions:bool, temp_matrix:np.ndarray, temp_result:list[np.ndarray],
    transformations:list[list[np.ndarray]], results:list[list[np.ndarray]]):
    if not all_solutions and results:
        return
    
    if not transformations:
        return

    for t in transformations[0]:
        # Backtracking
        temp = temp_matrix + t
        if np.max(temp) > 1:
            continue

        temp_result.append(t)
        if len(transformations) == 1:
            results.append(deepcopy(temp_result))
            if not all_solutions:
                return
        else:
            calculate_solutions(all_solutions, temp, temp_result, transformations[1:], results)
        
        temp -= t
        temp_result.pop()

def calculate_result_groups(results):
    result_groups = []
    for result in results:
        result2 = sum([i * r for i, r in enumerate(result)])
        if not result_groups:
            result_groups.append(result2)
        else:
            result_transformations = get_tranformations(result2)
            is_duplicate = False
            for t in result_transformations:
                for test_element in result_groups:
                    if np.array_equal(t, test_element):
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if not is_duplicate:
                result_groups.append(result_transformations[0])
    return result_groups

def main():
    start_time = time.time()

    with_tests = False
    all_solutions = False
    normal_cube = False

    for arg in sys.argv[1:]:
        if arg == "--with-tests":
            with_tests = True
        elif arg == "--all-solutions":
            all_solutions = True
        elif arg == "--normal-cube":
            normal_cube = True

    if with_tests:
        test()

    # Soma polycubes are described in top view.
    # Bottom -> Bit 0
    # Middle -> Bit 1
    # Top -> Bit 2
    raw_polycubes_custom = [
        [[1, 1, 0],
         [3, 0, 0],
         [1, 0, 0]],
        [[1, 0, 0],
         [1, 1, 0],
         [1, 0, 0]],
        [[1, 3, 0],
         [0, 1, 1],
         [0, 0, 0]],
        [[1, 0, 0],
         [1, 3, 0],
         [1, 0, 0]],
        [[1, 3, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 0]]
    ]
    # => transformation count: 600
    # => [96, 72, 96, 96, 96, 144]
    # => Result count: 24
    # => Distinct result count: 1
    # => Time elapsed: 21.532670736312866 s

    raw_polycubes_normal = [
        [[3, 1, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        [[1, 3, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[3, 1, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 1, 0],
         [0, 0, 0]],
    ]
    # => transformation count: 688
    # => [64, 144, 96, 96, 72, 144, 72]
    # => Result count: 11520
    # => Distinct result count: 480
    # => Time elapsed: 1846.049610376358 s

    raw_polycubes = raw_polycubes_normal if normal_cube else raw_polycubes_custom

    # # Create directories for output files.
    if with_tests and not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # Output: type(raw_polycubes)=<class 'list'>
    # print(f"{type(raw_polycubes)=}")

    polycubes = [build_polycube(rp) for rp in raw_polycubes]
    # Output: type(polycubes)=<class 'list'>
    # print(f"{type(polycubes)=}")

    transformations = []
    for i, pc in enumerate(polycubes):
        if with_tests:
            # "Up" position
            plot_polycube(pc, i, TEST_PATH + f"Figure{i}_up.png")
            if i == 0:
                plot_variations(pc, i)
        transformations.append(get_tranformations(pc))

    # Output:
    # transformation count: 600
    # [96, 72, 96, 96, 96, 144]
    print(f"transformation count: {sum(map(len, transformations))}")
    print(f"{list(map(len, transformations))}")

    results = []
    calculate_solutions(all_solutions, np.zeros((3, 3, 3), int), [], transformations, results)
    first_result = results[0]
    print("Result:")
    print(first_result)
    print(sum([i * r for i, r in enumerate(first_result)]))
    
    if all_solutions:
        print(f"Result count: {len(results)}")
        result_groups = calculate_result_groups(results)
        print(f"Distinct result count: {len(result_groups)}")

    fig = plt.figure()
    ax = add_subplot(fig)
    for i, pc in enumerate(first_result):
         plot_polycube(pc, i, "", ax)
    plt.savefig(RESULT_PATH + "Result.png")
    plt.close(fig)
    for i, pc in enumerate(first_result):
         plot_polycube(pc, i, RESULT_PATH + f"Result{i}.png")

    print(f"Time elapsed: {time.time() - start_time} s")


if __name__ == "__main__":
    main()


# Result:
# [array([[[1, 1, 0],
#         [1, 0, 0],
#         [1, 0, 0]],

#        [[0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 1, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 0]],

#        [[0, 1, 0],
#         [0, 1, 1],
#         [0, 0, 1]]]), array([[[0, 0, 1],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 1, 1],
#         [0, 1, 0],
#         [0, 0, 0]],

#        [[0, 0, 1],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 1, 1],
#         [0, 0, 1]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 1]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 0, 0],
#         [1, 0, 0],
#         [1, 0, 0]]])]
# [[[0 0 3]
#   [0 4 4]
#   [0 1 4]]

#  [[5 3 3]
#   [0 3 2]
#   [1 1 4]]

#  [[5 2 3]
#   [5 2 2]
#   [5 1 2]]]

# Output of "time python3 soma_cube_solver.py":
# real	0m1,897s
# user	0m2,977s
# sys	0m2,549s
