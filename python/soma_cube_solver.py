# soma_cube_solver.py
# Soma cube solver
# Author: Chi-Kit Pao

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
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
    # Axis 1 to 2 is counter-clockwise to axis 0, so use k = -1 for clockwise rotation.
    print(np.rot90(A, k=-1, axes=(1,2)))
    # Axis 0 to 2 is clockwise to axis 1, so use k = -1 for counter-clockwise rotation.
    print(np.rot90(A, k=-1, axes=(0,2)))
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

    r = [0,1]
    alpha_ = 0.1 if stand_alone else 0.3
    wire_color = 'k'
    linewidth_ = 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'pink', 'w']
    color_ = colors[id]

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for posx, posy, posz in itertools.product(list(range(3)), repeat=3):
        if polycube[posz, posx, posy] != 1:
            continue

        X, Y = np.meshgrid(r, r)
        one = np.array([[1, 1]])
        X_ = X+posx
        Y_ = Y+posy
        Z1_ = X+posz
        Z2_ = Y+posz

        # REMARK: Don't draw inner surfaces.
        # bottom and top
        if posz == 0 or polycube[posz-1][posx][posy] != 1:
            ax.plot_surface(X_, Y_, one*posz, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, Y_, one*posz, color=wire_color, linewidth=linewidth_)
        if posz == 2 or polycube[posz+1][posx][posy] != 1:
            ax.plot_surface(X_, Y_, one*(posz+1), alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, Y_, one*(posz+1), color=wire_color, linewidth=linewidth_)
        # left and right
        if posx == 0 or polycube[posz][posx-1][posy] != 1:
            ax.plot_surface(one*posx, Y_, Z1_, alpha=alpha_, color=color_)
        ax.plot_wireframe(one*posx, Y_, Z1_, color=wire_color, linewidth=linewidth_)
        if posx == 2 or polycube[posz][posx+1][posy] != 1:
            ax.plot_surface(one*(posx+1), Y_, Z1_, alpha=alpha_, color=color_)
        ax.plot_wireframe(one*(posx+1), Y_, Z1_, color=wire_color, linewidth=linewidth_)
        # front and back
        if posy == 0 or polycube[posz][posx][posy-1] != 1:
            ax.plot_surface(X_, one*posy, Z2_, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, one*posy, Z2_, color=wire_color, linewidth=linewidth_)
        if posy == 2 or polycube[posz][posx][posy+1] != 1:
            ax.plot_surface(X_, one*(posy+1), Z2_, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, one*(posy+1), Z2_, color=wire_color, linewidth=linewidth_)


    if stand_alone:
        plt.savefig(file_name)
        plt.close(fig)

def plot_variations(polycube:np.ndarray, id:int):
    pc = polycube
    i = id
    # Other 3 orientations of "up" position: Rotate around z-axis.
    plot_polycube(np.rot90(pc, k=1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_up_left.png")
    plot_polycube(np.rot90(pc, k=2, axes=(1,2)), i, TEST_PATH + f"Figure{i}_up_180.png")
    plot_polycube(np.rot90(pc, k=-1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_up_right.png")

    # "Left" position: Rotate left around y-axis. Original up-axis (= z-axis) has rotated to (negative) x-axis.
    left_pc = np.rot90(pc, k=1, axes=(1,0))
    plot_polycube(left_pc, i, TEST_PATH + f"Figure{i}_left.png")
    # Other 3 orientations: Rotate around x-axis.
    plot_polycube(np.rot90(left_pc, k=1, axes=(2,0)), i, TEST_PATH + f"Figure{i}_left_left.png")
    plot_polycube(np.rot90(left_pc, k=2, axes=(2,0)), i, TEST_PATH + f"Figure{i}_left_180.png")
    plot_polycube(np.rot90(left_pc, k=-1, axes=(2,0)), i, TEST_PATH + f"Figure{i}_left_right.png")

    # "Right" position: Rotate right around y-axis. Original up-axis (= z-axis) has rotated to x-axis.
    right_pc = np.rot90(pc, k=-1, axes=(1,0))
    plot_polycube(right_pc, i, TEST_PATH + f"Figure{i}_right.png")
    # Other 3 orientations: Rotate around x-axis.
    plot_polycube(np.rot90(right_pc, k=1, axes=(2,0)), i, TEST_PATH + f"Figure{i}_right_left.png")
    plot_polycube(np.rot90(right_pc, k=2, axes=(2,0)), i, TEST_PATH + f"Figure{i}_right_180.png")
    plot_polycube(np.rot90(right_pc, k=-1, axes=(2,0)), i, TEST_PATH + f"Figure{i}_right_right.png")

    # "Down" position: Rotate 180° around y-axis. Original up-axis (= z-axis) has rotated to (negative) z-axis.
    down_pc = np.rot90(pc, k=2, axes=(1,0))
    plot_polycube(down_pc, i, TEST_PATH + f"Figure{i}_down.png")
    # Other 3 orientations: Rotate around z-axis.
    plot_polycube(np.rot90(down_pc, k=1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_down_left.png")
    plot_polycube(np.rot90(down_pc, k=2, axes=(1,2)), i, TEST_PATH + f"Figure{i}_down_180.png")
    plot_polycube(np.rot90(down_pc, k=-1, axes=(1,2)), i, TEST_PATH + f"Figure{i}_down_right.png")

    # "Forward position": Rotate right around x-axis. Original up-axis (= z-axis) has rotated to y-axis.
    forward_pc = np.rot90(pc, k=-1, axes=(2,0))
    plot_polycube(forward_pc, i, TEST_PATH + f"Figure{i}_forward.png")
    # Other 3 orientations: Rotate around y-axis.
    plot_polycube(np.rot90(forward_pc, k=1, axes=(1,0)), i, TEST_PATH + f"Figure{i}_forward_left.png")
    plot_polycube(np.rot90(forward_pc, k=2, axes=(1,0)), i, TEST_PATH + f"Figure{i}_forward_180.png")
    plot_polycube(np.rot90(forward_pc, k=-1, axes=(1,0)), i, TEST_PATH + f"Figure{i}_forward_right.png")

    # "Backward position": Rotate left around x-axis. Original up-axis (= z-axis) has rotated to (negative) y-axis.
    backward_pc = np.rot90(pc, k=1, axes=(2,0))
    plot_polycube(backward_pc, i, TEST_PATH + f"Figure{i}_backward.png")
    # Other 3 orientations: Rotate around y-axis.
    plot_polycube(np.rot90(backward_pc, k=1, axes=(1,0)), i, TEST_PATH + f"Figure{i}_backward_left.png")
    plot_polycube(np.rot90(backward_pc, k=2, axes=(1,0)), i, TEST_PATH + f"Figure{i}_backward_180.png")
    plot_polycube(np.rot90(backward_pc, k=-1, axes=(1,0)), i, TEST_PATH + f"Figure{i}_backward_right.png")

def get_rotations(polycube:np.ndarray) -> list[np.ndarray]:
    # REMARK: 'numpy.ndarray' is not hashable, so put it into a
    # set will cause error:
    # TypeError: unhashable type: 'numpy.ndarray'
    result = list()
    pc = polycube
    # "Up" position (= z-axis)
    result.append(polycube)
    # Other 3 orientations of "up" position: Rotate around z-axis.
    result.append(np.rot90(pc, k=1, axes=(1,2)))
    result.append(np.rot90(pc, k=2, axes=(1,2)))
    result.append(np.rot90(pc, k=-1, axes=(1,2)))

    # "Left" position: Rotate left around y-axis. Original up-axis (= z-axis) has rotated to (negative) x-axis.
    left_pc = np.rot90(pc, k=1, axes=(1,0))
    result.append(left_pc)
    # Other 3 orientations: Rotate around x-axis.
    result.append(np.rot90(left_pc, k=1, axes=(2,0)))
    result.append(np.rot90(left_pc, k=2, axes=(2,0)))
    result.append(np.rot90(left_pc, k=-1, axes=(2,0)))

    # "Right" position: Rotate right around y-axis. Original up-axis (= z-axis) has rotated to x-axis.
    right_pc = np.rot90(pc, k=-1, axes=(1,0))
    result.append(right_pc)
    # Other 3 orientations: Rotate around x-axis.
    result.append(np.rot90(right_pc, k=1, axes=(2,0)))
    result.append(np.rot90(right_pc, k=2, axes=(2,0)))
    result.append(np.rot90(right_pc, k=-1, axes=(2,0)))

    # "Down" position: Rotate 180° around y-axis. Original up-axis (= z-axis) has rotated to (negative) z-axis.
    down_pc = np.rot90(pc, k=2, axes=(1,0))
    result.append(down_pc)
    # Other 3 orientations: Rotate around z-axis.
    result.append(np.rot90(down_pc, k=1, axes=(1,2)))
    result.append(np.rot90(down_pc, k=2, axes=(1,2)))
    result.append(np.rot90(down_pc, k=-1, axes=(1,2)))

    # "Forward position": Rotate right around x-axis. Original up-axis (= z-axis) has rotated to y-axis.
    forward_pc = np.rot90(pc, k=-1, axes=(2,0))
    result.append(forward_pc)
    # Other 3 orientations: Rotate around y-axis.
    result.append(np.rot90(forward_pc, k=1, axes=(1,0)))
    result.append(np.rot90(forward_pc, k=2, axes=(1,0)))
    result.append(np.rot90(forward_pc, k=-1, axes=(1,0)))

    # "Backward position": Rotate left around x-axis. Original up-axis (= z-axis) has rotated to (negative) y-axis.
    backward_pc = np.rot90(pc, k=1, axes=(2,0))
    result.append(backward_pc)
    # Other 3 orientations: Rotate around y-axis.
    result.append(np.rot90(backward_pc, k=1, axes=(1,0)))
    result.append(np.rot90(backward_pc, k=2, axes=(1,0)))
    result.append(np.rot90(backward_pc, k=-1, axes=(1,0)))

    return result

def get_transformations(polycube:np.ndarray) -> list[np.ndarray]:
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
            if not all_solutions and results:
                return

        temp -= t
        temp_result.pop()


class TransformFunc(Enum):
    ROTL90 = 1
    ROTR90 = 2
    ROT180 = 3

@dataclass
class Transform:
    func_id: TransformFunc
    axes: tuple[int, int]

@dataclass
class TransformCenterPiece:
    id: int
    center_piece1: int
    center_piece2: int
    transform_normal: list[Transform]
    transform_opposite: list[Transform]

def calculate_representation(result):
    # +x, -x, +y, -y, +z, -z
    center_pieces = [result[1, 2, 1], result[1, 0, 1], result[1, 1, 2], result[1, 1, 0],
        result[2, 1, 1], result[0, 1, 1]]
    s = defaultdict(int)
    for cp in center_pieces:
        s[cp] += 1

    pairs = [TransformCenterPiece(1, center_pieces[0], center_pieces[2], [Transform(TransformFunc.ROTR90, (2,0)), Transform(TransformFunc.ROTL90, (1,0))], [Transform(TransformFunc.ROTL90, (2,0))]), # (x, y), (y, x)
        TransformCenterPiece(2, center_pieces[0], center_pieces[3], [Transform(TransformFunc.ROTL90, (2,0)), Transform(TransformFunc.ROTL90, (1,0))], [Transform(TransformFunc.ROTR90, (2,0))]), # (x, -y), (-y, x)
        TransformCenterPiece(3, center_pieces[0], center_pieces[4], [Transform(TransformFunc.ROTR90, (1,0)), Transform(TransformFunc.ROT180, (2,0))], []),  # (x, z), (z, x)
        TransformCenterPiece(4, center_pieces[0], center_pieces[5], [Transform(TransformFunc.ROTL90, (1,0))], [Transform(TransformFunc.ROT180, (1,0)), Transform(TransformFunc.ROT180, (1,2))]), # (x, -z), (-z, x)
        TransformCenterPiece(5, center_pieces[1], center_pieces[2], [Transform(TransformFunc.ROTL90, (2,0)), Transform(TransformFunc.ROTR90, (1,0))], [Transform(TransformFunc.ROTR90, (2,0)), Transform(TransformFunc.ROT180, (1,0))]), # (-x, y), (y, -x)
        TransformCenterPiece(6, center_pieces[1], center_pieces[3], [Transform(TransformFunc.ROTR90, (2,0)), Transform(TransformFunc.ROTR90, (1,0))], [Transform(TransformFunc.ROTL90, (2,0)), Transform(TransformFunc.ROT180, (1,0))]), # (-x, -y), (-y, -x)
        TransformCenterPiece(7, center_pieces[1], center_pieces[4], [Transform(TransformFunc.ROTR90, (1,0))], [Transform(TransformFunc.ROT180, (1,2))]), # (-x, z), (z, -x)
        TransformCenterPiece(8, center_pieces[1], center_pieces[5], [Transform(TransformFunc.ROTL90, (1,0)), Transform(TransformFunc.ROT180, (2,0))], [Transform(TransformFunc.ROT180, (1,0))]), # (-x, -z), (-z, -x)
        TransformCenterPiece(9, center_pieces[2], center_pieces[4], [Transform(TransformFunc.ROTR90, (1,0)), Transform(TransformFunc.ROTL90, (2,0))], [Transform(TransformFunc.ROTR90, (1,2))]), # (y, z), (z, y)
        TransformCenterPiece(10, center_pieces[2], center_pieces[5], [Transform(TransformFunc.ROTL90, (1,0)), Transform(TransformFunc.ROTL90, (2,0))], [Transform(TransformFunc.ROT180, (1,0)), Transform(TransformFunc.ROTR90, (1,2))]), # (y, -z), (-z, y)
        TransformCenterPiece(11, center_pieces[3], center_pieces[4], [Transform(TransformFunc.ROTR90, (1,0)), Transform(TransformFunc.ROTR90, (2,0))], [Transform(TransformFunc.ROTL90, (1,2))]), # (-y, z), (z, -y)
        TransformCenterPiece(12, center_pieces[3], center_pieces[5], [Transform(TransformFunc.ROTL90, (1,0)), Transform(TransformFunc.ROTR90, (2,0))], [Transform(TransformFunc.ROT180, (1,0)), Transform(TransformFunc.ROTL90, (1,2))])] # (-y, -z), (-z, -y)

    # From the unique center piece values, find the smallest pair of adjacent
    # center piece values and transform the cube in such a way so the smallest
    # value is at the "up" position and the second smallest at the "right" position.
    transform_values = None
    for p in pairs:
        if s[p.center_piece1] > 1 or s[p.center_piece2] > 1:
            continue
        corrected_pair =  (p.id, p.center_piece1, p.center_piece2) if p.center_piece1 < p.center_piece2 else (-p.id, p.center_piece2, p.center_piece1)
        if transform_values is None:
            transform_values = corrected_pair
        elif (corrected_pair[1], corrected_pair[2]) < (transform_values[1], transform_values[2]):
            transform_values = corrected_pair

    if transform_values is not None:
        transformations = pairs[transform_values[0]-1].transform_normal if transform_values[0] > 0 else pairs[-transform_values[0]-1].transform_opposite
        representation = result
        for t in transformations:
            if t.func_id == TransformFunc.ROTL90:
                representation = np.rot90(representation, k=1, axes=t.axes)
            elif t.func_id == TransformFunc.ROTR90:
                representation = np.rot90(representation, k=-1, axes=t.axes)
            else:
                assert t.func_id == TransformFunc.ROT180
                representation = np.rot90(representation, k=2, axes=t.axes)

        # Check "up" and "right" values
        # assert transform_values[1] == representation[2,1,1] and transform_values[2] == representation[1,2,1]
        return representation

    return None

def calculate_result_groups(results):
    result_groups = []

    # Try to find representations for result
    representations = []
    for result in results:
        result2 = sum([i * r for i, r in enumerate(result)])
        representation = calculate_representation(result2)
        if representation is None:
            representations.clear()
            break
        else:
            is_duplicate = False
            for r in representations:
                if np.array_equal(r, representation):
                    is_duplicate = True
                    break
            if not is_duplicate:
                representations.append(representation)

    if representations:
        return representations

    # Fallback when cannot find representation for some results
    print("Fallback since cannot find representation for some results.")

    for result in results:
        result2 = sum([i * r for i, r in enumerate(result)])
        if not result_groups:
            result_groups.append(result2)
        else:
            result_transformations = get_transformations(result2)
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
    # REMARK: It could improve performance a little if I had swapped the first
    # polycube with the second, but it's only about a few hundredths of a
    # second. I decided against it since I've used the figure numbering in my
    # notes and the second polycube (T-shape) is less interesting for rotations.
    #
    # => transformation count: 600
    # => [96, 72, 96, 96, 96, 144]
    # => Result count: 24
    # => Distinct result count: 1
    # Time elapsed (before plotting): 10.025808811187744 s
    # Time elapsed: 10.86941933631897 s

    raw_polycubes_normal = [
        [[3, 1, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]],
        [[3, 1, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[1, 3, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
    ]
    # => transformation count: 688
    # => [64, 72, 72, 96, 96, 144, 144]
    # => Result count: 11520
    # => Distinct result count: 480
    # Time elapsed (before plotting): 366.37138652801514 s
    # Time elapsed: 367.2949261665344 s


    raw_polycubes = raw_polycubes_normal if normal_cube else raw_polycubes_custom

    # Output: type(raw_polycubes)=<class 'list'>
    # print(f"{type(raw_polycubes)=}")

    polycubes = [build_polycube(rp) for rp in raw_polycubes]
    # Output: type(polycubes)=<class 'list'>
    # print(f"{type(polycubes)=}")

    transformations = []
    for i, pc in enumerate(polycubes):
        transformations.append(get_transformations(pc))

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
        print(f"Time elapsed so far: {time.time() - start_time} s")
        result_groups = calculate_result_groups(results)
        print(f"Distinct result count: {len(result_groups)}")

    print(f"Time elapsed (before plotting): {time.time() - start_time} s")

    ### Plotting
    # Create directories for output files.
    if with_tests and not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    if with_tests:
        for i, pc in enumerate(polycubes):
            # "Up" position
            plot_polycube(pc, i, TEST_PATH + f"Figure{i}_up.png")
            if i == 0:
                plot_variations(pc, i)

    fig = plt.figure()
    ax = add_subplot(fig)
    for i, pc in enumerate(first_result):
         plot_polycube(pc, i, "", ax)
    plt.savefig(RESULT_PATH + "Result.png")
    plt.close(fig)
    for i, pc in enumerate(first_result):
         plot_polycube(pc, i, RESULT_PATH + f"Result{i}.png")

    print(f"Time elapsed: {time.time() - start_time} s")

    # Show result in GUI.
    fig = plt.figure()
    ax = add_subplot(fig)
    for i, pc in enumerate(first_result):
         plot_polycube(pc, i, "", ax)
    plt.show()


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

# Time elapsed (before plotting): 0.2074108123779297 s
# Time elapsed: 1.04561448097229 s
