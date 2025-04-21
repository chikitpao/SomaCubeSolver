# SomaCubeSolver
Solver for puzzles[1] similar to Soma cube using backtracking algorithm (in Julia and in Python).

[1] See [cube_puzzle.jpg](cube_puzzle.jpg). It's a puzzle brought to work one day by one of my coworkers.

Required packages
------------
**Julia:** Plots

**Python:** Matplotlib and NumPy

Command line parameters
------------
**"--with-tests":** Do tests on demand.

**"--all-solutions":** Find all solutions instead only the first one.

**"--normal-cube":** Find solution for the normal Soma cube (see Wikipedia page [https://en.wikipedia.org/wiki/Soma_cube](https://en.wikipedia.org/wiki/Soma_cube)) instead of the one in my workspace (see [cube_puzzle.jpg](cube_puzzle.jpg)).

Code / Algorithmus description
------------

### Polycube definition
Each polycube of the puzzle are defined as a 3x3 matrix in the top view. Bottom layer occupied -> Bit 0 is set; middle layer occupied -> Bit 1 is set; top layer occupied -> Bit 2 is set. E.g. the first polycube is defined like the following:

**Julia**
```
    [1 1 0
     3 0 0
     1 0 0]
```

**Python (NumPy)**
```
    [[1, 1, 0],
     [3, 0, 0],
     [1, 0, 0]]
```

Then these polycube definitions are converted into 3D matrices. However, Julia and Python (NumPy) use different indices (axes) for row, column and layer:

| Julia | Python (NumPy) |
| --- | --- |
| Axis 1: row | Axis 0: "layer" |
| Axis 2: column | Axis 1: row |
| Axis 3: "layer"| Axis 2: column |


### Polycube transformations
Now the polycubes need to be transformed to all possible positions and rotations.

A polycube can have **maximum 24 different transformations using rotation**. Simply put, the "up" position can be rotated to "left", "right", "front", "back", and "down" position (6 orientations), from there rotation 90° left, 180° and 90° right can be done around the axis which the original "up"-axis has rotated to (4 orientations).

**Julia**: Rotation is done by the functions *rotl90* (rotate left 90°), *rot180* (rotate 180°), and *rotr90* (rotate right 90°). These need to be specified as the first argument for function *mapslices*, which applies the rotation functions to every slice of matrix. E.g. Apply rotation to matrix *pc* around axis 3:
```
    mapslices(rotl90, pc, dims=[1,2])
    mapslices(rot180, pc, dims=[1,2])
    mapslices(rotr90, pc, dims=[1,2])
```


**Python (NumPy)** Rotation is done by the functions *numpy.rot90*, which rotates the matrix in the plane specified by the parameter *axes*. E.g. *axes=(0,1)* means rotate axes 2 in the direction from axis 0 to axis 1, which will cause a counter-clockwise turn. Rotation count is defined by the paramater *k* (which can also be negative). E.g. Apply rotation to matrix *pc* counter-clockwise around axis 2 for one time, two times, and minus one time respectively:
```
    numpy.rot90(pc, k=1, axes=(0,1))
    numpy.rot90(pc, k=2, axes=(0,1))
    numpy.rot90(pc, k=-1, axes=(0,1))
```

We also need to move the rotated polycube along x-, y- and z-axis (i.e. **translation**). For this purpose, the polycube is shifted to the lowest possible position so every movement with non-negative delta in all axes is a valid translation.

**Julia**: *circshift(rotation_mod, (dx, dy, dz)))*

**Python (NumPy)**: *np.roll(rotation_mod, (dz, dx, dy), (0, 1, 2)))*

It's possible that different transformations of the polycube produce the same result, so we want to eliminate duplicated transformations. Whereas **Julia matrices (type: *Array{Int64, 3}*)** can be used directly with a **Set**, **NumPy matrices (type *numpy.ndarray*) are not hashable** so putting it into a set will cause an *TypeError*. A possible approach is to **remove duplicated values by not inserting them into a new list:**
```
    # Remove duplicates
    result = []
    for candidate in temp_result:
        if all(not np.array_equal(M, candidate) for M in result):
            result.append(candidate)
```

### Calculate solutions
Solutions are calculated using **backtracking algorithm**: Partial solutions are built with transformations of first polycube, of second polycube etc. until either all polycubes are used with a valid solution. When a specific partial solution results in a valid solution or leads to no solution, either try the next transformation of the current polycube, or when all the transformations of the current polycube are exhausted, try the next transformation of the previous polycube.

Clashing polycubes are checked by whether the sum of polycube matrices has an element with value greater than one. So if variable *temp* is the sum, check is done like this:

**Julia**: *maximum(temp) > 1*

**Python (NumPy)**: *numpy.max(temp) > 1*


### Result groups
The results from the previous step usually contain groups of solutions where all solutions within a groups are only a rotated version of another within the same group. The result groups can be determined by trying all 24 possible rotations of every result and check whether one of these rotations will lead to another result already in the new list. 

|  | Result count | Distinct result count ( = groups) | Execution time (Julia) [2] | Execution time (Python) |
| --- | --- | --- | --- | --- |
| Puzzle cube from coworker | 24 | 1 | 23.0 s | 21.5 s |
| Soma cube | 11520 | 480 | 75.8 s | 1846 s |

[2] Initialization and usage of Plots.jl (for plotting) in Julia seems to be a performance bottleneck. And it is even worse on Windows. Without code related to Plots.jl, execution time is reduced by ca. 17 s.
