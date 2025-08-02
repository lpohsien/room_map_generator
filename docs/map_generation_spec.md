# project_spec.yaml

project:
  name: randomized_occupancy_grid_generator
  description: >
    A Python library that procedurally generates simplified indoor
    occupancy–grid maps (floor plans) as RGB images. White pixels are free
    space; black pixels are walls/obstacles. All white space must be
    4-connected. Supports two modes (“rooms” and “tunnels”), with config parameters
    for wall/tunnel thickness, seed control, target density, and minimum room size.

metadata:
  language: python3
  dependencies:
    - numpy
    - PIL                                 # or pillow
    - matplotlib                          # only for optional debugging plots
  package_manager: uv
  entry_point: generate_map.py
  resources: 
    example_maps_dir: ./samples/tunnels  # directory containing example tunnel maps

requirements:
  functional:
    - output_map(width, height, seed, plot_type, thickness, min_room_area, density)  
      generates:
        - a NumPy array of shape (height, width, 3), dtype=uint8
        - writes out “output.png” as the RGB visualization
    - seedable randomness: runs must be repeatable given the same seed
    - connectivity: every white pixel may reach every other white pixel via up/down/left/right
    - randomized but “straight-path” biased corridors and walls
    - two plot modes:
        * rooms: start free, carve straight black “walls” of configurable thickness to partition into rectangular-ish rooms connected by doorways
        * tunnels: start filled, carve straight white “tunnels” of configurable width, optionally adding small rectangular rooms off the main tunnels
    - post‐processing:
        1. BFS to identify each connected component of white pixels
        2. if component size < min_room_area: fill it in (turn to black)
           else: carve a 1-pixel “door” at a random location on its surrounding walls
    - density control: guide the generator toward the target % black‐pixel coverage but need not be exact—just bias splits/growth to approach it

  non-functional:
    - clean, PEP8-compliant code with docstrings on every public function
    - a simple CLI interface:
        ```bash
        python generate_map.py \
          --width 512 --height 512 \
          --seed 1234 \
          --plot_type rooms \
          --thickness 4 \
          --min_room_area 50 \
          --density 0.45
        ```
    - unit tests covering:
        * connectivity and BFS sealing logic
        * reproducibility with fixed seed

inputs:
  - name: width
    type: int
    description: pixel width of the output image
  - name: height
    type: int
    description: pixel height of the output image
  - name: seed
    type: int
    description: random seed for reproducibility
  - name: plot_type
    type: enum[rooms, tunnels]
    description: >
      “rooms”: carve walls into free space to make rooms & hallways  
      “tunnels”: carve free space out of occupied space to make tunnels & pockets
  - name: thickness
    type: int
    description: >
      wall thickness (in rooms mode) or tunnel width (in tunnels mode)
  - name: min_room_area
    type: int
    description: >
      any connected free region smaller than this will be sealed off
  - name: density
    type: float  # 0.0–1.0
    description: target fraction of black pixels (soft constraint)

outputs:
  - name: occupancy_grid
    type: ndarray(height, width, 3), dtype=uint8
    description: RGB image array: white=[255,255,255], black=[0,0,0]
  - name: output.png
    type: file
    description: PNG rendering of occupancy_grid

suggested_algorithm:
  rooms:
    - initialize grid to all free (white)
    - sample N “wall-seed” points uniformly
    - for each seed, grow straight black segments of length ∼O(min(width,height))
      in a couple of orthogonal directions, thickness = config.thickness
    - optionally jitter end-points to avoid perfectly parallel walls
  tunnels:
    - initialize grid to all occupied (black)
    - sample M “tunnel-seed” points
    - carve straight white corridors of width=config.thickness between seeds,
      biasing for L-shaped pair of straight runs
    - offshoot small rectangular “rooms” next to some corridors
  postprocessing:
    - run BFS/DFS to label white connected components
    - for each component:
        if size < min_room_area:
          fill it in (set to black)
        else:
          find its boundary pixels, choose one at random, and carve a 1-px doorway
          (set that black pixel to white)
    - compute final black/white ratio, compare to density; if too low/high,
      tweak by randomly adding/removing small wall/tunnel segments and repeat
  utilities:
    - helper(fn: carve_line(x1,y1,x2,y2, color, width))
    - helper(fn: find_boundary(component_mask))
    - helper(fn: save_png(array, path))

cli:
  flags:
    - --width
    - --height
    - --seed
    - --plot_type
    - --thickness
    - --min_room_area
    - --density
    - --output (default: output.png)
  examples:
    - python generate_map.py --width 256 --height 256 --seed 42 --plot_type tunnels --thickness 3 --min_room_area 20 --density 0.4

testing:
  unit_tests:
    - test_connectivity_after_carving
    - test_small_room_sealing
    - test_reproducibility_with_seed
    - test_density_biasing

license: MIT
