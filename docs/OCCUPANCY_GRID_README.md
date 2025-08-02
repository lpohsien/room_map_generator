# Occupancy Grid Map Generator

A Python implementation of a randomized occupancy grid generator that creates simplified indoor floor plans as RGB images. The implementation follows the specifications in `map_generation_spec.md`.

## Features

- **Two Generation Modes:**
  - **Rooms Mode**: Starts with free space (white) and carves walls (black) to create rooms and hallways
  - **Tunnels Mode**: Starts with occupied space (black) and carves tunnels (white) to create corridors and chambers

- **Configurable Parameters:**
  - Map dimensions (width × height)
  - Random seed for reproducibility
  - Wall thickness or tunnel width
  - Minimum room area (small regions are sealed)
  - Target density (fraction of black pixels)

- **Connectivity Guarantee**: All white pixels are 4-connected (reachable via up/down/left/right movement)

## Installation

The project uses UV for dependency management. Install dependencies with:

```bash
uv add numpy pillow matplotlib
```

## Usage

### Command Line Interface

Generate maps using the main CLI:

```bash
# Generate a rooms-style map
uv run python generate_map.py --width 512 --height 512 --seed 1234 --plot_type rooms --thickness 4 --min_room_area 50 --density 0.45

# Generate a tunnels-style map  
uv run python generate_map.py --width 256 --height 256 --seed 42 --plot_type tunnels --thickness 3 --min_room_area 20 --density 0.6 --output my_tunnel_map.png
```

### Python API

Use the generator programmatically:

```python
from generate_map import OccupancyGridGenerator, output_map

# Quick generation
occupancy_grid = output_map(
    width=256, height=256, seed=42,
    plot_type='rooms', thickness=3, 
    min_room_area=30, density=0.4,
    output_path='output.png'
)

# Advanced usage
generator = OccupancyGridGenerator(256, 256, seed=42)
grid = generator.generate_map('tunnels', thickness=4, min_room_area=20, density=0.6)
generator.save_png('custom_map.png')
```

### Alternative Interface

Generate sample maps using the convenience functions:

```python
from generate_maps import generate_rooms_map, generate_tunnels_map, generate_sample_maps

# Generate specific types
rooms_map = generate_rooms_map(256, 256, seed=42, wall_thickness=3, density=0.4)
tunnels_map = generate_tunnels_map(256, 256, seed=99, tunnel_width=4, density=0.6)

# Generate a variety of sample maps
generate_sample_maps()  # Creates maps in 'generated_samples/' directory
```

## Parameters

- `width`, `height`: Output image dimensions in pixels
- `seed`: Random seed for reproducible generation
- `plot_type`: Either 'rooms' or 'tunnels' 
- `thickness`: Wall thickness (rooms mode) or tunnel width (tunnels mode)
- `min_room_area`: Minimum size for connected free regions (smaller ones are sealed)
- `density`: Target fraction of black pixels (0.0 = all white, 1.0 = all black)

## Algorithm Overview

### Rooms Mode
1. Initialize grid to all white (free space)
2. Place random wall seed points
3. Grow straight black wall segments from each seed
4. Post-process to seal small rooms and add doors

### Tunnels Mode  
1. Initialize grid to all black (occupied space)
2. Place random tunnel seed points
3. Carve white L-shaped tunnels between seeds
4. Add occasional side chambers
5. Post-process to seal small areas and ensure connectivity

### Post-Processing
1. Find all connected components of white pixels using BFS
2. Seal components smaller than `min_room_area` (turn to black)
3. Add doors (random openings) to remaining components

## Testing

Run the unit tests to verify correctness:

```bash
uv run python -m pytest test_generate_map.py -v
```

The tests verify:
- Reproducibility with fixed seeds
- Correct output dimensions and data types
- Connectivity of free space
- Small room sealing functionality
- Density parameter effects
- Differences between modes

## Generated Files

The implementation creates:
- `generate_map.py`: Main implementation with CLI
- `generate_maps.py`: Alternative convenience interface
- `test_generate_map.py`: Comprehensive unit tests

## Example Output

The generator creates PNG images where:
- **White pixels (255,255,255)**: Free space (walkable areas)
- **Black pixels (0,0,0)**: Walls/obstacles

Maps are saved as standard RGB PNG files that can be viewed in any image viewer or processed by other applications.
