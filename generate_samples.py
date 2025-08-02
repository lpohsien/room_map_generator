#!/usr/bin/env python3
"""
Alternative interface for the occupancy grid map generator.
"""

from generate_map import OccupancyGridGenerator, output_map


def generate_rooms_map(width=256, height=256, seed=42, wall_thickness=3, 
                      min_room_area=25, density=0.4, output_path="rooms_output.png"):
    """
    Generate a rooms-style occupancy grid map.
    
    Args:
        width: Map width in pixels
        height: Map height in pixels  
        seed: Random seed for reproducibility
        wall_thickness: Thickness of walls
        min_room_area: Minimum room size to keep
        density: Target fraction of black pixels (walls)
        output_path: Where to save the PNG
        
    Returns:
        NumPy array representing the occupancy grid
    """
    return output_map(width=width, 
                      height=height, 
                      plot_type='rooms', 
                      thickness=wall_thickness,
                      min_room_area=min_room_area, 
                      density=density, 
                      seed=seed, 
                      output_path=output_path)


def generate_tunnels_map(width=256, height=256, seed=42, tunnel_width=4,
                        min_room_area=20, density=0.6, output_path="tunnels_output.png"):
    """
    Generate a tunnels-style occupancy grid map.
    
    Args:
        width: Map width in pixels
        height: Map height in pixels
        seed: Random seed for reproducibility  
        tunnel_width: Width of tunnels
        min_room_area: Minimum connected area to keep
        density: Target fraction of black pixels (walls)
        output_path: Where to save the PNG
        
    Returns:
        NumPy array representing the occupancy grid
    """
    return output_map(width, height, seed, 'tunnels', tunnel_width,
                     min_room_area, density, output_path)


def generate_sample_maps():
    """Generate a variety of sample maps for testing."""
    import os
    
    # Create samples directory
    os.makedirs("generated_samples", exist_ok=True)
    
    print("Generating sample maps...")
    seed = 42
    
    # Rooms examples
    generate_rooms_map(100, 100, seed, 3, 30, 0.3, "generated_samples/rooms_light.png")
    generate_rooms_map(100, 100, seed, 5, 50, 0.5, "generated_samples/rooms_medium.png")
    generate_rooms_map(100, 100, seed, 2, 20, 0.7, "generated_samples/rooms_dense.png")

    print("Sample maps generated in 'generated_samples/' directory")


if __name__ == "__main__":
    generate_sample_maps()