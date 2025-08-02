#!/usr/bin/env python3
"""
Test script to verify room connectivity with corridor connections.
"""

import numpy as np
from generate_map import OccupancyGridGenerator
from collections import deque


def test_room_connectivity():
    """Test that all large rooms are connected via corridors."""
    print("Testing room connectivity...")
    
    # Generate a map that's likely to have disconnected rooms
    generator = OccupancyGridGenerator(128, 128, seed=789)
    occupancy_grid = generator.generate_map('rooms', thickness=5, min_room_area=25, density=0.5)
    
    # Find all white pixels (free space)
    white_mask = np.all(occupancy_grid == [255, 255, 255], axis=2)
    total_white_pixels = np.sum(white_mask)
    
    if total_white_pixels == 0:
        print("No free space found - test skipped")
        return
    
    # Use BFS to find the largest connected component
    visited = np.zeros_like(white_mask, dtype=bool)
    largest_component_size = 0
    
    for y in range(white_mask.shape[0]):
        for x in range(white_mask.shape[1]):
            if white_mask[y, x] and not visited[y, x]:
                # BFS to find component size
                component_size = 0
                queue = deque([(x, y)])
                visited[y, x] = True
                
                while queue:
                    cx, cy = queue.popleft()
                    component_size += 1
                    
                    # Check 4-connected neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        
                        if (0 <= nx < white_mask.shape[1] and 0 <= ny < white_mask.shape[0] and
                            white_mask[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                
                largest_component_size = max(largest_component_size, component_size)
    
    # Calculate connectivity ratio
    connectivity_ratio = largest_component_size / total_white_pixels
    
    print(f"Total white pixels: {total_white_pixels}")
    print(f"Largest connected component: {largest_component_size}")
    print(f"Connectivity ratio: {connectivity_ratio:.3f}")
    
    # Good connectivity means most free space is connected
    if connectivity_ratio > 0.8:
        print("✅ EXCELLENT connectivity - most rooms are connected!")
    elif connectivity_ratio > 0.6:
        print("✅ GOOD connectivity - majority of rooms are connected")
    elif connectivity_ratio > 0.4:
        print("⚠️  MODERATE connectivity - some rooms may be isolated")
    else:
        print("❌ POOR connectivity - many rooms are disconnected")
    
    return connectivity_ratio


def test_multiple_maps():
    """Test connectivity across multiple random maps."""
    print("\nTesting connectivity across multiple maps...")
    
    connectivity_scores = []
    
    for i, seed in enumerate([100, 200, 300, 400, 500]):
        print(f"\nMap {i+1} (seed {seed}):")
        generator = OccupancyGridGenerator(96, 96, seed=seed)
        occupancy_grid = generator.generate_map('rooms', thickness=3, min_room_area=30, density=0.45)
        
        # Quick connectivity check
        white_mask = np.all(occupancy_grid == [255, 255, 255], axis=2)
        total_white = np.sum(white_mask)
        
        if total_white == 0:
            continue
        
        # Find largest component
        visited = np.zeros_like(white_mask, dtype=bool)
        max_component = 0
        
        for y in range(white_mask.shape[0]):
            for x in range(white_mask.shape[1]):
                if white_mask[y, x] and not visited[y, x]:
                    size = 0
                    queue = deque([(x, y)])
                    visited[y, x] = True
                    
                    while queue:
                        cx, cy = queue.popleft()
                        size += 1
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < white_mask.shape[1] and 0 <= ny < white_mask.shape[0] and
                                white_mask[ny, nx] and not visited[ny, nx]):
                                visited[ny, nx] = True
                                queue.append((nx, ny))
                    
                    max_component = max(max_component, size)
        
        ratio = max_component / total_white
        connectivity_scores.append(ratio)
        print(f"  Connectivity: {ratio:.3f}")
    
    if connectivity_scores:
        avg_connectivity = np.mean(connectivity_scores)
        print(f"\nAverage connectivity across {len(connectivity_scores)} maps: {avg_connectivity:.3f}")
        
        if avg_connectivity > 0.75:
            print("✅ Overall EXCELLENT connectivity!")
        elif avg_connectivity > 0.6:
            print("✅ Overall GOOD connectivity!")
        else:
            print("⚠️  Connectivity could be improved")


if __name__ == "__main__":
    test_room_connectivity()
    test_multiple_maps()
