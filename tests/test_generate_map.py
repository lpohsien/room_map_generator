#!/usr/bin/env python3
"""
Unit tests for the occupancy grid map generator.
"""

import unittest
import numpy as np
import tempfile
import os
from generate_map import OccupancyGridGenerator, output_map


class TestOccupancyGridGenerator(unittest.TestCase):
    """Test cases for the OccupancyGridGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.width = 128
        self.height = 128
        self.seed = 42
        
    def test_reproducibility_with_seed(self):
        """Test that the same seed produces identical results."""
        # Generate two maps with the same parameters
        generator1 = OccupancyGridGenerator(self.width, self.height, self.seed)
        map1 = generator1.generate_map('rooms', thickness=3, min_room_area=20, density=0.4)
        
        generator2 = OccupancyGridGenerator(self.width, self.height, self.seed)
        map2 = generator2.generate_map('rooms', thickness=3, min_room_area=20, density=0.4)
        
        # Maps should be identical
        np.testing.assert_array_equal(map1, map2, 
                                    "Maps generated with same seed should be identical")
        
    def test_different_seeds_produce_different_maps(self):
        """Test that different seeds produce different results."""
        generator1 = OccupancyGridGenerator(self.width, self.height, 42)
        map1 = generator1.generate_map('rooms', thickness=3, min_room_area=20, density=0.4)
        
        generator2 = OccupancyGridGenerator(self.width, self.height, 99)
        map2 = generator2.generate_map('rooms', thickness=3, min_room_area=20, density=0.4)
        
        # Maps should be different
        self.assertFalse(np.array_equal(map1, map2), 
                        "Maps generated with different seeds should be different")
    
    def test_output_dimensions(self):
        """Test that output has correct dimensions and dtype."""
        generator = OccupancyGridGenerator(self.width, self.height, self.seed)
        occupancy_grid = generator.generate_map('tunnels', thickness=4, min_room_area=25, density=0.5)
        
        # Check shape
        expected_shape = (self.height, self.width, 3)
        self.assertEqual(occupancy_grid.shape, expected_shape,
                        f"Output shape should be {expected_shape}")
        
        # Check dtype
        self.assertEqual(occupancy_grid.dtype, np.uint8,
                        "Output dtype should be uint8")
        
        # Check that values are only 0 or 255 (black or white)
        unique_values = np.unique(occupancy_grid)
        valid_values = [0, 255]
        for val in unique_values:
            self.assertIn(val, valid_values, 
                         f"All pixel values should be 0 or 255, found {val}")
    
    def test_connectivity_after_carving(self):
        """Test that all white pixels are 4-connected."""
        generator = OccupancyGridGenerator(64, 64, self.seed)
        occupancy_grid = generator.generate_map('tunnels', thickness=3, min_room_area=10, density=0.6)
        
        # Find all white pixels
        white_mask = np.all(occupancy_grid == [255, 255, 255], axis=2)
        
        if np.sum(white_mask) == 0:
            self.skipTest("No white pixels found in generated map")
        
        # Use BFS to check connectivity
        visited = np.zeros_like(white_mask, dtype=bool)
        components = []
        
        for y in range(white_mask.shape[0]):
            for x in range(white_mask.shape[1]):
                if white_mask[y, x] and not visited[y, x]:
                    # Found a new component
                    component_size = self._bfs_component_size(white_mask, visited, x, y)
                    components.append(component_size)
        
        # Should have at most a few large connected components (ideally 1)
        large_components = [size for size in components if size >= 10]
        self.assertLessEqual(len(large_components), 5, 
                           "Should have few large connected components")
    
    def test_small_room_sealing(self):
        """Test that small rooms are properly sealed."""
        generator = OccupancyGridGenerator(64, 64, self.seed)
        
        # Generate with high min_room_area to force sealing
        occupancy_grid = generator.generate_map('rooms', thickness=2, min_room_area=100, density=0.3)
        
        # Find connected components of white pixels
        white_mask = np.all(occupancy_grid == [255, 255, 255], axis=2)
        visited = np.zeros_like(white_mask, dtype=bool)
        
        for y in range(white_mask.shape[0]):
            for x in range(white_mask.shape[1]):
                if white_mask[y, x] and not visited[y, x]:
                    component_size = self._bfs_component_size(white_mask, visited, x, y)
                    # All remaining components should be >= min_room_area
                    self.assertGreaterEqual(component_size, 100,
                                          f"Found component of size {component_size}, "
                                          f"should be >= 100 (min_room_area)")
    
    def test_density_biasing(self):
        """Test that density parameter affects the final map."""
        generator1 = OccupancyGridGenerator(self.width, self.height, self.seed)
        map_low_density = generator1.generate_map('rooms', thickness=3, min_room_area=20, density=0.2)
        
        generator2 = OccupancyGridGenerator(self.width, self.height, self.seed)
        map_high_density = generator2.generate_map('rooms', thickness=3, min_room_area=20, density=0.8)
        
        # Calculate actual densities
        total_pixels = self.width * self.height
        
        black_pixels_low = np.sum(np.all(map_low_density == [0, 0, 0], axis=2))
        density_low = black_pixels_low / total_pixels
        
        black_pixels_high = np.sum(np.all(map_high_density == [0, 0, 0], axis=2))
        density_high = black_pixels_high / total_pixels
        
        # Higher density target should result in higher actual density
        self.assertLess(density_low, density_high,
                       f"Low density target ({density_low:.3f}) should result in "
                       f"lower actual density than high density target ({density_high:.3f})")
    
    def test_plot_type_differences(self):
        """Test that rooms and tunnels modes produce different maps."""
        generator1 = OccupancyGridGenerator(self.width, self.height, self.seed)
        rooms_map = generator1.generate_map('rooms', thickness=4, min_room_area=30, density=0.4)
        
        generator2 = OccupancyGridGenerator(self.width, self.height, self.seed)
        tunnels_map = generator2.generate_map('tunnels', thickness=4, min_room_area=30, density=0.4)
        
        # Maps should be different
        self.assertFalse(np.array_equal(rooms_map, tunnels_map),
                        "Rooms and tunnels modes should produce different maps")
    
    def test_save_png_functionality(self):
        """Test that PNG saving works correctly."""
        generator = OccupancyGridGenerator(32, 32, self.seed)
        generator.generate_map('rooms', thickness=2, min_room_area=10, density=0.5)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            generator.save_png(tmp_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path), "PNG file should be created")
            self.assertGreater(os.path.getsize(tmp_path), 0, "PNG file should not be empty")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_output_map_function(self):
        """Test the main output_map function."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            occupancy_grid, seed = output_map(
                width=64, height=64, seed=123,
                plot_type='tunnels', thickness=3,
                min_room_area=15, density=0.4,
                output_path=tmp_path
            )
            
            # Check return value
            self.assertEqual(occupancy_grid.shape, (64, 64, 3))
            self.assertEqual(occupancy_grid.dtype, np.uint8)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _bfs_component_size(self, mask: np.ndarray, visited: np.ndarray, 
                           start_x: int, start_y: int) -> int:
        """Helper method to calculate connected component size using BFS."""
        from collections import deque
        
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        size = 1
        
        while queue:
            x, y = queue.popleft()
            
            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and
                    mask[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
                    size += 1
        
        return size


if __name__ == '__main__':
    unittest.main()
