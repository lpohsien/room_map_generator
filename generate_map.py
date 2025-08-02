#!/usr/bin/env python3
"""
Randomized Occupancy Grid Map Generator

A Python library that procedurally generates simplified indoor
occupancy-grid maps (floor plans) as RGB images. White pixels are free
space; black pixels are walls/obstacles. All white space is 4-connected.
"""

import argparse
import random
import numpy as np
from PIL import Image
from collections import deque
from typing import Tuple, List

WALL_COLOR = np.array([255, 255, 255], dtype=np.uint8)  # White
FREE_COLOR = np.array([0, 0, 0], dtype=np.uint8)        # Black
# Value channel for the value map
RED = np.array([255, 0, 0], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 255], dtype=np.uint8)
VALUE_CHANNEL = 2


class ImagePoint:

    def __init__(self, coords: tuple[int, int]|None = None, index: tuple[int, int]|None = None):
        assert (not coords is None) or (not index is None), \
            "ImagePoint: at least coords or index must be specified"
        assert coords is None or index is None, \
            "ImagePoint: coords and index cannot be used together for init"
        
        if coords:
            self.x = coords[0]
            self.y = coords[1]
            self.r = coords[1]
            self.c = coords[0]
        else:
            self.x = index[1]
            self.y = index[0]
            self.r = index[0]
            self.c = index[1]

    def distance_to_point(self, p: 'ImagePoint'):
        a = np.array([self.r, self.c])
        b = np.array([p.r, p.c])
        return np.linalg.norm(a-b)
    
    def distance_to_index(self, index: tuple[int, int]):
        a = np.array([self.r, self.c])
        b = np.array(index)
        return np.linalg.norm(a-b)
        
    def distance_to_coords(self, coords: tuple[int, int]):
        a = np.array([self.x, self.y])
        b = np.array(coords)
        return np.linalg.norm(a-b)
    
    def __repr__(self):
        return f"ImagePoint(r={self.r}, c={self.c}, x={self.x}, y={self.y})"

class OccupancyGridGenerator:
    """
    Generates occupancy grid maps with two modes: 'rooms' and 'tunnels'.
    """
    
    def __init__(self, width: int, height: int, seed: int = None, margin: int = 5):
        """
        Initialize the generator.
        
        Args:
            width: Pixel width of the output image
            height: Pixel height of the output image  
            seed: Random seed for reproducibility
            margin: Margin around the map to ensure map is bounded
        """
        self.width = width - margin * 2
        self.height = height - margin * 2
        self.seed = int(np.random.randint(0, 2**31) if seed is None else seed)
        random.seed(self.seed )
        np.random.seed(self.seed )
        self.margin = margin

        assert self.width > 0 and self.height > 0, \
            "Width and height must be greater than twice the margin."

        # Initialize grid as RGB array excluding margin
        self.grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Color constants
        self.FREE_COLOR = FREE_COLOR
        self.WALL_COLOR = WALL_COLOR
    
    def generate_map(self, plot_type: str, thickness: int, 
                     min_room_area: int, density: float) -> np.ndarray:
        """
        Generate an occupancy grid map.
        
        Args:
            plot_type: Either 'rooms' or 'tunnels'
            thickness: Wall thickness (rooms mode) or tunnel width (tunnels mode)
            min_room_area: Minimum connected free region size
            density: Target fraction of black pixels
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        if plot_type == 'rooms':
            self._generate_rooms_map(thickness, density)
        elif plot_type == 'tunnels':
            self._generate_tunnels_map(thickness, density)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'rooms' or 'tunnels'")
        
        # Post-processing: seal small rooms and ensure connectivity
        self._postprocess_map(min_room_area, thickness)

        # Add back the margin as walls
        self.grid = np.pad(self.grid, ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
                           mode='constant', constant_values=0)

        return self.grid.copy()
    
    def _generate_rooms_map(self, wall_thickness: int, target_density: float):
        """Generate a rooms-style map by carving walls into free space."""
        # Initialize to all free space (white)
        self.grid[:, :] = self.FREE_COLOR
        
        # Calculate number of wall seeds based on target density
        total_pixels = self.width * self.height
        target_wall_pixels = int(total_pixels * target_density)
        
        # Sample wall seed points
        while not self._reached_target_density(target_wall_pixels):
            # randomly sample a new point and pair it up with another point seen so far to draw a rectangular wall
            p1 = ImagePoint(coords=(random.randint(0, self.width - 1), random.randint(0, self.height - 1)))
            p2 = ImagePoint(coords=(random.randint(0, self.width - 1), random.randint(0, self.height - 1)))

            self._carve_wall_from_seeds(p1, p2, wall_thickness)
        

    def _generate_tunnels_map(self, tunnel_width: int, target_density: float):
        """Generate a tunnels-style map by carving tunnels into occupied space."""
        # Initialize to all occupied space (black)
        self.grid[:, :] = self.WALL_COLOR
        
        # Calculate number of tunnel seeds
        total_pixels = self.width * self.height
        target_free_pixels = int(total_pixels * (1.0 - target_density))
        
        # Sample tunnel seed points
        num_seeds = max(3, target_free_pixels // (tunnel_width * min(self.width, self.height) // 6))
        
        seeds = []
        for _ in range(num_seeds):
            x = random.randint(tunnel_width, self.width - tunnel_width - 1)
            y = random.randint(tunnel_width, self.height - tunnel_width - 1)
            seeds.append((x, y))
        
        # Connect seeds with tunnels
        for i, (x1, y1) in enumerate(seeds):
            # Carve initial room at seed
            self._carve_rectangle(x1 - tunnel_width//2, y1 - tunnel_width//2,
                                tunnel_width, tunnel_width, self.FREE_COLOR)
            
            # Connect to next seed (or random previous seed)
            if i < len(seeds) - 1:
                x2, y2 = seeds[i + 1]
            else:
                x2, y2 = seeds[random.randint(0, len(seeds) - 1)]
            
            # Carve L-shaped tunnel
            self._carve_l_tunnel(x1, y1, x2, y2, tunnel_width)
            
            # Occasionally add side rooms
            if random.random() < 0.3:
                self._add_side_room(x1, y1, tunnel_width)

    def _carve_wall_from_seeds(self, p1: ImagePoint, p2: ImagePoint, thickness: int):
        """Carve walls given a pair of seed points in orthogonal directions."""
        # Draw a rectangle where the 2 chosen points are the daigonal corners
        self.grid[p1.r, p1.c] = np.array([255, 0, 0], dtype=np.uint8)
        self.grid[p2.r, p2.c] = np.array([0, 255, 0], dtype=np.uint8)
        self._draw_horizontal_line(p1, p2, thickness, self.WALL_COLOR, True)
        self._draw_horizontal_line(p2, p1, thickness, self.WALL_COLOR, True)
        self._draw_vertical_line(p1, p2, thickness, self.WALL_COLOR, True)
        self._draw_vertical_line(p2, p1, thickness, self.WALL_COLOR, True)


    def _draw_horizontal_line(self, p1: ImagePoint, p2: ImagePoint, width: int, color: np.ndarray, extend: bool = False):
        ref_r = p1.r
        half_width = width // 2
        r1, r2 = max(0, ref_r - half_width), min(self.grid.shape[0], ref_r + half_width)

        c1, c2 = p1.c, p2.c
        if c2 < c1: c1, c2 = c2, c1
        if extend:
            c1, c2 = max(0, c1 - half_width), min(self.grid.shape[1], c2 + half_width)

        self.grid[r1:r2+1, c1:c2+1] = color

    def _draw_vertical_line(self, p1: ImagePoint, p2: ImagePoint, width: int, color: np.ndarray, extend: bool = False):
        ref_c = p1.c
        half_width = width // 2
        c1, c2 = max(0, ref_c - half_width), min(self.grid.shape[1], ref_c + half_width)

        r1, r2 = p1.r, p2.r
        if r2 < r1: r1, r2 = r2, r1
        if extend:
            r1, r2 = max(0, r1 - half_width), min(self.grid.shape[0], r2 + half_width)

        self.grid[r1:r2+1, c1:c2+1] = color
    
    def _carve_l_tunnel(self, x1: int, y1: int, x2: int, y2: int, width: int):
        """Carve an L-shaped tunnel between two points."""
        # Choose corner point (either (x1, y2) or (x2, y1))
        if random.random() < 0.5:
            corner_x, corner_y = x1, y2
        else:
            corner_x, corner_y = x2, y1
        
        # Carve first segment
        self._carve_line(x1, y1, corner_x, corner_y, self.FREE_COLOR, width)
        # Carve second segment  
        self._carve_line(corner_x, corner_y, x2, y2, self.FREE_COLOR, width)
    
    def _add_side_room(self, tunnel_x: int, tunnel_y: int, tunnel_width: int):
        """Add a small rectangular room adjacent to a tunnel."""
        room_size = random.randint(tunnel_width * 2, tunnel_width * 4)
        
        # Choose side direction
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = random.choice(directions)
        
        # Calculate room position
        room_x = tunnel_x + dx * (tunnel_width + room_size // 2)
        room_y = tunnel_y + dy * (tunnel_width + room_size // 2)
        
        # Carve room
        self._carve_rectangle(room_x - room_size//2, room_y - room_size//2,
                            room_size, room_size, self.FREE_COLOR)
        
        # Connect to tunnel
        self._carve_line(tunnel_x, tunnel_y, room_x, room_y, self.FREE_COLOR, tunnel_width//2)
    
    def _carve_line(self, x1: int, y1: int, x2: int, y2: int, color: np.ndarray, width: int):
        """Carve a line of specified width and color."""
        # Bresenham-style line algorithm with thickness
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            return
        
        # Step directions
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        # Number of steps
        steps = max(dx, dy)
        
        for i in range(steps + 1):
            # Current position
            t = i / steps if steps > 0 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Draw thick point
            self._draw_thick_point(x, y, width, color)
    
    def _draw_thick_point(self, center_x: int, center_y: int, width: int, color: np.ndarray):
        """Draw a thick point (square) at the given position."""
        half_width = width // 2
        
        for dy in range(-half_width, half_width + 1):
            for dx in range(-half_width, half_width + 1):
                x = center_x + dx
                y = center_y + dy
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[y, x] = color
    
    def _carve_rectangle(self, x: int, y: int, width: int, height: int, color: np.ndarray):
        """Carve a filled rectangle."""
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + width)
        y2 = min(self.height, y + height)
        
        self.grid[y1:y2, x1:x2] = color
    
    def _postprocess_map(self, min_room_area: int, corridor_thickness: int):
        """Post-process the map to seal small rooms and connect larger ones"""
        # Convert to binary for connectivity analysis
        is_free = np.all(self.grid == self.FREE_COLOR, axis=2)
        
        # Find connected components of free space
        components = self._find_connected_components(is_free)
        
        # Separate large and small components
        large_components = []
        for component_mask in components:
            component_size = np.sum(component_mask)
            
            if component_size < min_room_area:
                # Seal small component (turn to walls)
                self.grid[component_mask] = self.WALL_COLOR
            else:
                large_components.append(component_mask)
        
        # Connect large components with corridors
        self._connect_components_with_corridors(large_components, corridor_thickness)

    
    def _find_connected_components(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """Find 4-connected components in a binary mask."""
        visited = np.zeros_like(binary_mask, dtype=bool)
        components = []
        
        for y in range(self.height):
            for x in range(self.width):
                if binary_mask[y, x] and not visited[y, x]:
                    # BFS to find component
                    component_mask = np.zeros_like(binary_mask, dtype=bool)
                    queue = deque([(x, y)])
                    visited[y, x] = True
                    component_mask[y, x] = True
                    
                    while queue:
                        cx, cy = queue.popleft()
                        
                        # Check 4-connected neighbors
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                binary_mask[ny, nx] and not visited[ny, nx]):
                                visited[ny, nx] = True
                                component_mask[ny, nx] = True
                                queue.append((nx, ny))
                    
                    components.append(component_mask)
        
        return components

    def _connect_components_with_corridors(self, components: List[np.ndarray], corridor_thickness: int):
        """Connect disconnected components with corridors."""
        if len(components) <= 1:
            return  # Already connected or no components to connect
        
        # Get centroids of each component
        centroids: list[ImagePoint] = []
        for component_mask in components:
            r_list, c_list = np.where(component_mask)
            if len(c_list) > 0:
                centroid_r = int(np.mean(c_list))
                centroid_c = int(np.mean(r_list))

                # sample a random point in the component if the centroid is not in the component
                if not component_mask[centroid_r, centroid_c]:
                    centroid = random.choice(list(zip(r_list, c_list)))
                else:
                    centroid = (centroid_r, centroid_c)

                centroids.append(ImagePoint(index=centroid))

        # Connect components using minimum spanning tree approach
        if len(centroids) > 1:
            # Connect each component to its nearest neighbor
            connected = [False] * len(components)
            connected[0] = True
            
            for _ in range(len(components) - 1):
                min_distance = float('inf')
                best_connection = None
                
                # Find closest unconnected component to any connected component
                for i, connected_i in enumerate(connected):
                    if not connected_i:
                        continue
                    for j, connected_j in enumerate(connected):
                        if connected_j:
                            continue

                        distance = centroids[i].distance_to_point(centroids[j])
                        if distance < min_distance:
                            min_distance = distance
                            best_connection = (i, j)
                
                if best_connection:
                    i, j = best_connection
                    self._carve_l_corridors(
                        centroids[i], centroids[j], corridor_thickness
                    )
                    connected[j] = True

    def _carve_l_corridors(self, p1: ImagePoint, p2: ImagePoint, corridor_thickness: int):
        """Carve a corridor connecting two components."""
        self._draw_vertical_line(p2, p1, corridor_thickness, self.FREE_COLOR)
        self._draw_horizontal_line(p1, p2, corridor_thickness, self.FREE_COLOR)

    def _find_best_connection_point(self, component_mask: np.ndarray,
                                  target_point: Tuple[int, int]) -> Tuple[int, int]:
        """Find the best point on a component's edge to connect to a target."""
        target_x, target_y = target_point
        
        # Find boundary pixels of the component
        boundary_pixels = []
        y_coords, x_coords = np.where(component_mask)
        
        for x, y in zip(x_coords, y_coords):
            # Check if this pixel is on the boundary (adjacent to non-component pixels)
            is_boundary = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not component_mask[ny, nx]):
                    is_boundary = True
                    break
            
            if is_boundary:
                boundary_pixels.append((x, y))
        
        # Find the boundary pixel closest to the target
        if boundary_pixels:
            min_distance = float('inf')
            best_point = boundary_pixels[0]
            
            for x, y in boundary_pixels:
                distance = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_point = (x, y)
            
            return best_point
        
        return None
    
    def _reached_target_density(self, target_count: int) -> bool:
        wall_mask = np.all(self.grid == self.WALL_COLOR, axis=-1)
        return np.count_nonzero(wall_mask) >= target_count
    
    def generate_value_map(self, num_value_points: int = 5, radius: int = 10, vref: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Add a point of interest to the map."""
        assert radius > 0, "Radius must be greater than 0"

        free_space_mask = np.all(self.grid == self.FREE_COLOR, axis=-1)
        freeIndices = list(np.argwhere(free_space_mask).tolist())

        random.shuffle(freeIndices)
        points_of_interest = list(map(lambda idx: ImagePoint(index=idx), freeIndices[:num_value_points]))
        value_map = np.zeros_like(self.grid, dtype=np.float64)

        # compute value for each free pixel based on distance to points of interest
        for (r, c) in freeIndices:
            if not np.all(self.grid[r, c] == self.FREE_COLOR): continue # Only consider free space
            for point in points_of_interest:
                if point.distance_to_index((r, c)) > radius: 
                    continue # point of interest is out of range
                value_map[r, c, VALUE_CHANNEL] += (radius - point.distance_to_index((r, c))) / radius * vref

        # Normalize value map to [0, 255]
        value_map = (value_map / np.max(value_map) * 255).astype(np.uint8)

        # Set the 3rd channel of free space pixels to the value map
        self.grid[free_space_mask, VALUE_CHANNEL] = value_map[free_space_mask, VALUE_CHANNEL]

        return self.grid.copy(), value_map.copy()

def save_png(map: np.ndarray, filepath: str):
    """Save the occupancy grid as a PNG image."""
    image = Image.fromarray(map)
    image.save(filepath)
    print(f"Map saved to {filepath}")


def output_map(width: int, 
               height: int, 
               plot_type: str, 
               thickness: int, 
               min_room_area: int, 
               density: float,
               seed: int | None = None,
               num_value_points: int = 10,
               radius_of_interest: int = 10,
               output_path: str | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate and save an occupancy grid map.
    
    Args:
        width: Pixel width of the output image
        height: Pixel height of the output image
        plot_type: Either 'rooms' or 'tunnels'
        thickness: Wall thickness (rooms mode) or tunnel width (tunnels mode)
        min_room_area: Minimum connected free region size
        density: Target fraction of black pixels
        seed: Random seed for reproducibility
        output_path: Path to save the output PNG
        
    Returns:
        NumPy array of shape (height, width, 3), dtype=uint8
    """
    generator = OccupancyGridGenerator(width, height, seed)
    occupancy_grid = generator.generate_map(plot_type, thickness, min_room_area, density)
    occupancy_grid, value_map = generator.generate_value_map(num_value_points, radius_of_interest, 10)
    if output_path == None:
        from datetime import datetime
        output_path = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_png(value_map, output_path.replace('.png', '_value.png'))
    save_png(occupancy_grid, output_path)

    return occupancy_grid, value_map, generator.seed


def main():
    """Command-line interface for the occupancy grid generator."""
    parser = argparse.ArgumentParser(
        description="Generate randomized occupancy grid maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--width', type=int, default=512,
                       help='Pixel width of the output image')
    parser.add_argument('--height', type=int, default=512,
                       help='Pixel height of the output image')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--plot_type', choices=['rooms', 'tunnels'], default='rooms',
                       help='Map generation mode')
    parser.add_argument('--thickness', type=int, default=4,
                       help='Wall thickness (rooms mode) or tunnel width (tunnels mode)')
    parser.add_argument('--min_room_area', type=int, default=50,
                       help='Minimum connected free region size')
    parser.add_argument('--density', type=float, default=0.45,
                       help='Target fraction of black pixels')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PNG file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.density < 0.0 or args.density > 1.0:
        raise ValueError("Density must be between 0.0 and 1.0")
    
    if args.thickness < 1:
        raise ValueError("Thickness must be at least 1")
    
    if args.min_room_area < 1:
        raise ValueError("Minimum room area must be at least 1")
    
    # Generate map
    print(f"Generating {args.plot_type} map...")
    print(f"Dimensions: {args.width}x{args.height}")
    print(f"Seed: {args.seed}")
    print(f"Thickness: {args.thickness}")
    print(f"Min room area: {args.min_room_area}")
    print(f"Target density: {args.density}")
    
    occupancy_grid, seed = output_map(
        width=args.width,
        height=args.height,
        plot_type=args.plot_type,
        thickness=args.thickness,
        min_room_area=args.min_room_area,
        seed=args.seed,
        density=args.density,
        output_path=args.output
    )
    
    # Report final statistics
    total_pixels = occupancy_grid.shape[0] * occupancy_grid.shape[1]
    black_pixels = np.sum(np.all(occupancy_grid == [0, 0, 0], axis=2))
    actual_density = black_pixels / total_pixels
    
    print(f"Actual density: {actual_density:.3f}")
    print(f"Actual seed value: {seed}")
    print(f"Generation complete!")


if __name__ == "__main__":
    main()
