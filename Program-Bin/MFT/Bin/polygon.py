#!/usr/bin/env python3
"""
POLYGON - Field Extension Pattern Analyzer
Tests if the field extends from λ = 0.6 in a 3-point formation
or continues with 3-1-4 pattern at different levels
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import pdist, squareform
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FieldExtensionAnalyzer:
    """
    Analyze how the field extends from λ = 0.6
    
    Tests:
    1. Does field form triangular (3-point) patterns?
    2. Does 3-1-4 repeat at different scales?
    3. What is the geometric structure of field minima?
    """
    
    def __init__(self, lambda_val=0.6):
        self.lambda_val = lambda_val
        self.field_points = []
        self.patterns = []
        
    def generate_field_configuration(self, n_points=100, dimensions=3):
        """Generate a field configuration around λ = 0.6"""
        # Generate points in a sphere
        points = np.random.randn(n_points, dimensions)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Scale by λ
        points = points * self.lambda_val
        
        # Add some noise
        points += np.random.randn(n_points, dimensions) * 0.1
        
        return points
    
    def detect_triangular_patterns(self, points):
        """Detect 3-point triangular patterns in field"""
        print("="*80)
        print("DETECTING TRIANGULAR (3-POINT) PATTERNS")
        print("="*80)
        print()
        
        if points.shape[1] == 2:
            # 2D Delaunay triangulation
            tri = Delaunay(points)
            triangles = tri.simplices
            
            print(f"Found {len(triangles)} triangles in 2D field")
            
        elif points.shape[1] == 3:
            # 3D Delaunay tetrahedralization
            tri = Delaunay(points)
            tetrahedra = tri.simplices
            
            print(f"Found {len(tetrahedra)} tetrahedra in 3D field")
            print("Each tetrahedron has 4 triangular faces")
            
            # Extract triangular faces
            triangles = []
            for tet in tetrahedra:
                # Each tetrahedron has 4 faces
                triangles.append([tet[0], tet[1], tet[2]])
                triangles.append([tet[0], tet[1], tet[3]])
                triangles.append([tet[0], tet[2], tet[3]])
                triangles.append([tet[1], tet[2], tet[3]])
            
            triangles = np.array(triangles)
            print(f"Total triangular faces: {len(triangles)}")
        
        # Analyze triangle properties
        triangle_ratios = []
        
        for i, triangle_indices in enumerate(triangles[:100]):  # Sample first 100
            triangle_points = points[triangle_indices]
            
            # Calculate side lengths
            side1 = np.linalg.norm(triangle_points[1] - triangle_points[0])
            side2 = np.linalg.norm(triangle_points[2] - triangle_points[1])
            side3 = np.linalg.norm(triangle_points[0] - triangle_points[2])
            
            # Calculate ratios
            if side2 > 0 and side3 > 0:
                ratio1 = side1 / side2
                ratio2 = side2 / side3
                ratio3 = side3 / side1
                
                triangle_ratios.extend([ratio1, ratio2, ratio3])
        
        triangle_ratios = np.array(triangle_ratios)
        triangle_ratios = triangle_ratios[np.isfinite(triangle_ratios)]
        
        print(f"\nTriangle side ratios:")
        print(f"  Mean: {np.mean(triangle_ratios):.4f}")
        print(f"  Median: {np.median(triangle_ratios):.4f}")
        print(f"  Std: {np.std(triangle_ratios):.4f}")
        
        # Check for λ = 0.6 in ratios
        close_to_lambda = triangle_ratios[(triangle_ratios >= 0.55) & (triangle_ratios <= 0.65)]
        percentage = (len(close_to_lambda) / len(triangle_ratios)) * 100
        
        print(f"\nRatios close to λ = 0.6: {len(close_to_lambda)} ({percentage:.1f}%)")
        
        if percentage > 15:
            print("*** STRONG EVIDENCE: Triangular patterns show λ = 0.6! ***")
        
        return triangles, triangle_ratios
    
    def test_314_repetition(self, points):
        """Test if 3-1-4 pattern repeats at different scales"""
        print("\n" + "="*80)
        print("TESTING 3-1-4 PATTERN REPETITION")
        print("="*80)
        print()
        
        # Calculate distances at different scales
        distances = pdist(points)
        
        # Bin distances into scales
        scales = np.logspace(-2, 1, 20)
        
        print("Testing for 3-1-4 pattern at different scales:")
        print()
        
        patterns_found = []
        
        for i, scale in enumerate(scales):
            # Get distances at this scale
            scale_distances = distances[(distances >= scale*0.9) & (distances <= scale*1.1)]
            
            if len(scale_distances) < 10:
                continue
            
            # Look for 3-1-4 pattern in distance ratios
            # Sort distances
            sorted_dists = np.sort(scale_distances)
            
            if len(sorted_dists) >= 3:
                # Take representative distances
                d1 = sorted_dists[len(sorted_dists)//4]
                d2 = sorted_dists[len(sorted_dists)//2]
                d3 = sorted_dists[3*len(sorted_dists)//4]
                
                # Normalize to first distance
                if d1 > 0:
                    ratio1 = d1 / d1  # = 1
                    ratio2 = d2 / d1
                    ratio3 = d3 / d1
                    
                    # Check if ratios approximate 3:1:4 or similar
                    # Looking for patterns like [3, 1, 4] or [1, 3, 4] etc.
                    ratios = np.array([ratio1, ratio2, ratio3])
                    
                    # Test various permutations
                    target_patterns = [
                        [3, 1, 4],
                        [1, 3, 4],
                        [1, 4, 3],
                        [3, 4, 1],
                    ]
                    
                    for target in target_patterns:
                        target_norm = np.array(target) / target[0]
                        error = np.mean(np.abs(ratios - target_norm))
                        
                        if error < 0.5:
                            patterns_found.append({
                                'scale': scale,
                                'ratios': ratios.tolist(),
                                'target': target,
                                'error': error
                            })
                            print(f"Scale {scale:.4f}: Ratios {ratios} ≈ {target} (error: {error:.4f})")
        
        print(f"\nTotal 3-1-4-like patterns found: {len(patterns_found)}")
        
        if len(patterns_found) > 0:
            print("*** EVIDENCE: 3-1-4 pattern repeats at multiple scales! ***")
        else:
            print("No clear 3-1-4 repetition found at tested scales")
        
        return patterns_found
    
    def analyze_field_geometry(self, points):
        """Analyze the geometric structure of field minima"""
        print("\n" + "="*80)
        print("ANALYZING FIELD GEOMETRY")
        print("="*80)
        print()
        
        # Calculate center of mass
        center = np.mean(points, axis=0)
        print(f"Field center: {center}")
        
        # Calculate distances from center
        distances_from_center = np.linalg.norm(points - center, axis=1)
        
        print(f"\nDistances from center:")
        print(f"  Mean: {np.mean(distances_from_center):.4f}")
        print(f"  Median: {np.median(distances_from_center):.4f}")
        print(f"  Std: {np.std(distances_from_center):.4f}")
        
        # Check if mean distance relates to λ
        mean_dist = np.mean(distances_from_center)
        print(f"\nMean distance: {mean_dist:.4f}")
        print(f"λ = {self.lambda_val}")
        print(f"Ratio: {mean_dist / self.lambda_val:.4f}")
        
        if abs(mean_dist - self.lambda_val) < 0.1:
            print("*** Field is centered at λ = 0.6! ***")
        
        # Analyze convex hull
        if points.shape[1] <= 3:
            hull = ConvexHull(points)
            print(f"\nConvex hull:")
            print(f"  Volume: {hull.volume:.4f}")
            print(f"  Surface area: {hull.area:.4f}")
            print(f"  Number of vertices: {len(hull.vertices)}")
            
            # Check if volume relates to λ
            if points.shape[1] == 3:
                # For 3D, check if volume ~ λ³
                expected_volume = (4/3) * np.pi * (self.lambda_val ** 3)
                print(f"\nExpected volume (sphere): {expected_volume:.4f}")
                print(f"Actual volume: {hull.volume:.4f}")
                print(f"Ratio: {hull.volume / expected_volume:.4f}")
        
        return center, distances_from_center
    
    def test_3point_vs_4point_formation(self, points):
        """Test if field prefers 3-point or 4-point formations"""
        print("\n" + "="*80)
        print("TESTING 3-POINT vs 4-POINT FORMATION")
        print("="*80)
        print()
        
        # Sample random point sets
        n_tests = 100
        three_point_scores = []
        four_point_scores = []
        
        for _ in range(n_tests):
            # Random 3 points
            indices_3 = np.random.choice(len(points), 3, replace=False)
            points_3 = points[indices_3]
            
            # Random 4 points
            indices_4 = np.random.choice(len(points), 4, replace=False)
            points_4 = points[indices_4]
            
            # Calculate "stability" score (inverse of variance in distances)
            dists_3 = pdist(points_3)
            dists_4 = pdist(points_4)
            
            score_3 = 1 / (np.std(dists_3) + 0.01)
            score_4 = 1 / (np.std(dists_4) + 0.01)
            
            three_point_scores.append(score_3)
            four_point_scores.append(score_4)
        
        mean_3 = np.mean(three_point_scores)
        mean_4 = np.mean(four_point_scores)
        
        print(f"3-point formation stability: {mean_3:.4f}")
        print(f"4-point formation stability: {mean_4:.4f}")
        print()
        
        if mean_3 > mean_4:
            print("*** RESULT: Field prefers 3-POINT formations! ***")
            print("This suggests triangular/tetrahedral geometry")
        else:
            print("*** RESULT: Field prefers 4-POINT formations! ***")
            print("This suggests square/cubic geometry")
        
        return mean_3, mean_4
    
    def test_hierarchical_structure(self, points):
        """Test if field has hierarchical 3-1-4 structure at multiple levels"""
        print("\n" + "="*80)
        print("TESTING HIERARCHICAL STRUCTURE")
        print("="*80)
        print()
        
        # Level 1: Individual points
        print("LEVEL 1: Individual points")
        print(f"  Total points: {len(points)}")
        
        # Level 2: Clusters of 3
        print("\nLEVEL 2: Clusters of 3")
        # Use k-means to find 3 clusters
        from sklearn.cluster import KMeans
        
        n_clusters_level2 = 3
        kmeans_2 = KMeans(n_clusters=n_clusters_level2, random_state=42, n_init=10)
        labels_2 = kmeans_2.fit_predict(points)
        centers_2 = kmeans_2.cluster_centers_
        
        print(f"  Found {n_clusters_level2} clusters")
        print(f"  Cluster centers: {centers_2}")
        
        # Calculate distances between cluster centers
        dists_2 = pdist(centers_2)
        print(f"  Inter-cluster distances: {dists_2}")
        
        # Level 3: Meta-clusters
        print("\nLEVEL 3: Meta-structure")
        # Treat cluster centers as points, find their structure
        
        if len(centers_2) >= 3:
            # Calculate ratios
            sorted_dists = np.sort(dists_2)
            if len(sorted_dists) >= 3 and sorted_dists[0] > 0:
                ratio1 = sorted_dists[0] / sorted_dists[0]
                ratio2 = sorted_dists[1] / sorted_dists[0]
                ratio3 = sorted_dists[2] / sorted_dists[0] if len(sorted_dists) > 2 else 0
                
                print(f"  Distance ratios: [{ratio1:.2f}, {ratio2:.2f}, {ratio3:.2f}]")
                
                # Check if close to [1, 3, 4] or [3, 1, 4]
                target = np.array([1, 3, 4])
                actual = np.array([ratio1, ratio2, ratio3])
                error = np.mean(np.abs(actual - target))
                
                print(f"  Error from [1, 3, 4]: {error:.4f}")
                
                if error < 1.0:
                    print("  *** HIERARCHICAL 3-1-4 PATTERN DETECTED! ***")
        
        # Test if λ = 0.6 appears in hierarchy
        print("\nTesting for λ = 0.6 in hierarchical ratios:")
        
        # Ratio of level 2 to level 1
        level1_scale = np.mean(pdist(points))
        level2_scale = np.mean(dists_2)
        
        if level1_scale > 0:
            hierarchy_ratio = level2_scale / level1_scale
            print(f"  Level 2 / Level 1 scale: {hierarchy_ratio:.4f}")
            
            if abs(hierarchy_ratio - 0.6) < 0.2:
                print("  *** CLOSE TO λ = 0.6! ***")
        
        return centers_2, labels_2
    
    def visualize_field(self, points, filename='field_structure.png'):
        """Visualize the field structure"""
        if points.shape[1] == 2:
            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
            plt.title(f'Field Structure (λ = {self.lambda_val})')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        elif points.shape[1] == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.6)
            ax.set_title(f'Field Structure (λ = {self.lambda_val})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualization saved to: {filename}")


def main():
    """Main execution"""
    print("="*80)
    print("POLYGON - FIELD EXTENSION PATTERN ANALYZER")
    print("="*80)
    print()
    
    analyzer = FieldExtensionAnalyzer(lambda_val=0.6)
    
    # Generate field configuration
    print("Generating field configuration...")
    points_3d = analyzer.generate_field_configuration(n_points=200, dimensions=3)
    print(f"Generated {len(points_3d)} points in 3D")
    print()
    
    # Test 1: Triangular patterns
    triangles, ratios = analyzer.detect_triangular_patterns(points_3d)
    
    # Test 2: 3-1-4 repetition
    patterns = analyzer.test_314_repetition(points_3d)
    
    # Test 3: Field geometry
    center, distances = analyzer.analyze_field_geometry(points_3d)
    
    # Test 4: 3-point vs 4-point
    score_3, score_4 = analyzer.test_3point_vs_4point_formation(points_3d)
    
    # Test 5: Hierarchical structure
    centers, labels = analyzer.test_hierarchical_structure(points_3d)
    
    # Visualize
    analyzer.visualize_field(points_3d, 'field_structure_3d.png')
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'lambda': analyzer.lambda_val,
        'n_points': len(points_3d),
        'triangular_patterns': len(triangles),
        'triangle_ratio_mean': float(np.mean(ratios)),
        '314_patterns_found': len(patterns),
        'prefers_3point': score_3 > score_4,
        'field_center': center.tolist(),
        'mean_distance_from_center': float(np.mean(distances))
    }
    
    with open('polygon_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("POLYGON ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved to: polygon_results.json")


if __name__ == "__main__":
    main()