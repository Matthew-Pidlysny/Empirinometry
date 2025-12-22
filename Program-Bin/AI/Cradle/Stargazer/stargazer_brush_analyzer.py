#!/usr/bin/env python3
"""
STARGAZER BRUSH STROKE ANALYZER
================================

Advanced AI brush stroke analysis and optimization system.
Analyzes AI generation patterns in real-time to improve artistic output.

Features:
- Real-time brush stroke pattern detection
- AI technique analysis and optimization
- Dynamic parameter adjustment based on analysis
- Self-improving generation pipeline
- Artistic quality assessment

Author: SuperNinja AI Research Division
Purpose: AI brush stroke analysis for enhanced artistic generation
"""

import numpy as np
import cv2
import math
from scipy import ndimage, signal
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time

class StargazerBrushAnalyzer:
    """
    Advanced brush stroke analysis system for AI artistry optimization.
    """
    
    def __init__(self):
        self.name = "Stargazer Brush Analyzer"
        self.stroke_patterns = []
        self.analysis_history = []
        self.optimization_params = {
            'stroke_smoothness': 0.7,
            'pressure_variation': 0.5,
            'color_bleeding': 0.3,
            'texture_complexity': 0.6
        }
        
        # Pattern recognition parameters
        self.min_stroke_length = 5
        self.stroke_threshold = 0.1
        
    def analyze_brush_strokes(self, image, generation_metadata=None):
        """
        Analyze brush strokes in generated image.
        
        Args:
            image: Generated image to analyze
            generation_metadata: Metadata about how the image was generated
        
        Returns:
            Analysis results and optimization suggestions
        """
        start_time = time.time()
        
        # Convert to grayscale for stroke detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges (potential brush strokes)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find stroke patterns
        stroke_patterns = self._detect_stroke_patterns(edges)
        
        # Analyze stroke characteristics
        stroke_analysis = self._analyze_stroke_characteristics(stroke_patterns, gray)
        
        # Analyze color transitions
        color_analysis = self._analyze_color_transitions(image)
        
        # Analyze texture patterns
        texture_analysis = self._analyze_texture_patterns(image)
        
        # Generate optimization suggestions
        optimizations = self._generate_optimization_suggestions(
            stroke_analysis, color_analysis, texture_analysis
        )
        
        # Update analysis history
        analysis_result = {
            'timestamp': time.time(),
            'processing_time': time.time() - start_time,
            'stroke_patterns': stroke_analysis,
            'color_transitions': color_analysis,
            'texture_patterns': texture_analysis,
            'optimizations': optimizations,
            'generation_metadata': generation_metadata
        }
        
        self.analysis_history.append(analysis_result)
        self._update_optimization_params(optimizations)
        
        return analysis_result
    
    def _detect_stroke_patterns(self, edges):
        """Detect individual brush stroke patterns."""
        # Find contours in edge detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stroke_patterns = []
        
        for contour in contours:
            # Filter out very small contours
            if cv2.contourArea(contour) < self.min_stroke_length:
                continue
            
            # Analyze contour shape
            stroke_info = {
                'contour': contour,
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True),
                'aspect_ratio': self._calculate_aspect_ratio(contour),
                'curvature': self._calculate_curvature(contour),
                'direction': self._calculate_stroke_direction(contour)
            }
            
            stroke_patterns.append(stroke_info)
        
        return stroke_patterns
    
    def _calculate_aspect_ratio(self, contour):
        """Calculate aspect ratio of stroke contour."""
        x, y, w, h = cv2.boundingRect(contour)
        return max(w, h) / max(min(w, h), 1)
    
    def _calculate_curvature(self, contour):
        """Calculate average curvature of stroke."""
        if len(contour) < 4:
            return 0
        
        # Simplify contour for curvature calculation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            return 0
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            p3 = approx[(i + 2) % len(approx)][0]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    def _calculate_stroke_direction(self, contour):
        """Calculate primary direction of stroke."""
        if len(contour) < 2:
            return 0
        
        # Use PCA to find primary direction
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 2:
            return 0
        
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # First principal component gives main direction
        direction = np.arctan2(pca.components_[0][1], pca.components_[0][0])
        
        return direction
    
    def _analyze_stroke_characteristics(self, stroke_patterns, gray_image):
        """Analyze characteristics of detected strokes."""
        if not stroke_patterns:
            return {
                'stroke_count': 0,
                'average_length': 0,
                'complexity': 0,
                'direction_variance': 0,
                'quality_score': 0
            }
        
        # Calculate various metrics
        areas = [s['area'] for s in stroke_patterns]
        perimeters = [s['perimeter'] for s in stroke_patterns]
        curvatures = [s['curvature'] for s in stroke_patterns]
        directions = [s['direction'] for s in stroke_patterns]
        
        # Calculate stroke characteristics
        analysis = {
            'stroke_count': len(stroke_patterns),
            'average_area': np.mean(areas),
            'average_perimeter': np.mean(perimeters),
            'average_curvature': np.mean(curvatures),
            'direction_variance': np.var(directions),
            'complexity': np.std(curvatures) + np.std(areas) / np.mean(areas) if areas else 0
        }
        
        # Calculate quality score based on stroke characteristics
        # Good strokes have moderate complexity and balanced directions
        ideal_complexity = 0.5
        complexity_score = 1 - abs(analysis['complexity'] - ideal_complexity)
        direction_score = 1 - min(analysis['direction_variance'], np.pi) / np.pi
        
        analysis['quality_score'] = (complexity_score * 0.6 + direction_score * 0.4)
        
        return analysis
    
    def _analyze_color_transitions(self, image):
        """Analyze color transitions and blending."""
        if len(image.shape) != 3:
            return {'smoothness': 0, 'gradient_quality': 0}
        
        # Calculate gradients for each color channel
        gradients = []
        for i in range(3):
            grad_x = cv2.Sobel(image[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(gradient_magnitude)
        
        # Average gradient across channels
        avg_gradient = np.mean(gradients, axis=0)
        
        # Analyze gradient characteristics
        smoothness = 1 - np.mean(avg_gradient) / 255  # Lower gradients = smoother
        gradient_quality = 1 - np.std(avg_gradient) / 255  # Consistent gradients = better quality
        
        return {
            'smoothness': float(smoothness),
            'gradient_quality': float(gradient_quality),
            'average_gradient': float(np.mean(avg_gradient)),
            'gradient_variance': float(np.var(avg_gradient))
        }
    
    def _analyze_texture_patterns(self, image):
        """Analyze texture patterns and complexity."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Use Local Binary Pattern for texture analysis
        lbp = self._calculate_lbp(gray)
        
        # Calculate texture characteristics
        texture_variance = np.var(lbp)
        texture_uniformity = self._calculate_texture_uniformity(lbp)
        
        # Analyze texture at different scales
        multi_scale_analysis = []
        for scale in [1, 2, 4]:
            scaled = cv2.resize(gray, None, fx=1/scale, fy=1/scale)
            scaled_lbp = self._calculate_lbp(scaled)
            multi_scale_analysis.append(np.var(scaled_lbp))
        
        return {
            'texture_complexity': float(texture_variance),
            'texture_uniformity': float(texture_uniformity),
            'multi_scale_complexity': multi_scale_analysis,
            'quality_score': float(min(1.0, texture_uniformity * texture_variance / 100))
        }
    
    def _calculate_lbp(self, image):
        """Calculate Local Binary Pattern for texture analysis."""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                # Calculate LBP
                binary = [1 if n >= center else 0 for n in neighbors]
                lbp_value = sum(bit << (7-i) for i, bit in enumerate(binary))
                lbp[i, j] = lbp_value
        
        return lbp
    
    def _calculate_texture_uniformity(self, lbp):
        """Calculate texture uniformity metric."""
        hist, _ = np.histogram(lbp, bins=256)
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate uniformity (inverse of entropy)
        uniformity = -np.sum(hist * np.log(hist + 1e-10))
        
        return uniformity
    
    def _generate_optimization_suggestions(self, stroke_analysis, color_analysis, texture_analysis):
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        params_to_adjust = {}
        
        # Analyze stroke quality
        if stroke_analysis.get('quality_score', 0) < 0.5:
            suggestions.append("Improve stroke smoothness and direction consistency")
            params_to_adjust['stroke_smoothness'] = min(1.0, self.optimization_params['stroke_smoothness'] + 0.1)
        
        # Analyze color transitions
        if color_analysis['smoothness'] < 0.3:
            suggestions.append("Reduce harsh color transitions")
            params_to_adjust['color_bleeding'] = min(1.0, self.optimization_params['color_bleeding'] + 0.1)
        
        # Analyze texture
        if texture_analysis['texture_complexity'] < 20:
            suggestions.append("Increase texture complexity for more artistic detail")
            params_to_adjust['texture_complexity'] = min(1.0, self.optimization_params['texture_complexity'] + 0.1)
        elif texture_analysis['texture_complexity'] > 100:
            suggestions.append("Reduce texture complexity to avoid noise")
            params_to_adjust['texture_complexity'] = max(0.1, self.optimization_params['texture_complexity'] - 0.1)
        
        # Pressure variation adjustments
        if stroke_analysis.get('direction_variance', 0) > 2:
            suggestions.append("Add more pressure variation for dynamic strokes")
            params_to_adjust['pressure_variation'] = min(1.0, self.optimization_params['pressure_variation'] + 0.1)
        
        return {
            'suggestions': suggestions,
            'parameter_adjustments': params_to_adjust,
            'overall_quality': self._calculate_overall_quality(
                stroke_analysis, color_analysis, texture_analysis
            )
        }
    
    def _calculate_overall_quality(self, stroke_analysis, color_analysis, texture_analysis):
        """Calculate overall quality score."""
        stroke_score = stroke_analysis.get('quality_score', 0)
        color_score = (color_analysis['smoothness'] + color_analysis['gradient_quality']) / 2
        texture_score = texture_analysis.get('quality_score', 0)
        
        overall = (stroke_score * 0.4 + color_score * 0.3 + texture_score * 0.3)
        
        return float(overall)
    
    def _update_optimization_params(self, optimizations):
        """Update optimization parameters based on analysis."""
        params = optimizations['parameter_adjustments']
        
        for param, value in params.items():
            if param in self.optimization_params:
                # Gradual adjustment to avoid sudden changes
                current = self.optimization_params[param]
                self.optimization_params[param] = current * 0.7 + value * 0.3
    
    def get_optimization_summary(self):
        """Get summary of current optimization state."""
        return {
            'current_parameters': self.optimization_params.copy(),
            'analysis_count': len(self.analysis_history),
            'average_quality': np.mean([a['optimizations']['overall_quality'] 
                                       for a in self.analysis_history]) if self.analysis_history else 0,
            'recent_trends': self._analyze_recent_trends()
        }
    
    def _analyze_recent_trends(self):
        """Analyze recent trends in analysis history."""
        if len(self.analysis_history) < 5:
            return "Insufficient data for trend analysis"
        
        recent_analyses = self.analysis_history[-5:]
        quality_trend = [a['optimizations']['overall_quality'] for a in recent_analyses]
        
        if quality_trend[-1] > quality_trend[0]:
            return "Improving quality trend detected"
        elif quality_trend[-1] < quality_trend[0]:
            return "Quality degradation detected - adjustments needed"
        else:
            return "Stable quality maintained"
    
    def apply_optimizations_to_generation(self, base_params):
        """Apply current optimizations to generation parameters."""
        optimized_params = base_params.copy()
        
        # Apply brush stroke optimizations
        if 'stroke_parameters' in optimized_params:
            stroke_params = optimized_params['stroke_parameters']
            stroke_params['smoothness'] *= (1 + self.optimization_params['stroke_smoothness'])
            stroke_params['pressure_variation'] *= (1 + self.optimization_params['pressure_variation'])
        
        # Apply color optimizations
        if 'color_parameters' in optimized_params:
            color_params = optimized_params['color_parameters']
            color_params['bleeding'] *= (1 + self.optimization_params['color_bleeding'])
        
        # Apply texture optimizations
        if 'texture_parameters' in optimized_params:
            texture_params = optimized_params['texture_parameters']
            texture_params['complexity'] *= (1 + self.optimization_params['texture_complexity'])
        
        return optimized_params