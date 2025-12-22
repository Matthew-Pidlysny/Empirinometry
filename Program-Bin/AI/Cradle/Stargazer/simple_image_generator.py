#!/usr/bin/env python3
"""
SIMPLIFIED IMAGE TEST GENERATOR
===============================

Generates test images for the Quantum Visual Processor using basic numpy operations.
Creates challenging and standard test cases without external dependencies.
"""

import numpy as np
import math
import random
import os
import json

class SimpleImageGenerator:
    """Generate test images using only numpy and basic math."""
    
    def __init__(self, output_dir="test_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/challenging", exist_ok=True)
        os.makedirs(f"{output_dir}/standard", exist_ok=True)
        
    def save_image(self, image, filename):
        """Save image as numpy array (simulating image format)."""
        np.save(filename.replace('.png', '.npy'), image)
        
    def generate_challenging_images(self, count=20):  # Reduced for demo
        """Generate challenging test images."""
        print(f"Generating {count} challenging test images...")
        
        for i in range(count):
            test_type = i % 5
            
            if test_type == 0:
                image = self._refraction_challenge(i)
                info = {'type': 'refraction_challenge'}
            elif test_type == 1:
                image = self._extreme_lighting(i)
                info = {'type': 'extreme_lighting'}
            elif test_type == 2:
                image = self._complex_texture(i)
                info = {'type': 'complex_texture'}
            elif test_type == 3:
                image = self._noise_corruption(i)
                info = {'type': 'noise_corruption'}
            else:
                image = self._optical_illusion(i)
                info = {'type': 'optical_illusion'}
            
            filename = f"{self.output_dir}/challenging/test_{i:03d}_{info['type']}.npy"
            self.save_image(image, filename)
            
        print("Challenging images complete!")
        
    def generate_standard_images(self, count=20):  # Reduced for demo
        """Generate standard test images."""
        print(f"Generating {count} standard test images...")
        
        for i in range(count):
            test_type = i % 5
            
            if test_type == 0:
                image = self._simple_gradient(i)
                info = {'type': 'simple_gradient'}
            elif test_type == 1:
                image = self._basic_shapes(i)
                info = {'type': 'basic_shapes'}
            elif test_type == 2:
                image = self._checkerboard(i)
                info = {'type': 'checkerboard'}
            elif test_type == 3:
                image = self._circles(i)
                info = {'type': 'circles'}
            else:
                image = self._stripes(i)
                info = {'type': 'stripes'}
            
            filename = f"{self.output_dir}/standard/test_{i:03d}_{info['type']}.npy"
            self.save_image(image, filename)
            
        print("Standard images complete!")
        
    def _refraction_challenge(self, index):
        """Generate refraction-distorted image."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        # Base gradient
        for y in range(height):
            for x in range(width):
                image[y, x] = [x/width*255, y/height*255, (x+y)/(width+height)*255]
        
        # Add refraction distortion
        for _ in range(5):
            cx, cy = random.randint(50, 200), random.randint(50, 200)
            radius = random.randint(20, 60)
            
            for y in range(max(0, cy-radius), min(height, cy+radius)):
                for x in range(max(0, cx-radius), min(width, cx+radius)):
                    dist = math.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < radius:
                        distortion = int(10 * math.sin(dist/radius * math.pi))
                        if 0 <= x+distortion < width:
                            image[y, x] = image[y, x+distortion]
        
        return image.astype(np.uint8)
        
    def _extreme_lighting(self, index):
        """Generate extreme lighting conditions."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        # Multiple light sources
        for _ in range(3):
            lx, ly = random.randint(0, width), random.randint(0, height)
            intensity = random.uniform(0.5, 2.0)
            color = [random.uniform(0, 255) for _ in range(3)]
            
            for y in range(height):
                for x in range(width):
                    dist = math.sqrt((x-lx)**2 + (y-ly)**2)
                    if dist > 0:
                        light_val = intensity / (1 + dist*0.01)
                        image[y, x] += [c * light_val for c in color]
        
        return np.clip(image, 0, 255).astype(np.uint8)
        
    def _complex_texture(self, index):
        """Generate complex fractal texture."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        for y in range(height):
            for x in range(width):
                value = 0
                for octave in range(4):
                    freq = 0.02 * (2 ** octave)
                    amp = 1 / (2 ** octave)
                    value += amp * (math.sin(x * freq) * math.cos(y * freq))
                
                color_val = int((value + 1) * 127)
                image[y, x] = [color_val, color_val//2, 255-color_val]
        
        return image.astype(np.uint8)
        
    def _noise_corruption(self, index):
        """Generate noise-corrupted image."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        # Base pattern
        for y in range(height):
            for x in range(width):
                image[y, x] = [128, 128, 128]
        
        # Add noise
        noise = np.random.normal(0, 30, (height, width, 3))
        image = np.clip(image + noise, 0, 255)
        
        return image.astype(np.uint8)
        
    def _optical_illusion(self, index):
        """Generate optical illusion pattern."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        # Spiral pattern
        for angle in np.linspace(0, 4*math.pi, 500):
            r = 5 * math.exp(0.1 * angle)
            if r < min(width, height) / 2:
                x = int(width/2 + r * math.cos(angle))
                y = int(height/2 + r * math.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    color_val = int(128 + 50 * math.sin(angle))
                    image[y, x] = [color_val, 255-color_val, color_val//2]
        
        return image.astype(np.uint8)
        
    def _simple_gradient(self, index):
        """Simple gradient."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        for x in range(width):
            intensity = int(255 * (x / width))
            image[:, x] = [intensity, intensity//2, 255-intensity]
        
        return image.astype(np.uint8)
        
    def _basic_shapes(self, index):
        """Basic geometric shapes."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        # Background
        image[:] = [100, 100, 100]
        
        # Rectangle
        image[50:100, 50:150] = [255, 0, 0]
        
        # Circle
        cx, cy, r = 180, 180, 40
        for y in range(max(0, cy-r), min(height, cy+r)):
            for x in range(max(0, cx-r), min(width, cx+r)):
                if (x-cx)**2 + (y-cy)**2 <= r**2:
                    image[y, x] = [0, 255, 0]
        
        return image.astype(np.uint8)
        
    def _checkerboard(self, index):
        """Checkerboard pattern."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        square_size = 32
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    image[y:y+square_size, x:x+square_size] = [255, 255, 255]
                else:
                    image[y:y+square_size, x:x+square_size] = [0, 0, 0]
        
        return image.astype(np.uint8)
        
    def _circles(self, index):
        """Random circles."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        image[:] = [200, 200, 200]
        
        for _ in range(5):
            cx, cy = random.randint(20, 236), random.randint(20, 236)
            r = random.randint(10, 30)
            color = [random.randint(0, 255) for _ in range(3)]
            
            for y in range(max(0, cy-r), min(height, cy+r)):
                for x in range(max(0, cx-r), min(width, cx+r)):
                    if (x-cx)**2 + (y-cy)**2 <= r**2:
                        image[y, x] = color
        
        return image.astype(np.uint8)
        
    def _stripes(self, index):
        """Stripe pattern."""
        height, width = 256, 256
        image = np.zeros((height, width, 3))
        
        stripe_width = 20
        for x in range(0, width, stripe_width*2):
            image[:, x:x+stripe_width] = [255, 255, 255]
            image[:, x+stripe_width:x+stripe_width*2] = [0, 0, 0]
        
        return image.astype(np.uint8)

def main():
    """Generate test images."""
    print("SIMPLIFIED IMAGE TEST GENERATOR")
    print("=" * 40)
    
    generator = SimpleImageGenerator()
    generator.generate_challenging_images(20)
    generator.generate_standard_images(20)
    
    print("Test image generation complete!")
    print("Generated 40 test images (20 challenging, 20 standard)")

if __name__ == "__main__":
    main()