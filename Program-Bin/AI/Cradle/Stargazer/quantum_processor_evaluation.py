#!/usr/bin/env python3
"""
QUANTUM PROCESSOR EVALUATION SYSTEM
===================================

Comprehensive testing and evaluation of the Quantum Visual Processor
against challenging and standard test cases. Includes performance metrics,
quality assessment, and bug detection.

Author: SuperNinja AI Research Division
Purpose: Advanced image processing evaluation and validation
"""

import numpy as np
import time
import os
import json
from quantum_visual_processor import QuantumVisualProcessor

class ProcessorEvaluation:
    """Evaluate Quantum Visual Processor performance."""
    
    def __init__(self, test_dir="test_images"):
        self.processor = QuantumVisualProcessor()
        self.test_dir = test_dir
        self.results = {
            'challenging_tests': [],
            'standard_tests': [],
            'performance_metrics': {},
            'quality_scores': {},
            'bugs_detected': []
        }
        
    def load_test_images(self):
        """Load all test images."""
        images = {
            'challenging': [],
            'standard': []
        }
        
        # Load challenging images
        challenging_dir = f"{self.test_dir}/challenging"
        if os.path.exists(challenging_dir):
            for filename in os.listdir(challenging_dir):
                if filename.endswith('.npy'):
                    image = np.load(os.path.join(challenging_dir, filename))
                    images['challenging'].append({
                        'filename': filename,
                        'image': image
                    })
        
        # Load standard images
        standard_dir = f"{self.test_dir}/standard"
        if os.path.exists(standard_dir):
            for filename in os.listdir(standard_dir):
                if filename.endswith('.npy'):
                    image = np.load(os.path.join(standard_dir, filename))
                    images['standard'].append({
                        'filename': filename,
                        'image': image
                    })
        
        return images
        
    def evaluate_image_processing(self, image_data, test_type):
        """Evaluate processing of single image."""
        image = image_data['image']
        filename = image_data['filename']
        
        print(f"Processing {test_type} image: {filename}")
        
        # Measure processing time
        start_time = time.time()
        
        try:
            # Process image
            processed_image, analysis = self.processor.process_image_complete(image)
            
            processing_time = time.time() - start_time
            
            # Evaluate quality metrics
            quality_metrics = self._calculate_quality_metrics(image, processed_image)
            
            # Check for bugs/errors
            bugs = self._detect_processing_bugs(image, processed_image, analysis)
            
            result = {
                'filename': filename,
                'processing_time': processing_time,
                'quality_metrics': quality_metrics,
                'analysis': analysis,
                'bugs': bugs,
                'success': True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = {
                'filename': filename,
                'processing_time': processing_time,
                'error': str(e),
                'success': False
            }
            
        return result
        
    def _calculate_quality_metrics(self, original, processed):
        """Calculate image quality metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['original_mean'] = np.mean(original)
        metrics['processed_mean'] = np.mean(processed)
        metrics['original_std'] = np.std(original)
        metrics['processed_std'] = np.std(processed)
        
        # Noise reduction (if processed has lower std)
        noise_reduction = (metrics['original_std'] - metrics['processed_std']) / metrics['original_std'] if metrics['original_std'] > 0 else 0
        metrics['noise_reduction_percent'] = max(0, noise_reduction * 100)
        
        # Contrast improvement
        original_contrast = metrics['original_std'] / (metrics['original_mean'] + 1e-6)
        processed_contrast = metrics['processed_std'] / (metrics['processed_mean'] + 1e-6)
        contrast_improvement = (processed_contrast - original_contrast) / (abs(original_contrast) + 1e-6)
        metrics['contrast_improvement_percent'] = contrast_improvement * 100
        
        # Sharpness estimate (using Laplacian variance)
        original_laplacian = np.var(np.array([[original[i, j, 0] + original[i, j, 1] + original[i, j, 2]] 
                                             for i in range(1, original.shape[0]-1) 
                                             for j in range(1, original.shape[1]-1)]))
        processed_laplacian = np.var(np.array([[processed[i, j, 0] + processed[i, j, 1] + processed[i, j, 2]] 
                                              for i in range(1, processed.shape[0]-1) 
                                              for j in range(1, processed.shape[1]-1)]))
        
        sharpness_change = (processed_laplacian - original_laplacian) / (abs(original_laplacian) + 1e-6)
        metrics['sharpness_change_percent'] = sharpness_change * 100
        
        # Overall quality score
        metrics['overall_quality_score'] = (
            0.3 * min(100, max(0, metrics['noise_reduction_percent'])) +
            0.3 * min(100, max(0, metrics['contrast_improvement_percent'])) +
            0.4 * min(100, max(0, metrics['sharpness_change_percent']))
        )
        
        return metrics
        
    def _detect_processing_bugs(self, original, processed, analysis):
        """Detect common processing bugs and issues."""
        bugs = []
        
        # Check for dimension changes
        if original.shape != processed.shape:
            bugs.append(f"Dimension changed: {original.shape} -> {processed.shape}")
        
        # Check for black or white output
        if np.mean(processed) < 5:
            bugs.append("Output is mostly black")
        elif np.mean(processed) > 250:
            bugs.append("Output is mostly white")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(processed)):
            bugs.append("Output contains NaN values")
        if np.any(np.isinf(processed)):
            bugs.append("Output contains infinite values")
        
        # Check for clipping artifacts
        clipped_pixels = np.sum((processed == 0) | (processed == 255))
        total_pixels = processed.size
        clipping_ratio = clipped_pixels / total_pixels
        if clipping_ratio > 0.5:
            bugs.append(f"High clipping detected: {clipping_ratio:.2%}")
        
        # Check analysis completeness
        if analysis:
            required_keys = ['material_properties', 'curvature_analysis', 'texture_features']
            for key in required_keys:
                if key not in analysis:
                    bugs.append(f"Missing analysis component: {key}")
        else:
            bugs.append("No analysis results returned")
        
        return bugs
        
    def run_comprehensive_evaluation(self):
        """Run complete evaluation on all test images."""
        print("Starting comprehensive evaluation of Quantum Visual Processor...")
        print("=" * 70)
        
        # Load test images
        test_images = self.load_test_images()
        
        print(f"Loaded {len(test_images['challenging'])} challenging images")
        print(f"Loaded {len(test_images['standard'])} standard images")
        print()
        
        # Evaluate challenging images
        print("Evaluating challenging test cases...")
        challenging_results = []
        for i, image_data in enumerate(test_images['challenging']):
            result = self.evaluate_image_processing(image_data, 'challenging')
            challenging_results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_images['challenging'])} challenging images")
        
        # Evaluate standard images
        print("\nEvaluating standard test cases...")
        standard_results = []
        for i, image_data in enumerate(test_images['standard']):
            result = self.evaluate_image_processing(image_data, 'standard')
            standard_results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_images['standard'])} standard images")
        
        # Calculate aggregate metrics
        self.results['challenging_tests'] = challenging_results
        self.results['standard_tests'] = standard_results
        self.results['performance_metrics'] = self._calculate_aggregate_metrics()
        self.results['quality_scores'] = self._calculate_quality_scores()
        self.results['bugs_detected'] = self._summarize_bugs()
        
        print("\nEvaluation complete!")
        return self.results
        
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate performance metrics."""
        metrics = {}
        
        # Success rates
        challenging_success = sum(1 for r in self.results['challenging_tests'] if r['success'])
        standard_success = sum(1 for r in self.results['standard_tests'] if r['success'])
        
        metrics['challenging_success_rate'] = challenging_success / len(self.results['challenging_tests']) if self.results['challenging_tests'] else 0
        metrics['standard_success_rate'] = standard_success / len(self.results['standard_tests']) if self.results['standard_tests'] else 0
        metrics['overall_success_rate'] = (challenging_success + standard_success) / (len(self.results['challenging_tests']) + len(self.results['standard_tests']))
        
        # Processing times
        challenging_times = [r['processing_time'] for r in self.results['challenging_tests'] if r['success']]
        standard_times = [r['processing_time'] for r in self.results['standard_tests'] if r['success']]
        
        if challenging_times:
            metrics['challenging_avg_time'] = np.mean(challenging_times)
            metrics['challenging_max_time'] = np.max(challenging_times)
        if standard_times:
            metrics['standard_avg_time'] = np.mean(standard_times)
            metrics['standard_max_time'] = np.max(standard_times)
        
        return metrics
        
    def _calculate_quality_scores(self):
        """Calculate average quality scores."""
        scores = {}
        
        challenging_scores = [r['quality_metrics']['overall_quality_score'] 
                            for r in self.results['challenging_tests'] 
                            if r['success'] and 'quality_metrics' in r]
        standard_scores = [r['quality_metrics']['overall_quality_score'] 
                         for r in self.results['standard_tests'] 
                         if r['success'] and 'quality_metrics' in r]
        
        if challenging_scores:
            scores['challenging_avg_quality'] = np.mean(challenging_scores)
            scores['challenging_min_quality'] = np.min(challenging_scores)
            scores['challenging_max_quality'] = np.max(challenging_scores)
        
        if standard_scores:
            scores['standard_avg_quality'] = np.mean(standard_scores)
            scores['standard_min_quality'] = np.min(standard_scores)
            scores['standard_max_quality'] = np.max(standard_scores)
        
        return scores
        
    def _summarize_bugs(self):
        """Summarize all detected bugs."""
        bug_summary = {
            'total_bugs': 0,
            'bug_types': {},
            'affected_images': []
        }
        
        all_results = self.results['challenging_tests'] + self.results['standard_tests']
        
        for result in all_results:
            if not result['success']:
                bug_summary['total_bugs'] += 1
                if 'error' in result:
                    error_type = result['error'].split(':')[0]
                    bug_summary['bug_types'][error_type] = bug_summary['bug_types'].get(error_type, 0) + 1
                bug_summary['affected_images'].append(result['filename'])
            elif 'bugs' in result and result['bugs']:
                bug_summary['total_bugs'] += len(result['bugs'])
                for bug in result['bugs']:
                    bug_type = bug.split(':')[0]
                    bug_summary['bug_types'][bug_type] = bug_summary['bug_types'].get(bug_type, 0) + 1
                bug_summary['affected_images'].append(result['filename'])
        
        return bug_summary
        
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        report = {
            'evaluation_summary': {
                'total_images_tested': len(self.results['challenging_tests']) + len(self.results['standard_tests']),
                'challenging_images': len(self.results['challenging_tests']),
                'standard_images': len(self.results['standard_tests']),
                'evaluation_date': '2024-12-22'
            },
            'performance_metrics': self.results['performance_metrics'],
            'quality_scores': self.results['quality_scores'],
            'bug_analysis': self.results['bugs_detected'],
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self):
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Success rate recommendations
        if self.results['performance_metrics']['challenging_success_rate'] < 0.8:
            recommendations.append("Improve handling of challenging test cases - success rate below 80%")
        
        if self.results['performance_metrics']['standard_success_rate'] < 0.95:
            recommendations.append("Standard test cases should have >95% success rate")
        
        # Quality recommendations
        if 'challenging_avg_quality' in self.results['quality_scores']:
            if self.results['quality_scores']['challenging_avg_quality'] < 50:
                recommendations.append("Quality scores on challenging images are low - improve enhancement algorithms")
        
        # Performance recommendations
        if 'challenging_avg_time' in self.results['performance_metrics']:
            if self.results['performance_metrics']['challenging_avg_time'] > 5.0:
                recommendations.append("Processing time is high - consider optimization")
        
        # Bug recommendations
        if self.results['bugs_detected']['total_bugs'] > 0:
            recommendations.append(f"Fix {self.results['bugs_detected']['total_bugs']} detected bugs")
        
        if not recommendations:
            recommendations.append("Performance is excellent - no major issues detected")
        
        return recommendations

def main():
    """Run comprehensive evaluation."""
    print("QUANTUM VISUAL PROCESSOR EVALUATION")
    print("=" * 50)
    
    evaluator = ProcessorEvaluation()
    results = evaluator.run_comprehensive_evaluation()
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Images Tested: {report['evaluation_summary']['total_images_tested']}")
    print(f"Overall Success Rate: {report['performance_metrics']['overall_success_rate']:.2%}")
    print(f"Challenging Success Rate: {report['performance_metrics']['challenging_success_rate']:.2%}")
    print(f"Standard Success Rate: {report['performance_metrics']['standard_success_rate']:.2%}")
    
    if 'challenging_avg_quality' in report['quality_scores']:
        print(f"Challenging Quality Score: {report['quality_scores']['challenging_avg_quality']:.1f}")
    if 'standard_avg_quality' in report['quality_scores']:
        print(f"Standard Quality Score: {report['quality_scores']['standard_avg_quality']:.1f}")
    
    print(f"Total Bugs Detected: {report['bug_analysis']['total_bugs']}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed results saved to: evaluation_results.json")
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main()