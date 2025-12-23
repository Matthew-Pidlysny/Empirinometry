#!/usr/bin/env python3
"""
Analysis of Program-Bin/AI folder programs
Testing and running Python and C++ programs as requested
"""

import os
import subprocess
import sys
import time

def compile_and_run_cpp(filename):
    """Compile and run C++ program"""
    print(f"\n{'='*60}")
    print(f"COMPILING AND RUNNING: {filename}")
    print(f"{'='*60}")
    
    filepath = f"Empirinometry/Program-Bin/AI/{filename}"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found")
        return False
    
    # Compile
    executable = f"/tmp/{filename.replace('.cpp', '')}"
    try:
        compile_cmd = f"g++ -o {executable} {filepath} -std=c++11"
        print(f"Compiling: {compile_cmd}")
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"COMPILATION ERROR for {filename}:")
            print(result.stderr)
            return False
        else:
            print("✓ Compilation successful")
    
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout for {filename}")
        return False
    
    # Run with timeout
    try:
        print(f"\nRunning {executable}...")
        start_time = time.time()
        result = subprocess.run(executable, shell=True, capture_output=True, text=True, timeout=10)
        runtime = time.time() - start_time
        
        print(f"Runtime: {runtime:.3f} seconds")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"Program exited with code {result.returncode}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Program {filename} timed out after 10 seconds")
        return False

def run_python_script(filename):
    """Run Python script"""
    print(f"\n{'='*60}")
    print(f"RUNNING PYTHON: {filename}")
    print(f"{'='*60}")
    
    filepath = f"Empirinometry/Program-Bin/AI/{filename}"
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found")
        return False
    
    try:
        print(f"Running: python {filepath}")
        start_time = time.time()
        result = subprocess.run([sys.executable, filepath], capture_output=True, text=True, timeout=10)
        runtime = time.time() - start_time
        
        print(f"Runtime: {runtime:.3f} seconds")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"Script exited with code {result.returncode}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Python script {filename} timed out after 10 seconds")
        return False

def analyze_file_content(filename):
    """Quick analysis of file content"""
    filepath = f"Empirinometry/Program-Bin/AI/{filename}"
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"\nFile Analysis: {filename}")
        print(f"Size: {len(content)} characters")
        print(f"Lines: {len(content.splitlines())}")
        
        # Quick content preview
        lines = content.splitlines()
        if lines:
            print(f"First line: {lines[0][:100]}...")
            if len(lines) > 1:
                print(f"Last line: {lines[-1][:100]}...")
        
        # Look for key indicators
        if "main(" in content:
            print("✓ Contains main() function")
        if "include" in content and ".cpp" in filename:
            print("✓ C++ includes detected")
        if "import" in content and ".py" in filename:
            print("✓ Python imports detected")
        
        return True
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return False

def run_ai_folder_analysis():
    """Comprehensive analysis of AI folder"""
    print("AI FOLDER COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # List of files to analyze
    cpp_files = [
        'advanced-torsion.cpp',
        'advanced_torsion.cpp', 
        'carl-bool-interactive.cpp',
        'connie-pedpenki.cpp'
    ]
    
    python_files = [
        'Numbers Truth Discovery.py',
        'circus.py',
        'geopatra.py',
        'johnnyb.py',
        'letheridge.py',
        'perplexus.py',
        'soldat.py',
        'tinker-bell.py'
    ]
    
    text_files = [
        'ethical-briefing-first-contact.txt',
        'ethical-briefing-standard.txt',
        'good-boy-rules.txt',
        'johnnyb-readme.txt',
        'letheridge.txt',
        'letter-about-ubarr.txt',
        'message-from-connie.txt',
        'the-bellworthy-letter.txt',
        'war tutorial.txt'
    ]
    
    results = {
        'cpp_success': 0,
        'cpp_failed': 0,
        'python_success': 0,
        'python_failed': 0,
        'text_analyzed': 0
    }
    
    # Analyze C++ files
    print("\n" + "="*80)
    print("C++ PROGRAMS ANALYSIS")
    print("="*80)
    
    for cpp_file in cpp_files:
        print(f"\nProcessing C++ file: {cpp_file}")
        analyze_file_content(cpp_file)
        
        if compile_and_run_cpp(cpp_file):
            results['cpp_success'] += 1
        else:
            results['cpp_failed'] += 1
    
    # Analyze Python files  
    print("\n" + "="*80)
    print("PYTHON PROGRAMS ANALYSIS")
    print("="*80)
    
    for py_file in python_files:
        print(f"\nProcessing Python file: {py_file}")
        analyze_file_content(py_file)
        
        if run_python_script(py_file):
            results['python_success'] += 1
        else:
            results['python_failed'] += 1
    
    # Quick text file summary
    print("\n" + "="*80)
    print("TEXT FILES SUMMARY")
    print("="*80)
    
    for txt_file in text_files:
        analyze_file_content(txt_file)
        results['text_analyzed'] += 1
    
    # Summary
    print("\n" + "="*80)
    print("AI FOLDER ANALYSIS SUMMARY")
    print("="*80)
    print(f"C++ programs successfully run: {results['cpp_success']}")
    print(f"C++ programs failed: {results['cpp_failed']}")
    print(f"Python programs successfully run: {results['python_success']}")
    print(f"Python programs failed: {results['python_failed']}")
    print(f"Text files analyzed: {results['text_analyzed']}")
    
    total_programs = results['cpp_success'] + results['cpp_failed'] + results['python_success'] + results['python_failed']
    successful_programs = results['cpp_success'] + results['python_success']
    
    if total_programs > 0:
        print(f"Overall success rate: {successful_programs}/{total_programs} ({100*successful_programs/total_programs:.1f}%)")
    
    return results

if __name__ == "__main__":
    run_ai_folder_analysis()