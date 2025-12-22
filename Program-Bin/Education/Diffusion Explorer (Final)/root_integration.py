"""
ROOT Integration for Advanced Physics Analysis
CERN's ROOT framework integration for sophisticated diffusion calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

# Try to import ROOT - provide fallback if not available
try:
    import ROOT
    ROOT_AVAILABLE = True
    print("ROOT framework successfully loaded")
except ImportError:
    ROOT_AVAILABLE = False
    print("ROOT not available - using mock implementations")
    # Create mock ROOT classes for development
    class MockROOT:
        class TCanvas:
            def __init__(self, name, title, w, h): pass
            def cd(self): pass
            def Update(self): pass
            def SaveAs(self, filename): pass
        
        class TGraph:
            def __init__(self, n): self.n = n; self.x = []; self.y = []
            def SetPoint(self, i, x, y): 
                if i < self.n: self.x.append(x); self.y.append(y)
            def SetTitle(self, title): pass
            def SetMarkerStyle(self, style): pass
            def SetMarkerColor(self, color): pass
            def Draw(self, option): pass
        
        class TH1F:
            def __init__(self, name, title, nbins, xmin, xmax): 
                self.name = name; self.data = np.zeros(nbins)
                self.xmin = xmin; self.xmax = xmax
            def Fill(self, x, weight=1): 
                bin_idx = int((x - self.xmin) / (self.xmax - self.xmin) * len(self.data))
                if 0 <= bin_idx < len(self.data):
                    self.data[bin_idx] += weight
            def SetTitle(self, title): pass
            def SetFillColor(self, color): pass
            def Draw(self, option): pass
            def GetMean(self): return np.mean(self.data)
            def GetRMS(self): return np.std(self.data)
        
        class TF1:
            def __init__(self, name, formula, xmin, xmax): 
                self.name = name; self.formula = formula
                self.xmin = xmin; self.xmax = xmax
            def SetParameter(self, i, value): pass
            def SetParNames(self, *names): pass
            def Eval(self, x): return eval(self.formula.replace('x', str(x)))
            def Draw(self): pass
        
        class TFitResult:
            def __init__(self): pass
            def Parameter(self, i): return 0.0
            def ParError(self, i): return 0.0
            def Chi2(self): return 0.0
            def Ndf(self): return 1
    
    ROOT = MockROOT()
    warnings.warn("Using mock ROOT implementation - some features may be limited")

class ROOTDiffusionAnalyzer:
    """Advanced diffusion analysis using ROOT framework"""
    
    def __init__(self):
        self.canvas = None
        self.graphs = []
        self.histograms = []
        self.functions = []
        self.fit_results = []
        
    def create_diffusion_histogram(self, data: np.ndarray, name: str = "diffusion_hist", 
                                 title: str = "Diffusion Distribution") -> Any:
        """
        Create ROOT histogram for diffusion data analysis
        """
        if not ROOT_AVAILABLE:
            return self._create_mock_histogram(data, name, title)
        
        # Calculate histogram parameters
        nbins = min(50, len(data) // 10)
        xmin, xmax = np.min(data), np.max(data)
        
        # Create ROOT histogram
        hist = ROOT.TH1F(name, title, nbins, xmin, xmax)
        
        # Fill histogram with data
        for value in data:
            hist.Fill(value)
        
        # Customize histogram
        hist.SetTitle(title)
        hist.SetFillColor(ROOT.kAzure - 3)
        
        self.histograms.append(hist)
        return hist
    
    def create_arrhenius_plot(self, temperatures: np.ndarray, diffusion_coeffs: np.ndarray,
                            material_name: str = "Material") -> Tuple[Any, Any]:
        """
        Create Arrhenius plot and fit using ROOT
        """
        if not ROOT_AVAILABLE:
            return self._create_mock_arrhenius(temperatures, diffusion_coeffs, material_name)
        
        # Calculate 1/T and ln(D) for Arrhenius plot
        inv_T = 1.0 / temperatures
        ln_D = np.log(diffusion_coeffs)
        
        # Create TGraph for plotting
        n_points = len(temperatures)
        graph = ROOT.TGraph(n_points)
        
        for i in range(n_points):
            graph.SetPoint(i, inv_T[i], ln_D[i])
        
        graph.SetTitle(f"Arrhenius Plot - {material_name}")
        graph.SetMarkerStyle(20)
        graph.SetMarkerColor(ROOT.kRed)
        
        # Create Arrhenius function: ln(D) = ln(D0) - Qa/(R*T)
        # ln(D) = ln(D0) - Qa/R * (1/T)
        arrhenius_func = ROOT.TF1("arrhenius", "pol1", min(inv_T), max(inv_T))
        arrhenius_func.SetParNames("ln(D0)", "-Qa/R")
        
        # Fit the function
        fit_result = graph.Fit(arrhenius_func, "S")
        
        # Extract activation energy and pre-exponential factor
        R = 8.314  # J/(mol·K)
        slope = arrhenius_func.GetParameter(1)
        intercept = arrhenius_func.GetParameter(0)
        
        Qa = -slope * R / 1000  # Convert to kJ/mol
        D0 = np.exp(intercept)
        
        self.graphs.append(graph)
        self.functions.append(arrhenius_func)
        
        return graph, {
            'activation_energy': Qa,
            'pre_exponential': D0,
            'fit_quality': {
                'chi2': fit_result.Chi2(),
                'ndf': fit_result.Ndf(),
                'r_squared': self._calculate_r_squared(inv_T, ln_D, arrhenius_func)
            }
        }
    
    def perform_error_analysis(self, experimental_data: Dict, theoretical_model: Any) -> Dict:
        """
        Perform statistical analysis using ROOT tools
        """
        if not ROOT_AVAILABLE:
            return self._mock_error_analysis(experimental_data, theoretical_model)
        
        # Extract data
        exp_temps = np.array(experimental_data['temperatures'])
        exp_diffusion = np.array(experimental_data['diffusion_coefficients'])
        
        # Calculate theoretical predictions
        theo_diffusion = []
        for temp in exp_temps:
            theo_diffusion.append(theoretical_model.Eval(temp))
        
        theo_diffusion = np.array(theo_diffusion)
        
        # Calculate residuals
        residuals = exp_diffusion - theo_diffusion
        relative_errors = residuals / exp_diffusion * 100
        
        # Create histogram of residuals
        res_hist = ROOT.TH1F("residuals", "Residuals Distribution", 20, 
                           np.min(relative_errors), np.max(relative_errors))
        
        for error in relative_errors:
            res_hist.Fill(error)
        
        res_hist.SetFillColor(ROOT.kOrange)
        res_hist.SetTitle("Relative Error Distribution (%)")
        
        # Statistical analysis
        mean_error = np.mean(relative_errors)
        std_error = np.std(relative_errors)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Chi-square calculation
        chi_square = np.sum((residuals / theo_diffusion)**2)
        reduced_chi2 = chi_square / len(exp_temps)
        
        analysis_results = {
            'mean_relative_error': mean_error,
            'std_relative_error': std_error,
            'rmse': rmse,
            'chi_square': chi_square,
            'reduced_chi2': reduced_chi2,
            'histogram': res_hist,
            'quality_assessment': self._assess_quality(reduced_chi2, std_error)
        }
        
        self.histograms.append(res_hist)
        return analysis_results
    
    def create_multi_material_comparison(self, materials_data: Dict[str, Dict]) -> Any:
        """
        Create multi-material comparison plot using ROOT
        """
        if not ROOT_AVAILABLE:
            return self._create_mock_comparison(materials_data)
        
        # Create canvas for comparison
        canvas = ROOT.TCanvas("comparison", "Diffusion Comparison", 800, 600)
        canvas.cd()
        
        # Create legend
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        
        colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange]
        
        for i, (material_name, data) in enumerate(materials_data.items()):
            if i >= len(colors):
                break
                
            temps = np.array(data['temperatures'])
            coeffs = np.array(data['diffusion_coefficients'])
            
            # Create graph for this material
            graph = ROOT.TGraph(len(temps))
            for j in range(len(temps)):
                graph.SetPoint(j, temps[j], coeffs[j])
            
            graph.SetMarkerStyle(20 + i)
            graph.SetMarkerColor(colors[i])
            graph.SetLineColor(colors[i])
            graph.SetTitle(f"{material_name}")
            
            # Draw graph
            if i == 0:
                graph.Draw("ALP")
            else:
                graph.Draw("LP SAME")
            
            # Add to legend
            legend.AddEntry(graph, material_name, "LP")
            
            self.graphs.append(graph)
        
        legend.Draw()
        canvas.Update()
        
        self.canvas = canvas
        return canvas
    
    def perform_monte_carlo_uncertainty(self, base_model: Any, parameters: Dict,
                                      n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo uncertainty analysis using ROOT
        """
        if not ROOT_AVAILABLE:
            return self._mock_monte_carlo(base_model, parameters, n_simulations)
        
        # Initialize results storage
        all_results = []
        
        # Parameter uncertainties (example values)
        param_uncertainties = {
            'D0': parameters.get('D0_uncertainty', 0.1 * parameters.get('D0', 1e-5)),
            'Qa': parameters.get('Qa_uncertainty', 0.05 * parameters.get('Qa', 50))
        }
        
        # Run Monte Carlo simulations
        for i in range(n_simulations):
            # Generate perturbed parameters
            D0_perturbed = np.random.normal(parameters['D0'], param_uncertainties['D0'])
            Qa_perturbed = np.random.normal(parameters['Qa'], param_uncertainties['Qa'])
            
            # Calculate with perturbed parameters
            temp_range = np.linspace(300, 1200, 50)
            D_perturbed = D0_perturbed * np.exp(-Qa_perturbed * 1000 / (8.314 * temp_range))
            
            all_results.append(D_perturbed)
        
        all_results = np.array(all_results)
        
        # Calculate statistics
        mean_results = np.mean(all_results, axis=0)
        std_results = np.std(all_results, axis=0)
        
        # Create confidence band histogram
        confidence_hist = ROOT.TH1F("confidence", "95% Confidence Bands", len(temp_range),
                                  min(temp_range), max(temp_range))
        
        for i, temp in enumerate(temp_range):
            confidence_hist.Fill(temp, std_results[i])
        
        uncertainty_results = {
            'mean_values': mean_results,
            'std_values': std_results,
            'confidence_95': 1.96 * std_results,
            'temperature_range': temp_range,
            'histogram': confidence_hist,
            'total_uncertainty': np.mean(std_results / mean_results)
        }
        
        self.histograms.append(confidence_hist)
        return uncertainty_results
    
    def save_root_file(self, filename: str = "diffusion_analysis.root"):
        """
        Save all ROOT objects to a ROOT file
        """
        if not ROOT_AVAILABLE:
            print("ROOT not available - skipping file save")
            return
        
        # Create ROOT file
        root_file = ROOT.TFile(filename, "RECREATE")
        
        # Save all objects
        for i, hist in enumerate(self.histograms):
            hist.Write()
        
        for i, graph in enumerate(self.graphs):
            graph.Write()
        
        for i, func in enumerate(self.functions):
            func.Write()
        
        # Close file
        root_file.Close()
        print(f"ROOT objects saved to {filename}")
    
    def export_analysis_results(self, filename: str = "root_analysis_results.json"):
        """
        Export analysis results to JSON format
        """
        results = {
            'n_histograms': len(self.histograms),
            'n_graphs': len(self.graphs),
            'n_functions': len(self.functions),
            'root_available': ROOT_AVAILABLE,
            'analysis_summary': []
        }
        
        # Add analysis results if available
        for result in self.fit_results:
            results['analysis_summary'].append(result)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis results exported to {filename}")
    
    def _calculate_r_squared(self, x_data, y_data, function):
        """Calculate R-squared for fit quality"""
        y_pred = [function.Eval(x) for x in x_data]
        ss_res = np.sum((y_data - y_pred)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _assess_quality(self, reduced_chi2, std_error):
        """Assess quality of fit"""
        if reduced_chi2 < 1.0 and abs(std_error) < 5:
            return "Excellent"
        elif reduced_chi2 < 2.0 and abs(std_error) < 10:
            return "Good"
        elif reduced_chi2 < 3.0 and abs(std_error) < 20:
            return "Fair"
        else:
            return "Poor"
    
    # Mock implementations for when ROOT is not available
    def _create_mock_histogram(self, data, name, title):
        """Create mock histogram when ROOT is not available"""
        return {
            'type': 'histogram',
            'name': name,
            'title': title,
            'data': data,
            'mean': np.mean(data),
            'std': np.std(data)
        }
    
    def _create_mock_arrhenius(self, temperatures, diffusion_coeffs, material_name):
        """Create mock Arrhenius plot when ROOT is not available"""
        # Simple linear fit to log(data) vs 1/T
        inv_T = 1.0 / temperatures
        ln_D = np.log(diffusion_coeffs)
        
        # Linear regression
        coeffs = np.polyfit(inv_T, ln_D, 1)
        slope, intercept = coeffs
        
        R = 8.314
        Qa = -slope * R / 1000
        D0 = np.exp(intercept)
        
        return {
            'type': 'arrhenius_plot',
            'material': material_name,
            'data': {'x': inv_T, 'y': ln_D},
            'fit_params': {'slope': slope, 'intercept': intercept},
            'activation_energy': Qa,
            'pre_exponential': D0
        }, {
            'activation_energy': Qa,
            'pre_exponential': D0,
            'fit_quality': {'r_squared': 0.95, 'mock': True}
        }
    
    def _mock_error_analysis(self, experimental_data, theoretical_model):
        """Mock error analysis when ROOT is not available"""
        return {
            'mean_relative_error': 5.2,
            'std_relative_error': 3.1,
            'rmse': 1e-10,
            'chi_square': 12.3,
            'reduced_chi2': 1.5,
            'quality_assessment': 'Good',
            'mock': True
        }
    
    def _create_mock_comparison(self, materials_data):
        """Mock comparison when ROOT is not available"""
        return {
            'type': 'comparison_plot',
            'materials': list(materials_data.keys()),
            'n_materials': len(materials_data),
            'mock': True
        }
    
    def _mock_monte_carlo(self, base_model, parameters, n_simulations):
        """Mock Monte Carlo when ROOT is not available"""
        return {
            'mean_values': np.random.random(50) * 1e-5,
            'std_values': np.random.random(50) * 1e-6,
            'confidence_95': 1.96 * np.random.random(50) * 1e-6,
            'temperature_range': np.linspace(300, 1200, 50),
            'total_uncertainty': 0.15,
            'n_simulations': n_simulations,
            'mock': True
        }

# Utility functions for ROOT integration
def setup_root_environment():
    """Setup ROOT environment with proper configurations"""
    if not ROOT_AVAILABLE:
        print("ROOT environment setup skipped - ROOT not available")
        return False
    
    # Set ROOT style
    ROOT.gStyle.SetOptStat(1111)
    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetPadGridX(True)
    ROOT.gStyle.SetPadGridY(True)
    
    # Create default canvas
    canvas = ROOT.TCanvas("canvas", "Diffusion Analysis", 800, 600)
    
    return True

def create_latex_formula(formula_string: str, title: str = "Formula") -> str:
    """
    Create LaTeX-formatted string for mathematical expressions
    """
    latex_map = {
        'exp': 'e^{',
        'log': '\\ln(',
        'sqrt': '\\sqrt{',
        'sigma': '\\sigma',
        'tau': '\\tau',
        'inf': '\\infty',
        'sum': '\\sum',
        'int': '\\int',
        'partial': '\\partial',
        'alpha': '\\alpha',
        'beta': '\\beta',
        'gamma': '\\gamma',
        'delta': '\\delta'
    }
    
    # Convert to proper LaTeX format
    latex_formula = formula_string
    for symbol, latex_symbol in latex_map.items():
        latex_formula = latex_formula.replace(symbol, latex_symbol)
    
    return f"\\textbf{{{title}}}: ${latex_formula}$"