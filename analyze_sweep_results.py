#!/usr/bin/env python3
"""
Analysis tools for parameter sweep results.

This script provides functions to analyze and visualize results from parameter sweeps.

Usage:
    python analyze_sweep_results.py --summary-file results_summary.csv
    python analyze_sweep_results.py --results-dir parameter_sweep_results
    
Author: Generated for Siyona Project
Date: 2025-09-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class SweepResultsAnalyzer:
    """
    Analyzer for parameter sweep results.
    """
    
    def __init__(self, summary_df: pd.DataFrame):
        """
        Initialize analyzer with summary results.
        
        Args:
            summary_df: DataFrame with parameter sweep summary results
        """
        self.df = summary_df.copy()
        self.successful_df = self.df[self.df['success'] == True].copy()
        
        if self.successful_df.empty:
            raise ValueError("No successful simulations found in the data")
        
        # Extract parameter columns
        self.param_columns = []
        if 'parameters' in self.df.columns:
            # Extract parameter names from the first successful run
            first_params = self.successful_df.iloc[0]['parameters']
            if isinstance(first_params, dict):
                self.param_columns = list(first_params.keys())
                # Expand parameters into separate columns
                self._expand_parameters()
        
        print(f"Loaded {len(self.df)} total runs, {len(self.successful_df)} successful")
        print(f"Parameter columns: {self.param_columns}")
    
    def _expand_parameters(self):
        """Expand parameter dictionaries into separate columns."""
        for idx, row in self.successful_df.iterrows():
            params = row['parameters']
            if isinstance(params, dict):
                for param_name, param_value in params.items():
                    self.successful_df.loc[idx, f'param_{param_name}'] = param_value
    
    def print_summary_statistics(self):
        """Print summary statistics of the sweep results."""
        print("\n" + "="*60)
        print("PARAMETER SWEEP SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nTotal simulations: {len(self.df)}")
        print(f"Successful simulations: {len(self.successful_df)}")
        print(f"Failed simulations: {len(self.df) - len(self.successful_df)}")
        
        if not self.successful_df.empty:
            print(f"\nEscape Fraction Statistics:")
            print(f"  Mean: {self.successful_df['escape_fraction'].mean():.4f}")
            print(f"  Std:  {self.successful_df['escape_fraction'].std():.4f}")
            print(f"  Min:  {self.successful_df['escape_fraction'].min():.4f}")
            print(f"  Max:  {self.successful_df['escape_fraction'].max():.4f}")
            
            print(f"\nRecapture Fraction Statistics:")
            print(f"  Mean: {self.successful_df['recapture_fraction'].mean():.4f}")
            print(f"  Std:  {self.successful_df['recapture_fraction'].std():.4f}")
            print(f"  Min:  {self.successful_df['recapture_fraction'].min():.4f}")
            print(f"  Max:  {self.successful_df['recapture_fraction'].max():.4f}")
            
            if 'avg_beta_crossings' in self.successful_df.columns:
                print(f"\nAverage Beta Crossings:")
                print(f"  Mean: {self.successful_df['avg_beta_crossings'].mean():.2f}")
                print(f"  Std:  {self.successful_df['avg_beta_crossings'].std():.2f}")
        
        # Parameter ranges
        print(f"\nParameter Ranges:")
        for param in self.param_columns:
            param_col = f'param_{param}'
            if param_col in self.successful_df.columns:
                values = self.successful_df[param_col].dropna()
                if not values.empty:
                    if values.dtype in ['int64', 'float64']:
                        print(f"  {param}: {values.min():.3f} to {values.max():.3f}")
                    else:
                        unique_vals = values.unique()
                        print(f"  {param}: {list(unique_vals)}")
    
    def plot_escape_fraction_vs_parameters(self, save_plots: bool = True, output_dir: str = "plots"):
        """
        Plot escape fraction vs each parameter.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        # Find numeric parameters
        numeric_params = []
        for param in self.param_columns:
            param_col = f'param_{param}'
            if param_col in self.successful_df.columns:
                if self.successful_df[param_col].dtype in ['int64', 'float64']:
                    numeric_params.append(param)
        
        if not numeric_params:
            print("No numeric parameters found for plotting")
            return
        
        # Create subplots
        n_params = len(numeric_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(numeric_params):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            param_col = f'param_{param}'
            x_data = self.successful_df[param_col]
            y_data = self.successful_df['escape_fraction']
            
            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=50)
            
            # Add trend line if enough unique values
            if len(x_data.unique()) > 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Escape Fraction')
            ax.set_title(f'Escape Fraction vs {param}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/escape_fraction_vs_parameters.png", dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_dir}/escape_fraction_vs_parameters.png")
        
        plt.show()
    
    def plot_correlation_matrix(self, save_plots: bool = True, output_dir: str = "plots"):
        """
        Plot correlation matrix of parameters and results.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        # Select numeric columns for correlation
        numeric_cols = []
        
        # Add parameter columns
        for param in self.param_columns:
            param_col = f'param_{param}'
            if param_col in self.successful_df.columns:
                if self.successful_df[param_col].dtype in ['int64', 'float64']:
                    numeric_cols.append(param_col)
        
        # Add result columns
        result_cols = ['escape_fraction', 'recapture_fraction', 'resimulate_fraction']
        for col in result_cols:
            if col in self.successful_df.columns:
                numeric_cols.append(col)
        
        if 'avg_beta_crossings' in self.successful_df.columns:
            numeric_cols.append('avg_beta_crossings')
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_data = self.successful_df[numeric_cols]
        corr_matrix = corr_data.corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Parameter and Result Correlation Matrix')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_dir}/correlation_matrix.png")
        
        plt.show()
    
    def plot_result_distributions(self, save_plots: bool = True, output_dir: str = "plots"):
        """
        Plot distributions of simulation results.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Escape fraction distribution
        axes[0, 0].hist(self.successful_df['escape_fraction'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Escape Fraction')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Escape Fractions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recapture fraction distribution
        axes[0, 1].hist(self.successful_df['recapture_fraction'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Recapture Fraction')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Recapture Fractions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Resimulate fraction distribution
        axes[1, 0].hist(self.successful_df['resimulate_fraction'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Resimulate Fraction')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Resimulate Fractions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Beta crossings distribution (if available)
        if 'avg_beta_crossings' in self.successful_df.columns:
            axes[1, 1].hist(self.successful_df['avg_beta_crossings'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Average Beta Crossings')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Average Beta Crossings')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Beta crossings\ndata not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Beta Crossings (N/A)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/result_distributions.png", dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_dir}/result_distributions.png")
        
        plt.show()
    
    def find_optimal_parameters(self, criterion: str = "min_escape") -> pd.DataFrame:
        """
        Find optimal parameter combinations based on specified criterion.
        
        Args:
            criterion: Optimization criterion ("min_escape", "max_recapture", "min_resimulate")
            
        Returns:
            DataFrame with top parameter combinations
        """
        print(f"\nFinding optimal parameters based on criterion: {criterion}")
        
        if criterion == "min_escape":
            sorted_df = self.successful_df.sort_values('escape_fraction')
            print("Top 5 parameter combinations with lowest escape fraction:")
        elif criterion == "max_recapture":
            sorted_df = self.successful_df.sort_values('recapture_fraction', ascending=False)
            print("Top 5 parameter combinations with highest recapture fraction:")
        elif criterion == "min_resimulate":
            sorted_df = self.successful_df.sort_values('resimulate_fraction')
            print("Top 5 parameter combinations with lowest resimulate fraction:")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Select columns to display
        display_cols = ['run_id']
        
        # Add parameter columns
        for param in self.param_columns:
            param_col = f'param_{param}'
            if param_col in sorted_df.columns:
                display_cols.append(param_col)
        
        # Add result columns
        result_cols = ['escape_fraction', 'recapture_fraction', 'resimulate_fraction']
        display_cols.extend(result_cols)
        
        top_results = sorted_df[display_cols].head()
        print(top_results.to_string(index=False))
        
        return top_results


def load_summary_results(file_path: str) -> pd.DataFrame:
    """Load summary results from CSV or Excel file."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Parse parameters column if it's stored as string
    if 'parameters' in df.columns:
        if df['parameters'].dtype == 'object':
            # Try to parse as JSON
            try:
                df['parameters'] = df['parameters'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except:
                print("Warning: Could not parse parameters column as JSON")
    
    return df


def find_latest_summary_file(results_dir: str) -> Optional[str]:
    """Find the most recent summary file in the results directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return None
    
    # Look for summary files
    summary_files = list(results_path.glob("*summary*.csv")) + list(results_path.glob("*summary*.xlsx"))
    
    if not summary_files:
        return None
    
    # Return the most recent file
    latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results")
    parser.add_argument("--summary-file", type=str, help="Path to summary results file (CSV or Excel)")
    parser.add_argument("--results-dir", type=str, help="Directory containing parameter sweep results")
    parser.add_argument("--output-dir", type=str, default="analysis_plots", help="Directory to save analysis plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Determine summary file to use
    summary_file = None
    if args.summary_file:
        summary_file = args.summary_file
    elif args.results_dir:
        summary_file = find_latest_summary_file(args.results_dir)
        if summary_file:
            print(f"Found summary file: {summary_file}")
        else:
            print(f"No summary files found in {args.results_dir}")
            return
    else:
        # Look in current directory
        summary_file = find_latest_summary_file(".")
        if summary_file:
            print(f"Found summary file in current directory: {summary_file}")
        else:
            print("No summary file specified and none found in current directory")
            parser.print_help()
            return
    
    # Load and analyze results
    try:
        print(f"Loading results from: {summary_file}")
        df = load_summary_results(summary_file)
        
        analyzer = SweepResultsAnalyzer(df)
        
        # Print summary statistics
        analyzer.print_summary_statistics()
        
        # Generate plots unless disabled
        if not args.no_plots:
            print(f"\nGenerating analysis plots in: {args.output_dir}")
            analyzer.plot_result_distributions(save_plots=True, output_dir=args.output_dir)
            analyzer.plot_escape_fraction_vs_parameters(save_plots=True, output_dir=args.output_dir)
            analyzer.plot_correlation_matrix(save_plots=True, output_dir=args.output_dir)
        
        # Find optimal parameters
        print("\n" + "="*60)
        analyzer.find_optimal_parameters("min_escape")
        analyzer.find_optimal_parameters("max_recapture")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
