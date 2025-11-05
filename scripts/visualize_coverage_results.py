#!/usr/bin/env python3
"""
Utility script to visualize and analyze TARP coverage test results.
This script can re-plot coverage diagnostics from saved .npz files.

Usage:
    python visualize_coverage_results.py <coverage_data.npz>
    python visualize_coverage_results.py <coverage_data.npz> --output plot.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_coverage_from_data(data_file, output_file=None, show=True):
    """Plot coverage diagnostics from saved TARP data."""
    
    # Load data
    print(f"Loading coverage data from: {data_file}")
    data = np.load(data_file)
    
    ecp = data['ecp']
    alpha = data['alpha']
    bootstrap = data.get('bootstrap', False)
    
    print(f"  Bootstrap: {bootstrap}")
    print(f"  ECP shape: {ecp.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    if bootstrap:
        # Load bootstrap statistics if available
        if 'ecp_mean' in data and 'ecp_std' in data:
            ecp_mean = data['ecp_mean']
            ecp_std = data['ecp_std']
        else:
            # Compute from bootstrap samples
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
        
        # Plot with uncertainty band
        ax.plot(alpha, ecp_mean, 'b-', linewidth=2.5, label='TARP Coverage')
        ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std, 
                        alpha=0.3, color='blue', label='Bootstrap uncertainty (1σ)')
        
        # Compute deviation metrics
        deviation = np.abs(ecp_mean - alpha)
        max_deviation = np.max(deviation)
        mean_deviation = np.mean(deviation)
        
    else:
        # Plot without uncertainty
        ax.plot(alpha, ecp, 'b-', linewidth=2.5, label='TARP Coverage')
        
        # Compute deviation metrics
        deviation = np.abs(ecp - alpha)
        max_deviation = np.max(deviation)
        mean_deviation = np.mean(deviation)
    
    # Plot ideal calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal calibration', alpha=0.7)
    
    # Add shaded region for small deviations
    ax.fill_between([0, 1], [0, 1], [0.1, 1.1], alpha=0.05, color='green', 
                    label='±0.1 tolerance zone')
    ax.fill_between([0, 1], [-0.1, 0.9], [0, 1], alpha=0.05, color='green')
    
    # Formatting
    ax.set_xlabel('Credibility Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expected Coverage Probability', fontsize=14, fontweight='bold')
    ax.set_title('TARP Coverage Diagnostic', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Add text box with statistics
    textstr = f'Max deviation: {max_deviation:.3f}\nMean deviation: {mean_deviation:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add calibration assessment
    if mean_deviation < 0.05:
        assessment = "✓ Excellent calibration"
        color = 'green'
    elif mean_deviation < 0.1:
        assessment = "✓ Good calibration"
        color = 'darkgreen'
    elif mean_deviation < 0.15:
        assessment = "⚠ Moderate calibration"
        color = 'orange'
    else:
        assessment = "⚠ Poor calibration"
        color = 'red'
    
    ax.text(0.95, 0.05, assessment, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontweight='bold', color=color)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("TARP Coverage Statistics")
    print("="*60)
    print(f"Mean deviation from ideal: {mean_deviation:.4f}")
    print(f"Max deviation from ideal: {max_deviation:.4f}")
    print(f"Assessment: {assessment}")
    
    if bootstrap:
        print(f"\nBootstrap samples: {ecp.shape[0]}")
        print(f"Mean uncertainty (1σ): {np.mean(ecp_std):.4f}")
        print(f"Max uncertainty (1σ): {np.max(ecp_std):.4f}")
    
    print("="*60 + "\n")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TARP coverage test results from saved data"
    )
    parser.add_argument("data_file", type=str, 
                       help="Path to coverage data .npz file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for plot (PDF, PNG, etc.)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plot window (useful for batch processing)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data_file):
        print(f"Error: File not found: {args.data_file}")
        return 1
    
    # Plot coverage
    plot_coverage_from_data(
        args.data_file, 
        output_file=args.output,
        show=not args.no_show
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
