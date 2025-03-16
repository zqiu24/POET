import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
import re

def get_method_color(method_name):
    """Get color for a method based on its name."""
    method_lower = method_name.lower()
    
    if 'adamw' in method_lower:
        return '#404040'  # Dark gray for AdamW
    elif 'galore' in method_lower:
        return '#808080'  # Light gray for Galore
    elif ('2' in method_lower and 'poet' in method_lower):
        return '#d62728'  # Red for POET with rank 2
    elif ('4' in method_lower and 'poet' in method_lower):
        return '#1f77b4'  # Blue for POET with rank 4
    else:
        return '#a0a0a0'  # Default light gray for other methods

def extract_memory_value(value_str):
    """Extract numeric memory value from string like '0.328418 GB'"""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    # Extract number from string
    match = re.search(r'(\d+\.\d+)', str(value_str))
    if match:
        return float(match.group(1))
    return np.nan

def create_memory_plot(csv_path, output_dir, plot_type='bar'):
    """
    Create a memory usage plot from a CSV file.
    
    Args:
        csv_path: Path to CSV file with memory usage data
        output_dir: Directory to save the plot
        plot_type: Type of plot ('bar' or 'line')
    """
    # Set up seaborn for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Use consistent figure size and styling
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10
    })
    
    # Read the CSV with semicolon separator
    df = pd.read_csv(csv_path, sep=';')
    
    # Clean up column names by removing quotes and extra spaces
    df.columns = [col.replace('"', '').strip() for col in df.columns]
    
    # Get the model column (first column) and method columns (all other columns)
    model_col = df.columns[0]
    method_columns = df.columns[1:]
    
    # Process memory values - extract numeric values from strings like "0.328418 GB"
    for col in method_columns:
        df[col] = df[col].apply(extract_memory_value).round(3)
    
    # Check if we have valid numeric data
    if df[method_columns].isnull().all().any():
        problematic_cols = df[method_columns].columns[df[method_columns].isnull().all()].tolist()
        raise ValueError(f"Columns with all NaN values: {problematic_cols}")
    
    # Create the plot with fixed dimensions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set colors for methods
    method_colors = {
        method: get_method_color(method) for method in method_columns
    }
    
    # Find the maximum value for consistent y-axis scaling
    max_value = df[method_columns].max().max()
    y_max = max_value * 1.1  # Add 10% padding
    
    # Print debug info
    print(f"Max value: {max_value}, Y-max: {y_max}")
    print(f"Data summary:\n{df[method_columns].describe()}")
    
    if plot_type == 'bar':
        # Create a grouped bar chart
        bar_width = 0.8 / len(method_columns)
        x = np.arange(len(df))
        
        for i, method in enumerate(method_columns):
            offset = (i - len(method_columns)/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, df[method], bar_width, label=method, color=method_colors[method])
            
            # Add value labels on top of bars with 3 decimal places
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max_value * 0.01),
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(df[model_col], rotation=0)
        
    else:  # Line plot
        # Extract model sizes in millions of parameters
        df['model_size_M'] = df[model_col].apply(lambda x: extract_model_size(x))
        
        # Sort by model size
        df = df.sort_values('model_size_M')
        
        # Plot each method
        for method in method_columns:
            # Plot line
            sns.lineplot(data=df, x='model_size_M', y=method, marker='o', markersize=8, 
                         linewidth=2, label=method, ax=ax, color=method_colors[method])
            
            # Add value labels next to points with 3 decimal places
            for i, row in df.iterrows():
                ax.text(row['model_size_M'] * 1.05, row[method] * 1.02, 
                        f"{row[method]:.3f}", fontsize=9)
        
        # Set x-axis to log scale
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (Millions of Parameters)')
    
    # Set y-axis to start from 0 and go to max_value + 10%
    ax.set_ylim(0, y_max)
    
    # Set labels and title
    if plot_type == 'bar':
        ax.set_xlabel('Model Size')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Memory Usage Across Different Models and Methods')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add legend
    ax.legend(title='Methods', loc='best')
    
    # Get output filename
    output_filename = f"memory_usage_{plot_type}"
    
    # Save the figure
    output_path = output_dir / f"{output_filename}.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    plt.close()

def extract_model_size(model_name):
    """Extract model size in millions of parameters from model name."""
    # Look for patterns like "60M", "350M", "1.5B", "7B"
    m_match = re.search(r'(\d+(?:\.\d+)?)M', model_name)
    b_match = re.search(r'(\d+(?:\.\d+)?)B', model_name)
    
    if m_match:
        return float(m_match.group(1))
    elif b_match:
        return float(b_match.group(1)) * 1000  # Convert B to M
    else:
        # Try to extract just a number
        num_match = re.search(r'(\d+(?:\.\d+)?)', model_name)
        if num_match:
            # Assume it's in millions if no unit
            return float(num_match.group(1))
    
    # Default if no size found
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate memory usage plot from CSV data")
    parser.add_argument("csv_path", type=str, help="Path to CSV file with memory usage data")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plot")
    parser.add_argument("--plot_type", type=str, choices=['bar', 'line'], default='bar',
                        help="Type of plot to generate (bar or line)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Create the plot
        create_memory_plot(args.csv_path, output_dir, args.plot_type)
        print("Memory usage plot generated successfully!")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()