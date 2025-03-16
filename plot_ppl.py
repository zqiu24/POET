import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import matplotlib as mpl
import numpy as np
import re

def extract_model_name(filename):
    """Extract model name from filename for the plot title, always including LLaMA."""
    # Remove file extension and _perplexity suffix
    base_name = Path(filename).stem
    if '_perplexity' in base_name:
        base_name = base_name.split('_perplexity')[0]
    
    # Always default to LLaMA unless explicitly another model
    default_model = "LLaMA"
    
    # Check if it's explicitly a different model
    if 'gpt2' in base_name.lower() or 'gpt-2' in base_name.lower():
        default_model = "GPT-2"
    elif 'opt' in base_name.lower() and not ('llama' in base_name.lower()):
        default_model = "OPT"
    # Add other model checks as needed
    
    # Try to extract a size (any number followed by optional m or b)
    size_match = re.search(r'(\d+(?:[mb])?)', base_name, re.IGNORECASE)
    if size_match:
        size = size_match.group(1)
        # Format the size properly
        if size.lower().endswith('m'):
            size = f"{size[:-1]}M"
        elif size.lower().endswith('b'):
            size = f"{size[:-1]}B"
        else:
            # If no unit, assume millions
            size = f"{size}M"
        
        return f"{default_model} {size}"
    
    # If no size found, just return the model name
    return default_model

def get_method_style(method_name):
    """
    Determine color, line style, marker, and z-order based on method name.
    Returns (color, linestyle, marker, zorder)
    """
    method_lower = method_name.lower()
    
    # Custom coloring and ordering based on method name
    if ('4' in method_lower and 'poet' in method_lower):
        return '#1f77b4', '-', 'o', 40  # Blue solid line, highest priority
    elif ('2' in method_lower and 'poet' in method_lower):
        return '#d62728', '-', 'o', 30  # Red solid line, second priority
    elif 'galore' in method_lower:
        return '#808080', ':', 's', 20  # Gray with dotted line, third priority
    else:
        return '#404040', '--', 'o', 10  # Dark gray with dashed line, lowest priority
        
def format_step_ticks(ax, step_col):
    """Format step values on x-axis to use K for thousands."""
    # Get current tick locations
    locs = ax.get_xticks()
    
    # Format tick labels
    labels = []
    for loc in locs:
        if loc >= 1000:
            labels.append(f"{loc/1000:.0f}K")
        else:
            labels.append(f"{loc:.0f}")
    
    # Set the formatted tick labels
    ax.set_xticks(locs)
    ax.set_xticklabels(labels)

def create_simple_plot(csv_path, output_dir, fixed_ylim=None, figsize=(8, 6)):
    """
    Create a single clean plot from a CSV file with perplexity data.
    
    Args:
        csv_path: Path to CSV file with perplexity data
        output_dir: Directory to save the plot
        fixed_ylim: Optional tuple (ymin, ymax) to set fixed y-axis limits
        figsize: Tuple (width, height) for the figure size in inches
    """
    # Set up seaborn for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Use consistent figure size and styling
    plt.rcParams.update({
        "figure.figsize": figsize,  # Now defaults to 8×6
        "figure.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10
    })
    
    # Read the CSV with flexible separator detection
    df = pd.read_csv(csv_path, sep=';')
    
    # Clean up column names by removing quotes and extra spaces
    df.columns = [col.replace('"', '').strip() for col in df.columns]
    
    # Get the step column (first column) and method columns (all other columns)
    step_col = df.columns[0]
    method_columns = df.columns[1:]
    
    # Create the plot with fixed dimensions
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort methods by priority for legend ordering
    method_priority = []
    for method in method_columns:
        method_lower = method.lower()
        if ('4' in method_lower and 'poet' in method_lower):
            priority = 3
        elif ('2' in method_lower and 'poet' in method_lower):
            priority = 2
        elif 'galore' in method_lower:
            priority = 1
        else:
            priority = 0
        method_priority.append((method, priority))
    
    # Sort methods by priority (highest first)
    sorted_methods = [m[0] for m in sorted(method_priority, key=lambda x: x[1], reverse=True)]
    
    # Plot each method with custom styling in the sorted order
    for method in sorted_methods:
        # Simplify method names for the legend
        display_name = method
        if len(method) > 25 and "GB" in method:
            match = re.search(r'([^\(]+).*?(\d+\.\d+\s*GB)', method)
            if match:
                display_name = f"{match.group(1).strip()} ({match.group(2)})"
        
        # Get custom style for this method
        color, linestyle, marker, zorder = get_method_style(method)
        
        # Plot with appropriate style and zorder
        sns.lineplot(data=df, x=step_col, y=method, marker=marker, markersize=4, 
                     linewidth=2, label=display_name, ax=ax, 
                     color=color, linestyle=linestyle, zorder=zorder)
    
    # Extract model name from filename for title
    model_name = extract_model_name(csv_path)
    
    # Set labels and title
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title(model_name)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set consistent y-axis limits if provided
    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
    else:
        # Calculate reasonable y-axis limits based on data
        all_values = np.concatenate([df[col].values for col in method_columns])
        y_min = max(0, np.min(all_values) * 0.9)  # 10% below minimum
        y_max = np.max(all_values) * 1.1  # 10% above maximum
        ax.set_ylim(y_min, y_max)
    
    # Set consistent x-axis limits
    ax.set_xlim(df[step_col].min(), df[step_col].max())
    
    # Format step values on x-axis to use K for thousands
    format_step_ticks(ax, step_col)
    
    # Place legend inside the plot
    # Use the handles and labels to maintain the order we want
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Methods', loc='best')
    
    # Get output filename from input filename
    output_filename = Path(csv_path).stem
    
    # Save the figure with consistent dimensions
    output_path = output_dir / f"{output_filename}.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    plt.close()

def process_multiple_files(csv_paths, output_dir, use_common_ylim=True, figsize=(8, 6)):
    """
    Process multiple CSV files with consistent styling.
    
    Args:
        csv_paths: List of paths to CSV files
        output_dir: Directory to save plots
        use_common_ylim: Whether to use the same y-axis limits for all plots
        figsize: Tuple (width, height) for the figure size in inches
    """
    # Determine common y-axis limits if requested
    fixed_ylim = None
    if use_common_ylim and len(csv_paths) > 1:
        print("Calculating common y-axis limits...")
        all_min_values = []
        all_max_values = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, sep=';')
            df.columns = [col.replace('"', '').strip() for col in df.columns]
            method_columns = df.columns[1:]
            
            for col in method_columns:
                all_min_values.append(df[col].min())
                all_max_values.append(df[col].max())
        
        # Set common y-axis limits with some padding
        y_min = max(0, min(all_min_values) * 0.9)
        y_max = max(all_max_values) * 1.1
        fixed_ylim = (y_min, y_max)
        print(f"Using common y-axis limits: {fixed_ylim}")
    
    # Process each file
    for csv_path in csv_paths:
        create_simple_plot(csv_path, output_dir, fixed_ylim, figsize)

def main():
    parser = argparse.ArgumentParser(description="Generate consistent perplexity plots from CSV data")
    parser.add_argument("csv_paths", type=str, nargs='+', help="Path(s) to CSV file(s) with perplexity data")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--common_ylim", action="store_true", help="Use common y-axis limits for all plots")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 6], 
                        help="Figure size in inches (width height), default: 8 6")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process all files
    process_multiple_files(args.csv_paths, output_dir, args.common_ylim, tuple(args.figsize))
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()