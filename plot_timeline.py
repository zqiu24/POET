import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib import rcParams
import matplotlib.font_manager as fm
import numpy as np
import os

# Create plots directory if it doesn't exist
if not os.path.exists('./plots'):
    os.makedirs('./plots')

# Define the tasks and their timeframes
tasks = [
    # Sub-project 1
    {"name": "Sub-project 1", "start": -0.5, "duration": 4, "is_subproject": True},
    {"name": "SP1_Design", "start": -0.5, "duration": 2, "is_subproject": False, "parent": "Sub-project 1", "display_name": "Design and Implementation"},
    {"name": "SP1_Deployment", "start": 1, "duration": 2.5, "is_subproject": False, "parent": "Sub-project 1", "display_name": "Deployment and Testing"},
    {"name": "SP1_Optimization", "start": 1, "duration": 1.5, "is_subproject": False, "parent": "Sub-project 1", "display_name": "Optimization"},
    {"name": "SP1_Documentation", "start": 0.5, "duration": 3, "is_subproject": False, "parent": "Sub-project 1", "display_name": "Documentation"},
    
    # Sub-project 2
    {"name": "Sub-project 2", "start": -0.5, "duration": 5, "is_subproject": True},
    {"name": "SP2_Design", "start": -0.5, "duration": 3, "is_subproject": False, "parent": "Sub-project 2", "display_name": "Design and Implementation"},
    {"name": "SP2_Deployment", "start": 1.5, "duration": 3, "is_subproject": False, "parent": "Sub-project 2", "display_name": "Deployment and Testing"},
    {"name": "SP2_Optimization", "start": 2, "duration": 1.5, "is_subproject": False, "parent": "Sub-project 2", "display_name": "Optimization"},
    {"name": "SP2_Documentation", "start": 0.5, "duration": 4, "is_subproject": False, "parent": "Sub-project 2", "display_name": "Documentation"},
    
    # Sub-project 3
    {"name": "Sub-project 3", "start": 2.5, "duration": 4, "is_subproject": True},
    {"name": "SP3_Design", "start": 2.5, "duration": 3, "is_subproject": False, "parent": "Sub-project 3", "display_name": "Design and Implementation"},
    {"name": "SP3_Deployment", "start": 3.5, "duration": 3, "is_subproject": False, "parent": "Sub-project 3", "display_name": "Deployment and Testing"},
    {"name": "SP3_Optimization", "start": 4, "duration": 1.5, "is_subproject": False, "parent": "Sub-project 3", "display_name": "Optimization"},
    {"name": "SP3_Documentation", "start": 2.5, "duration": 4, "is_subproject": False, "parent": "Sub-project 3", "display_name": "Documentation"}
]

# Define colors for sub-projects with specified colors
colors = {
    "Pre-project preparation": "#d62728",  # Red
    "Sub-project 1": "#1f77b4",  # Blue
    "Sub-project 2": "#d62728",  # Red
    "Sub-project 3": "#2ca02c"   # Green
}

# Define full names for the legend
subproject_full_names = {
    "Sub-project 1": "Memory-Efficient Pretraining via Orthogonal Equivalence Transformation",
    "Sub-project 2": "Normalized Neural Architectures for Language Modeling",
    "Sub-project 3": "Advancing Open-Source Language Models through Efficient Pretraining"
}

# Create the figure and axis with the same dimensions
fig, ax = plt.subplots(figsize=(12, 8))

# Create display names for y-axis labels that include the parent project
display_names = []
for task in tasks:
    if task["is_subproject"]:
        # For subprojects, just use the name
        display_names.append(task["name"])
    else:
        # For tasks, include the parent project
        parent_prefix = task["parent"].split()[1]  # Gets "1", "2", or "3"
        if "display_name" in task:
            # Use display name but add parent prefix
            display_names.append(f"SP{parent_prefix}: {task['display_name']}")
        else:
            # Just use the name
            display_names.append(task["name"])

# Create a list of unique labels (to avoid duplicates for split tasks)
unique_labels = []
seen = set()
for name in display_names:
    if name not in seen:
        unique_labels.append(name)
        seen.add(name)

# Map each task to its position in the unique labels list
position_map = {}
for i, task in enumerate(tasks):
    if task["is_subproject"]:
        # For subprojects, just use the name
        display_name = task["name"]
    else:
        # For tasks, we need to use the same prefix format as we did above
        parent_prefix = task["parent"].split()[1]
        if "display_name" in task:
            display_name = f"SP{parent_prefix}: {task['display_name']}"
        else:
            display_name = task["name"]
    
    position_map[i] = unique_labels.index(display_name)

# Create y-positions based on the unique labels
num_unique = len(unique_labels)
vertical_spacing = 0.6  # Reduced from default 1.0 to decrease distance between bars
start = 1 + (num_unique - 1) * vertical_spacing
y_positions_base = np.linspace(start, 1, num_unique)

# Map each task to its actual y-position
y_positions = []
for i in range(len(tasks)):
    position_idx = position_map[i]
    y_positions.append(y_positions_base[position_idx])

# Configure the plot with larger fonts
ax.set_yticks(y_positions_base)
ax.set_yticklabels(unique_labels, fontsize=11)

# Configure x-axis with larger fonts
ax.set_xticks(np.arange(-0.5, 7, 1), minor=False)
ax.set_xticks(np.arange(0, 7), minor=True)
ax.set_xticklabels([], minor=False)
ax.set_xticklabels(['Pre-project', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'], 
                  minor=True, fontsize=12)

# Set tick parameters to show only the bottom major ticks (at boundaries)
ax.tick_params(axis='x', which='major', length=6, width=1, direction='out')
ax.tick_params(axis='x', which='minor', length=0)  # Hide minor ticks but keep labels

# Add vertical lines at month boundaries (same position as major ticks)
for x in np.arange(-0.5, 6.5 + 1):
    ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5, zorder=0)

# Turn off the grid that we had previously
ax.grid(False)

# Compress the y-axis to reduce spacing
ax.set_ylim(0.5, max(y_positions_base) + 0.5)

# Add title and labels with larger font
ax.set_title('Project Timeline', fontsize=20, fontweight='bold')

# Create a list to store legend handles
legend_handles = []

# Add the task bars
for i, task in enumerate(tasks):
    start = task["start"]
    duration = task["duration"]
    y_pos = y_positions[i]
    
    # Determine the color
    if task["is_subproject"]:
        color = colors[task["name"]]
        alpha = 0.8
        edge_color = 'black'
        linewidth = 1.5
        height = 0.45  # Reduced height for thinner bars
        
        # Add to legend if it's a subproject
        legend_patch = patches.Patch(
            facecolor=color, 
            edgecolor=edge_color,
            alpha=alpha,
            label=f"{task['name']}: {subproject_full_names[task['name']]}"
        )
        legend_handles.append(legend_patch)
    else:
        # Get parent sub-project directly from the task data
        parent = task["parent"]
        color = colors[parent]
        alpha = 0.5  # Lighter tone for tasks
        edge_color = color
        linewidth = 1
        height = 0.32  # Reduced height for thinner bars
    
    # Create the bar
    rect = patches.Rectangle(
        (start, y_pos - height/2),  # Adjusted start position to align with month bins
        duration,
        height,
        linewidth=linewidth,
        edgecolor=edge_color,
        facecolor=color,
        alpha=alpha
    )
    ax.add_patch(rect)
    
    # Add text in the middle of the bar 
    text_x = start + duration / 2
    
    # Only add text if the bar is wide enough
    if duration >= 1.5:  # Skip text for very narrow bars
        # Use display_name if available
        display_text = task.get("display_name", task["name"])
        ax.text(text_x, y_pos, display_text, ha='center', va='center',
                color='white', fontweight='bold' if task["is_subproject"] else 'normal',
                fontsize=11)

# Add legend with larger font
plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
           ncol=1, frameon=True, fontsize=13)
           
# Adjust layout to compress vertical spacing
plt.tight_layout()
plt.subplots_adjust(bottom=0.22, hspace=0.1)  # Added hspace to reduce vertical spacing

# Save the figure with more appropriate padding
plt.savefig('./plots/project_timeline.pdf', bbox_inches='tight', pad_inches=0.2)
plt.savefig('./plots/project_timeline.png', dpi=300, bbox_inches='tight', pad_inches=0.2)

print("Gantt chart saved as './plots/project_timeline.pdf' and './plots/project_timeline.png'")

# plt.show()