import plotly.graph_objects as go
import pandas as pd
import os

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# Define sub-projects with their full titles
sub_projects = {
    "Sub-Project 1": "Sub-Project 1: Memory-Efficient Pretraining via Orthogonal Equivalence Transformation",
    "Sub-Project 2": "Sub-Project 2: Hyperspherical Neural Architectures for Language Modeling"
}

# Define tasks with discrete month-based timing (start month, duration in months)
# Now organized by sub-projects with adjusted positions for Sub-Project 2
tasks_data = [
    # Sub-Project 1
    {"Task": "Project Initiation", "Start": 1, "Duration": 1, "Phase": "Planning Phase", "SubProject": "Sub-Project 1"},
    {"Task": "Requirements Gathering", "Start": 1, "Duration": 2, "Phase": "Planning Phase", "SubProject": "Sub-Project 1"},
    {"Task": "System Design", "Start": 2, "Duration": 1, "Phase": "Design Phase", "SubProject": "Sub-Project 1"},
    {"Task": "Development", "Start": 3, "Duration": 2, "Phase": "Implementation Phase", "SubProject": "Sub-Project 1"},
    {"Task": "Testing", "Start": 5, "Duration": 1, "Phase": "Verification Phase", "SubProject": "Sub-Project 1"},
]

# Create second sub-project tasks separately
tasks_data_2 = [
    # Sub-Project 2 - with adjusted y-positions
    {"Task": "Requirements Analysis", "Start": 2, "Duration": 1, "Phase": "Planning Phase", "SubProject": "Sub-Project 2"},
    {"Task": "Architecture Design", "Start": 3, "Duration": 1, "Phase": "Design Phase", "SubProject": "Sub-Project 2"},
    {"Task": "Implementation", "Start": 4, "Duration": 2, "Phase": "Implementation Phase", "SubProject": "Sub-Project 2"},
    {"Task": "Integration", "Start": 5, "Duration": 1, "Phase": "Implementation Phase", "SubProject": "Sub-Project 2"},
    {"Task": "Deployment", "Start": 6, "Duration": 1, "Phase": "Deployment Phase", "SubProject": "Sub-Project 2"}
]

# Convert to DataFrames and calculate end month
df1 = pd.DataFrame(tasks_data)
df1['End'] = df1['Start'] + df1['Duration']
df1['Task_Display'] = df1.apply(lambda x: x['Task'] + '      ', axis=1)
df1['Group'] = 1  # Add group identifier

df2 = pd.DataFrame(tasks_data_2)
df2['End'] = df2['Start'] + df2['Duration']
df2['Task_Display'] = df2.apply(lambda x: x['Task'] + '      ', axis=1)
df2['Group'] = 2  # Add group identifier

# Create multiple spacers between the groups
spacers = []
for i in range(4):  # Adding four spacers
    spacer = pd.DataFrame([{
        'Task': f'SPACER{i}',
        'Start': 0,
        'Duration': 0,
        'Phase': '',
        'SubProject': 'SPACER',
        'End': 0,
        'Task_Display': '      ',
        'Group': 1.5 + (i * 0.1)  # Groups: 1.5, 1.6, 1.7, 1.8
    }])
    spacers.append(spacer)

# Combine all dataframes
df = pd.concat([df1] + spacers + [df2], ignore_index=True)
df = df.sort_values(by='Group')  # Sort by group to maintain order

# Professional color palette
phase_colors = {
    'Planning Phase': '#1f77b4',      # Blue
    'Design Phase': '#ff7f0e',        # Orange
    'Implementation Phase': '#2ca02c', # Green
    'Verification Phase': '#d62728',   # Red
    'Deployment Phase': '#9467bd'      # Purple
}

# Create a more professional Gantt chart using go.Figure
fig = go.Figure()

# Add tasks as horizontal bars
for i, task in df.iterrows():
    # Skip the spacer row
    if task['SubProject'] == 'SPACER':
        continue
        
    fig.add_trace(go.Bar(
        x=[task['Duration']],
        y=[task['Task_Display']],
        orientation='h',
        base=task['Start'],
        marker_color=phase_colors[task['Phase']],
        marker_line_width=1,
        marker_line_color='white',
        hoverinfo='text',
        hovertext=f"{task['Task']}<br>Duration: {task['Duration']} month(s)<br>Phase: {task['Phase']}<br>{sub_projects[task['SubProject']]}",
        showlegend=False
    ))

# Add phase information for legend
for phase, color in phase_colors.items():
    fig.add_trace(go.Bar(
        x=[None],
        y=[None],
        orientation='h',
        marker_color=color,
        name=phase,
        hoverinfo='none'
    ))

# Update layout for professional appearance with grid lines at month boundaries
fig.update_layout(
    title={
        'text': 'Project Implementation Timeline',
        'font': {'size': 24, 'family': 'Arial', 'color': '#333333'},
        'x': 0.5,
        'xanchor': 'center',
        'y': 0.98,
        'yanchor': 'top',
    },
    xaxis=dict(
        title=None,
        tickmode='array',
        tickvals=[],
        ticktext=[],
        range=[1, 7],
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        title='',
        autorange='reversed',
        showgrid=False,
        domain=[0.05, 0.95],
    ),
    height=650,  # Further reduced height
    width=1000,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=220, r=120, t=30, b=10),  # Significantly reduced bottom margin
    legend_title='Project Phases',
    legend=dict(
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='right',
        x=1.15,
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=12),
        bordercolor='#CCCCCC',
        borderwidth=1
    ),
    barmode='overlay',
    bargap=0.4,  # Adjusted gap between bars
)

# Add milestone markers at the end of each task
for i, task in df.iterrows():
    # Skip the spacer row
    if task['SubProject'] == 'SPACER':
        continue
        
    fig.add_trace(go.Scatter(
        x=[task['End'] - 0.1],
        y=[task['Task_Display']],
        mode='markers',
        marker=dict(symbol='diamond', size=10, color='#555555'),
        hoverinfo='text',
        hovertext=f"Milestone: {task['Task']} Complete",
        showlegend=False
    ))

# Get tasks for each sub-project
sp1_tasks = df[df['SubProject'] == 'Sub-Project 1']['Task_Display'].tolist()
sp2_tasks = df[df['SubProject'] == 'Sub-Project 2']['Task_Display'].tolist()

# Calculate the top and bottom positions for each sub-project
sp1_top = sp1_tasks[0]
sp1_bottom = sp1_tasks[-1]
sp2_top = sp2_tasks[0]
sp2_bottom = sp2_tasks[-1]

# Add vertical lines for month boundaries (grid lines) - only within task areas
for month in range(1, 8):
    # Add line for sub-project 1
    fig.add_shape(
        type='line',
        x0=month,
        y0=sp1_bottom,  # Bottom of sub-project 1
        x1=month,
        y1=sp1_top,  # Top of sub-project 1
        line=dict(color='#CCCCCC', width=1.5),
        layer='below'
    )
    
    # Add line for sub-project 2
    fig.add_shape(
        type='line',
        x0=month,
        y0=sp2_bottom,  # Bottom of sub-project 2
        x1=month,
        y1=sp2_top,  # Top of sub-project 2
        line=dict(color='#CCCCCC', width=1.5),
        layer='below'
    )

# Add month period annotations below the chart
for month in range(1, 7):
    fig.add_annotation(
        x=month + 0.5,
        y=sp2_bottom,  # Position below the bottom-most task
        text=f"Month {month}",
        showarrow=False,
        font=dict(size=14, color="#333333"),
        yshift=-40  # Increased from 40 to 80 to position labels well below the Deployment task
    )

# Add colored backgrounds for each sub-project
# Sub-project 1
fig.add_shape(
    type="rect",
    x0=0.5,
    y0=sp1_bottom,
    x1=7.5,
    y1=sp1_top,
    fillcolor="#E6F7FF",  # Light blue
    line=dict(width=1, color="#CCCCCC"),
    layer="below"
)

# Sub-project 2
fig.add_shape(
    type="rect",
    x0=0.5,
    y0=sp2_bottom,
    x1=7.5,
    y1=sp2_top,
    fillcolor="#FFF7E6",  # Light orange
    line=dict(width=1, color="#CCCCCC"),
    layer="below"
)

# Add the sub-project titles
fig.add_annotation(
    x=4,
    y=sp1_top,
    text=f"<b>{sub_projects['Sub-Project 1']}</b>",
    showarrow=False,
    font=dict(size=14, color="#333333", family="Arial"),
    bgcolor="#E6F7FF",
    bordercolor="#CCCCCC",
    borderwidth=1,
    borderpad=4,
    yshift=25,
    xanchor="center",
    yanchor="bottom",
    opacity=0.9
)

fig.add_annotation(
    x=4,
    y=sp2_top,
    text=f"<b>{sub_projects['Sub-Project 2']}</b>",
    showarrow=False,
    font=dict(size=14, color="#333333", family="Arial"),
    bgcolor="#FFF7E6",
    bordercolor="#CCCCCC",
    borderwidth=1,
    borderpad=4,
    yshift=-20,  # Changed to negative value to move down instead of up
    xanchor="center",
    yanchor="bottom",
    opacity=0.9
)

# Save as PDF (high quality) in the plots folder
pdf_path = os.path.join(plots_dir, 'project_timeline.pdf')
fig.write_image(pdf_path, scale=2)
print(f"Saved PDF to: {pdf_path}")

# Also save as PNG for easy sharing
png_path = os.path.join(plots_dir, 'project_timeline.png')
fig.write_image(png_path, scale=2)
print(f"Saved PNG to: {png_path}")

# Display the chart
# fig.show()
