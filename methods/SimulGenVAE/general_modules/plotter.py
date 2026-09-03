import matplotlib.pyplot as plt
import numpy as np

def temporal_plotter(data, axis, a, c, print_graph, n):
    """Plot temporal evolution at selected nodes or parameters."""
    if axis==0:
        for i in range(n):
            plt.plot(data[a+i, :, c], label = f'param_{a+i}')
    elif axis==2:
        for i in range(n):
            plt.plot(data[a, :, c+i], label = f'node_{c+i}')

    plt.legend()
    plt.title('Temporal Evolution')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    if print_graph != "0":
        plt.show()

def nodal_plotter(data, axis, a, c, print_graph, n, time_idx=None):
    """Plot nodal (spatial) distribution at selected time steps or parameters."""
    if time_idx is None:
        time_idx = data.shape[1] // 2  # Middle time step
    
    plt.figure(figsize=(12, 6))
    
    if axis==0:  # Different parameters at same time and node range
        for i in range(n):
            plt.plot(data[a+i, time_idx, :], '.', label=f'param_{a+i}', markersize=1)
    elif axis==1:  # Same parameter, different time steps
        for i in range(n):
            t_idx = min(time_idx + i*10, data.shape[1]-1)  # Offset time indices
            plt.plot(data[a, t_idx, :], '.', label=f't_{t_idx}', markersize=1)
    
    plt.legend()
    plt.title(f'Nodal Distribution (t={time_idx})')
    plt.xlabel('Node Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    if print_graph != "0":
        plt.show()

def dual_view_plotter(data, param_idx=0, node_indices=None, time_indices=None, print_graph="1"):
    """Create dual view plot showing both temporal and nodal perspectives."""
    if node_indices is None:
        num_nodes = data.shape[2]
        node_indices = [num_nodes//4, num_nodes//2, 3*num_nodes//4]
    
    if time_indices is None:
        num_time = data.shape[1]
        time_indices = [num_time//4, num_time//2, 3*num_time//4]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Dual View - Parameter {param_idx}', fontsize=14)
    
    # Temporal view (left): Time evolution at selected nodes
    axes[0].set_title('Temporal View - Time Evolution')
    colors = ['blue', 'green', 'red']
    for i, node_idx in enumerate(node_indices):
        temporal_trace = data[param_idx, :, node_idx]
        axes[0].plot(temporal_trace, '-', color=colors[i], label=f'node_{node_idx}')
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Nodal view (right): Spatial distribution at selected times
    axes[1].set_title('Nodal View - Spatial Distribution')
    for i, time_idx in enumerate(time_indices):
        nodal_slice = data[param_idx, time_idx, :]
        axes[1].plot(nodal_slice, '.', color=colors[i], label=f't_{time_idx}', markersize=1)
    axes[1].set_xlabel('Node Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if print_graph != "0":
        plt.show()
    
    return fig