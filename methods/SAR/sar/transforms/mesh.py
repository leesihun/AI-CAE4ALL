import torch
from torch_geometric.utils import coalesce, remove_self_loops
from typing import Union
    
from ..graph import Graph


def edge_pruning(
    max_indegree: int,
    edge_index:   torch.LongTensor,
    pos:          torch.Tensor,
    edge_attr:    torch.Tensor = None,
) -> tuple[torch.LongTensor, torch.Tensor | None]:
    """Removes the longest edges incident to nodes with in-degree greater than max_indegree until the in-degree is equal to max_indegree.

    Args:
        max_indegree (int): Maximum in-degree allowed.
        edge_index (torch.LongTensor): Edge index tensor. Shape (2, num_edges).
        pos (torch.Tensor): Node position tensor. Shape (num_nodes, dim).
        edge_attr (torch.Tensor, optional): Edge attribute tensor. Shape (num_edges, num_edge_features). Defaults to None.

    Returns:
        tuple[torch.LongTensor, torch.Tensor | None]: The pruned edge index and edge attribute tensors (if provided).

    """
    assert edge_index.size(0) == 2, f'Expected edge_index to have shape (2, num_edges), got {edge_index.size()}'
    assert max_indegree > 0, f'Expected max_indegree to be greater than 0, got {max_indegree}'
    device = edge_index.device
    num_nodes = pos.size(0)
    indegree = torch.bincount(edge_index[1], minlength=num_nodes) # Compute the in-degree of each node
    mask = indegree > max_indegree                                # Mask of nodes with in-degree greater than max_in_degree. Shape (num_nodes,)
    # If there are no nodes with in-degree greater than max_in_degree, return the input edge_index and edge_attr
    if mask.sum() == 0:
        return edge_index, edge_attr
    masked_nodes = torch.arange(num_nodes, device=device)[mask]   # Nodes with in-degree greater than max_in_degree. Shape (num_masked_nodes,)
    senders = edge_index[0].split(indegree.tolist())              # Senders of each node. Shape (num_nodes,)
    masked_senders = [senders[i] for i in masked_nodes.tolist()]  # Senders of nodes with in-degree greater than max_in_degree. Shape (num_masked_nodes,)
    # Compute the edges to be removed
    edges_to_be_removed = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    # Iterate over the nodes with in-degree greater than max_in_degree
    for i, s in zip(masked_nodes, masked_senders):
        num_to_be_removed = indegree[i] - max_indegree # Number of edges to be removed
        # Get the neighbourhood-wise index of the longest edges
        lengths = torch.norm(pos[s] - pos[i], dim=1)                          # Shape (num_senders,)
        indices = torch.argsort(lengths, descending=True)[:num_to_be_removed] # Shape (num_to_be_removed,)
        # Compute the global index of the longest edges
        indices = indegree[:i].sum() + indices
        edges_to_be_removed[indices] = True
    # Remove the edges
    edge_index = edge_index[:, ~edges_to_be_removed]
    if edge_attr is not None:
        edge_attr = edge_attr[~edges_to_be_removed]
    return edge_index, edge_attr

    
def cells_to_edge_index(
    cells:        torch.LongTensor, # Shape (num_cells, max_num_nodes_per_cell)
    max_indegree: int = None,
    pos:          torch.Tensor = None,
) -> torch.LongTensor:
    """Converts a cell list to an edge index tensor.

    Args:
        cells (torch.LongTensor): Cell list tensor. Shape (num_cells, max_num_nodes_per_cell).
        max_indegree (int, optional): Maximum in-degree allowed. Defaults to None.
        pos (torch.Tensor, optional): Node position tensor. Shape (num_nodes, dim). Defaults to None.

    Returns:
        torch.LongTensor: Edge index tensor. Shape (2, num_edges).

    """
    max_num_nodes_per_cell = cells.shape[1]
    edge_index = torch.cat([
        cells[:, [i, i + 1]] for i in range(max_num_nodes_per_cell - 1)
    ], dim=0).T
    # Remove any edge with a negative or np.nan index
    mask = (edge_index >= 0).all(dim=0)
    edge_index = edge_index[:, mask]
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    # Make undirected graph
    edge_index = torch.cat([
        edge_index,
        edge_index[[1, 0]],
    ], dim=1)
    # Remove duplicated columns (edges) and sort them by increasing value of the second row
    edge_index = coalesce(edge_index, sort_by_row=False)
    # Edge pruning
    if max_indegree is not None:
        assert pos is not None, 'Expected pos to be provided when max_indegree is not None'
        edge_index = edge_pruning(max_indegree, edge_index, pos, edge_attr=None)[0]
    return edge_index  


def guillard_coarsening(
    pos:        torch.Tensor,
    edge_index: torch.Tensor,
    max_iter:   int = 5,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    """Performs the Guillard coarsening algorithm and clustering to create a LR graph and a mapping from HR (child nodes) to LR (parent) indices.

    Args:
        pos (torch.Tensor): Node position tensor. Shape (num_nodes, dim).
        edge_index (torch.Tensor): Edge index tensor. Shape (2, num_edges).
        max_iter (int, optional): Maximum number of iterations. Defaults to 5.

    Returns:
        tuple[torch.BoolTensor, torch.LongTensor]: Coarse mask and mapping from HR (child nodes) to LR (parent) indices.
    """
    assert pos.shape[0] > 1, f'Expected at least 2 nodes, got {pos.shape[0]}'
    num_nodes = edge_index.max().item() + 1
    row, col = edge_index
    # Determine the indegree of each node
    indegree = col.bincount()
    # Find the senders of each node
    senders = row.split(indegree.tolist())
    # Node-nested coarsening by Guillard
    coarse_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    for coarse_node, s in zip(coarse_mask, senders):
        if coarse_node: coarse_mask[s] = False
    # Create clusters by:
    # For each node with coarse_node==False, find its closest incoming neighbour with coarse_node==True.
    # For each node with coarse_node==True, the result is the node itself.
    parents = torch.arange(num_nodes, device=edge_index.device)
    for i, coarse_node, s in zip(range(num_nodes), coarse_mask, senders):
        if not coarse_node:
            dist = torch.norm(pos[i] - pos[s], dim=1)
            dist[~coarse_mask[s]] = float('inf')
            if dist.min() < float('inf'):
                parents[i] = s[dist.argmin()]
            else:
                parents[i] = -1 # No parent node found yet
    # For those nodes that have not found a parent, set the parent to the parent of the closest neighbour which has a parent
    iter = 0
    while (parents == -1).any():
        iter += 1
        for i, coarse_node, s in zip(range(num_nodes), coarse_mask, senders):
            if parents[i] == -1:
                s = s[parents[s] != -1]                    # Remove the neighbours that have not found a parent
                dist = torch.norm(pos[i] - pos[s], dim=1)  # Compute the distance to the remaining neighbours
                if dist.numel() > 0:
                    parents[i] = parents[s[dist.argmin()]] # Set the parent to the parent of the closest neighbour that has a parent. If no neighbour has a parent, the parent is -1
        if iter >= max_iter:
            print(pos.shape[0], coarse_mask.sum())
            print(edge_index)
            print(parents)
            raise RuntimeError(f'Maximum number of iterations reached in Guillard coarsening during cluster creation. The graph may contain isolated nodes.')
    idxHR_to_idxLR = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    idxHR_to_idxLR[coarse_mask] = torch.arange(coarse_mask.sum(), device=edge_index.device)
    idxHR_to_idxLR = idxHR_to_idxLR[parents] # This maps each high-res node (in high-res indices) to its parent node (in low-res indices
    return coarse_mask, idxHR_to_idxLR


def pool_edges(
    coarse_mask:       torch.BoolTensor,
    idxHR_to_idxLR:    torch.LongTensor,
    edge_index:        torch.LongTensor,
    max_indegree:      int = None,
    pos:               torch.Tensor = None,
) -> torch.LongTensor:
    """Pools the edges of a graph to create a lower-resolution graph. The pooling is performed by merging the edges of the high-resolution graph into a single edge in the lower-resolution graph, 
    i.e., the spatial connectivity is preserved.

    Args:
        coarse_mask (torch.BoolTensor): Mask of the lower-resolution nodes. Shape (num_nodes,).
        idxHR_to_idxLR (torch.LongTensor): Mapping from high-resolution to low-resolution indices. Shape (num_nodes,).
        edge_index (torch.LongTensor): Edge index tensor. Shape (2, num_edges).
        max_indegree (int, optional): Maximum in-degree allowed. Defaults to None.
        pos (torch.Tensor, optional): Node position tensor. Shape (num_nodes, dim). Defaults to None.

    Returns:
        torch.LongTensor: Edge index tensor of the lower-resolution graph. Shape (2, num_edges).
    """
    coarse_num_nodes = coarse_mask.sum().item() # Number of lower resolution nodes
    # Express `coarse_edge_index` in terms of the lower resolution indices
    coarse_edge_index = idxHR_to_idxLR[edge_index]
    # Remove the resulting self-loops
    coarse_edge_index = remove_self_loops(coarse_edge_index)[0]
    # Aggregate the resulting edges
    coarse_edge_index = coalesce(coarse_edge_index, num_nodes=coarse_num_nodes, sort_by_row=False)
    # Edge pruning
    if max_indegree is not None:
        assert pos is not None, 'Expected pos to be provided when max_indegree is not None'
        coarse_edge_index = edge_pruning(max_indegree, coarse_edge_index, pos)[0]
    # Checks
    if coarse_edge_index.numel() > 0:
        assert coarse_edge_index.min() >= 0
        assert coarse_edge_index.max() == coarse_num_nodes - 1
    else: 
        assert pos.shape[0] == 1, f'There are disconnected nodes on the lower resolution graph. Expected 1 node, got {pos.shape[0]}'
    return coarse_edge_index


def mesh_coarsening(
    pos_1:                 torch.Tensor,
    edge_index_1:          torch.Tensor,
    batch_1:               torch.Tensor = None,
    k:                     int          = None,
    max_indegree:          int          = None,
    rel_pos_scaling_lr:    float        = None,
    rel_pos_scaling_hr_lr: float        = None,
    scalar_rel_pos:        bool         = False,
) -> tuple[torch.BoolTensor, torch.Tensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    """Performs (Guillard) mesh coarsening on a graph. The coarsening is performed by dropping nodes, assigning non-dropped nodes to dropped nodes, and merging the edges to preserve the spatial connectivity.

    Args:
        pos_1 (torch.Tensor): Node position tensor. Shape (num_nodes, dim).
        edge_index_1 (torch.Tensor): Edge index tensor. Shape (2, num_edges).
        batch_1 (torch.Tensor, optional): Batch tensor. Shape (num_nodes,). Defaults to None.
        k (int, optional): Number of clusters for k-means clustering. Defaults to None.
        max_indegree (int, optional): Maximum in-degree allowed. Defaults to None.
        rel_pos_scaling_lr (float, optional): Scaling factor for the relative position in the lower-resolution graph. Defaults to None.
        rel_pos_scaling_hr_lr (float, optional): Scaling factor for the relative position between child and parent nodes. Defaults to None.
        scalar_rel_pos (bool, optional): Whether to use the scalar relative position (distance) or the vector relative position. Defaults to False (vector relative position).

    Returns:
        tuple[torch.BoolTensor, torch.Tensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]: Coarse mask, node position, mapping from HR to LR indices, edge index, edge attribute, relative position of each node with respect to the parent node, and batch tensor.
    """
    coarse_mask_2, idx1_to_idx2 = guillard_coarsening(pos_1, edge_index_1)
    pos_2 = pos_1[coarse_mask_2]
    edge_index_2 = pool_edges(
        coarse_mask       = coarse_mask_2,
        idxHR_to_idxLR    = idx1_to_idx2,
        edge_index        = edge_index_1,
        max_indegree      = max_indegree,
        pos               = pos_2,
    )
    edge_attr_2 = pos_2[edge_index_2[1]] - pos_2[edge_index_2[0]] # Relative position in the lower-resolution graph
    e_12 = pos_2[idx1_to_idx2] - pos_1 # Relative position of each node with respect to the parent node
    if scalar_rel_pos:
        e_12 = e_12.norm(dim=-1, keepdim=True)
    if rel_pos_scaling_hr_lr is not None:
        e_12 = e_12 / (2 * rel_pos_scaling_hr_lr)
    if rel_pos_scaling_lr is not None:
        edge_attr_2 = edge_attr_2 / (2 * rel_pos_scaling_lr)
    batch_2 = batch_1[coarse_mask_2] if batch_1 is not None else None
    return coarse_mask_2, pos_2, idx1_to_idx2, edge_index_2, edge_attr_2, e_12, batch_2


class MeshCoarsening:
    """Transform that performs multiple mesh coarsenings on a graph. The coarsening is performed by dropping nodes, assigning non-dropped nodes to dropped nodes, and merging the edges to preserve the spatial connectivity.

    Args:
        num_scales (int): Number of scales.
        method (str, optional): Coarsening method. One of ["guillard", "kmeans"]. Defaults to 'guillard'.
        k (list[Union[int, float]], optional): Number of clusters for k-means clustering. Defaults to None.
        max_indegree (int, optional): Maximum in-degree allowed. Defaults to None.
        rel_pos_scaling (list[float, None], optional): Scaling factor for the relative position in the lower-resolution graph. Defaults to None.
        scalar_rel_pos (bool, optional): Whether to use the scalar relative position (distance) or the vector relative position. Defaults to False (vector relative position).
    """

    def __init__(
        self,
        num_scales:      int,
        k:               list[Union[int, float]] = None,
        max_indegree:    int                     = None,
        rel_pos_scaling: list[float, None]       = None,
        scalar_rel_pos:  bool                    = False,
        scale_idx:       list[int]               = None,
    ) -> None:
        # assert num_scales > 1, f'Expected num_scales to be greater than 1, got {num_scales}'
        assert max_indegree is None or max_indegree > 0, f'Expected max_indegree to be greater than 0, got {max_indegree}'
        self.num_scales      = num_scales
        self.max_indegree    = max_indegree
        self.rel_pos_scaling = rel_pos_scaling if rel_pos_scaling is not None else [None] * num_scales
        self.scalar_rel_pos  = scalar_rel_pos
        self.k               = k if k is not None else [None] * (num_scales - 1)
        self.scale_idx       = scale_idx
        assert len(self.rel_pos_scaling) == num_scales, f'Expected rel_pos_scaling to have length {num_scales}, got {len(self.rel_pos_scaling)}'
        assert len(self.k) == num_scales - 1, f'Expected k to have length {num_scales - 1}, got {len(self.k)}'

    def __call__(
        self,
        graph: Graph
    ) -> Graph:
        graph.edge_index_1 = graph.edge_index
        if graph.batch is None:
            graph.batch = torch.zeros(graph.pos.size(0), dtype=torch.long, device=graph.pos.device)
            graph.batch_1 = graph.batch
        if self.num_scales == 1:
            graph.scale = torch.ones(graph.pos.size(0), dtype=torch.long, device=graph.pos.device)
            return graph
        # Initialize the scale attribute with all nodes at scale 1 (finest scale)
        device = graph.pos.device
        num_nodes = graph.pos.size(0)
        graph.scale = torch.ones(num_nodes, dtype=torch.long, device=device)
        graph.coarse_mask_2, graph.pos_2, graph.idx1_to_idx2, graph.edge_index_2, graph.edge_attr_2, graph.e_12, graph.batch_2 = mesh_coarsening(
            pos_1                 = graph.pos,
            edge_index_1          = graph.edge_index,
            batch_1               = graph.batch,
            k                     = self.k[0],
            max_indegree          = self.max_indegree,
            rel_pos_scaling_lr    = self.rel_pos_scaling[1],
            rel_pos_scaling_hr_lr = self.rel_pos_scaling[0],
            scalar_rel_pos        = self.scalar_rel_pos,
        )
        # For nodes preserved in scale 2, update their scale
        graph.scale[graph.coarse_mask_2] = 2
        # Map from original graph nodes to each scale
        # Keep track of original indices at each scale
        scale_to_original = {1: torch.arange(num_nodes, device=device)}
        scale_to_original[2] = torch.arange(num_nodes, device=device)[graph.coarse_mask_2]
        for i in range(2, self.num_scales):
            coarse_mask, pos, idx_to_parent, edge_index, edge_attr, e, batch = mesh_coarsening(
                pos_1                 = getattr(graph, f'pos_{i}'),
                edge_index_1          = getattr(graph, f'edge_index_{i}'),
                batch_1               = getattr(graph, f'batch_{i}'),
                k                     = self.k[i-1],
                max_indegree          = self.max_indegree,
                rel_pos_scaling_lr    = self.rel_pos_scaling[i],
                rel_pos_scaling_hr_lr = self.rel_pos_scaling[i-1],
                scalar_rel_pos        = self.scalar_rel_pos,
            )
            setattr(graph, f'coarse_mask_{i+1}', coarse_mask)
            setattr(graph, f'pos_{i+1}', pos)
            setattr(graph, f'idx{i}_to_idx{i+1}', idx_to_parent)
            setattr(graph, f'edge_index_{i+1}', edge_index)
            setattr(graph, f'edge_attr_{i+1}', edge_attr)
            setattr(graph, f'e_{i}{i+1}', e)
            if batch is not None: setattr(graph, f'batch_{i+1}', batch)
            # Find which nodes from scale i are preserved in scale i+1
            preserved_scale_i = torch.arange(getattr(graph, f'pos_{i}').size(0), device=device)[coarse_mask]
            # Map these preserved nodes back to original graph nodes
            preserved_original = scale_to_original[i][preserved_scale_i]
            # Update scale for these nodes
            graph.scale[preserved_original] = i + 1
            # Update mapping for next scale
            scale_to_original[i+1] = preserved_original
        if self.scale_idx is not None:
            # Replace the scale idx with the new index for each scale
            assert len(self.scale_idx) == self.num_scales, f'Expected scale_idx to have length {self.num_scales}, got {len(self.scale_idx)}'
            assert set(self.scale_idx) == set(range(1, max(self.scale_idx) + 1)), f'A scale cannot be skipped' 
            scale = graph.scale.clone()
            for old_scale, new_scale in zip(range(1, self.num_scales + 1), self.scale_idx):
                graph.scale[scale == old_scale] = new_scale
        return graph
    
class RandomizeScales:
    def __call__(
        self,
        graph: Graph
    ) -> Graph:
        # Permute the scale indices in graph.scale
        graph.scale = graph.scale[torch.randperm(graph.scale.size(0))]
        return graph