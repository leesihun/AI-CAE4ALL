import torch

from ...graph import Graph


class HierarchicalMasker:
    """Hierarchical mask creator for scale-based autoregressive graph prediction."""
    
    def __init__(self, num_iters: int, ref_scale: int = 1) -> None:
        """Initialize masker with number of hierarchical scales.
        
        Args:
            num_iters: Total number of autoregressive iterations.
        """
        super().__init__()
        self.num_iters  = num_iters
        self.ref_scale = ref_scale

    def __call__(self, graph: Graph, iters: torch.Tensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Generate masks for nodes to predict at the current iteration.
        
        Args:
            graph: Input graph with node features and hierarchy information.
            iters: Iteration numbers for each graph in batch.
        
        Returns:
            Tuple of masks:
            - unknown_mask: True for nodes that are unknown (scale <= next_scale)
            - target_mask: True for nodes at current prediction scale
        """
        assert torch.all(iters >= 0) and torch.all(iters < self.num_iters), f"iters must be in range [0, {self.num_iters - 1}]"
        batch = getattr(graph, f'batch_{self.ref_scale}', graph.batch)
        device = batch.device
        batch_size = batch.max().item() + 1
        assert batch_size == iters.size(0)        
        num_nodes = batch.size(0)
        # Create masks
        unknown_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        target_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        # Get scale
        scale = graph.scale[graph.scale >= self.ref_scale] - self.ref_scale + 1
        assert scale.size(0) == num_nodes
        assert scale.max() == self.num_iters
        assert scale.min() == 1
        # Iterate over the batch
        for i in range(batch_size):
            graph_mask = batch == i
            next_scale = self.num_iters - iters[i]
            unknown_mask[graph_mask & (scale <= next_scale)] = True
            target_mask [graph_mask & (scale == next_scale)] = True
        return unknown_mask, target_mask
    
    def sample(self, graph: Graph) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Sample random iteration numbers for each graph in batch. 
        In order to ensure balance memory usage during training, the iters 
        
        Args:
            graph: Input graph with node features and hierarchy information.
        
        Returns:
            Tuple of masks:
            - unknown_mask: True for nodes that are unknown (scale <= next_scale)
            - target_mask: True for nodes at current prediction scale
        """
        device = graph.batch.device
        batch_size = graph.batch.max() + 1
        iters = torch.arange(self.num_iters, device=device).repeat(batch_size // self.num_iters)
        rand_iters = torch.randint(0, self.num_iters, (batch_size % self.num_iters,), device=device)
        iters = torch.cat([iters, rand_iters])
        assert iters.size(0) == batch_size
        return iters, *self(graph, iters)