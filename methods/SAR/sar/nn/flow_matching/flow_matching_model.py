import numpy as np
import torch

from ..model import Model


class FlowMatchingModel(Model):
    def __init__(
        self,
    ) -> None:
        """
        Initialize the FlowMatchingModel.

        Args:
            *args: Variable length argument list for the base Model.
            **kwargs: Arbitrary keyword arguments for the base Model.
        """
        super().__init__()
        self.sigma_min = 0.001
        self.num_features = None # Must be set by the child class

    def flow(
        self,
        field_start: torch.Tensor,        # Shape: (num_nodes, features)
        field_end:   torch.Tensor,        # Shape: (num_nodes, features)
        r:           torch.Tensor,        # Shape: (batch_size,)
        batch:       torch.Tensor = None, # Shape: (num_nodes,)
    ) -> torch.Tensor:
        """
        Push function for the flow matching model.

        Args:
            field_start (torch.Tensor): Tensor of shape (num_nodes, d_emb), representing the initial field.
            field_end (torch.Tensor): Tensor of shape (num_nodes, d_emb), representing the target field.
            r (torch.Tensor): Tensor of shape (batch_size,), representing the interpolation parameter(s) between 0 and 1.
            batch (torch.Tensor, optional): Tensor of shape (num_nodes,), representing the batch assignment of the nodes.
        """
        r = r[batch].unsqueeze(-1) # Shape: (num_nodes, 1)
        return (1 - (1 - self.sigma_min) * r) * field_start + r * field_end # Shape: (num_nodes, d_emb)
    
    def advection_field(
        self,
        field_start: torch.Tensor, # Shape: (num_nodes, features) 
        field_end:   torch.Tensor, # Shape: (num_nodes, features)
    ) -> torch.Tensor:
        """
        Compute the flow-matching (optimal transport) vector field between the start and end physical fields.

        Args:
            field_start (torch.Tensor): Tensor of shape (num_nodes, d_emb), representing the initial field.
            field_end (torch.Tensor): Tensor of shape (num_nodes, d_emb), representing the target field.

        Returns:
            torch.Tensor: The advection field of the same shape as the input fields.
        """
        return field_end - (1 - self.sigma_min) * field_start # Shape: (num_nodes, features)

    def sample(
        self,
        y:     torch.Tensor,     # (num_nodes, d_in)
        batch: torch.LongTensor, # (num_nodes,)
        steps: int = 10,
    ) -> torch.Tensor:
        """Generate samples by solving ODE using Euler integration.
        
        Args:
            y: Conditioning input features
            batch: Batch assignment for nodes
            steps: Number of integration steps
            
        Returns:
            Generated samples from target distribution
        """
        batch_size = batch.max().item() + 1
        # Create time steps for integration
        steps_array = np.linspace(0, 1, steps+1)[:-1]
        try: 
            dr = steps_array[1] - steps_array[0]  # Time step size
        except:
            dr = 1
        device = y.device
        # Initialize from standard normal distribution
        x = torch.randn(y.size(0), self.num_features, device=device)
        # Euler integration of ODE
        for r in steps_array:
            r_values = torch.full((batch_size,), r, device=device)
            x += dr * self(x, y, r_values, batch)
        return x 