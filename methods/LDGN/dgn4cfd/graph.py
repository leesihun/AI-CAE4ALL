from torch_geometric.data import Data


class Graph(Data):
    r"""A data object describing a graph. Same as torch_geometric.data.Data.

    AI-CAE4ALL port: the upstream plotting methods (plot_pos, plot_field,
    plot_pos_field) were removed together with ``dgn4cfd/plot.py``
    (matplotlib helpers for the paper's 2-D examples); nothing in training or
    sampling calls them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = None
