import torch
from torch.nn import Parameter
from torch_geometric.nn import GINConv
from torch_geometric.utils import add_remaining_self_loops

import pickle

class CachedGINConv(GINConv):
    r"""Graph Isomorphism Network (GIN) operator as introduced in the
    `"How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathrm{MLP} \left( (1 + \epsilon) \cdot \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        eps (float, optional): If set to :obj:`True`, the model will learn
            an additive epsilon value. (default: :obj:`True`)
        train_eps (bool, optional): If set to :obj:`True`, the epsilon value
            will be learned during training. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 eps=0, train_eps=False, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = Parameter(torch.Tensor([eps]))
        self.train_eps = train_eps
        self.cache_dict = {}

        # MLP to learn node representations
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        if not cache_name in self.cache_dict:
            try:
                with open ('tmp/'+cache_name + str(x.device) +'.pkl', 'rb') as f:
                    self.cache_dict[cache_name] = pickle.load(f)
                edge_index, norm = self.cache_dict[cache_name]
            except: 
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
                self.cache_dict[cache_name] = edge_index, norm
                with open ('tmp/'+cache_name + str(x.device) +'.pkl', 'wb') as f:
                    pickle.dump(self.cache_dict[cache_name],f)
        else:
            edge_index, norm = self.cache_dict[cache_name]

        if self.train_eps:
            self.eps.data = torch.tensor([eps])

        x = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, norm=norm))

        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
