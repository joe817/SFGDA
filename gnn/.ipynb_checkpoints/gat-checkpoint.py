import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros, uniform

import pickle

class CachedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 bias=True, dropout=0.0, **kwargs):
        super(CachedGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat

        self.cache_dict = {}

        self.weight = Parameter(torch.Tensor(in_channels, num_heads * out_channels))
        glorot(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(num_heads * out_channels))
            uniform(self.bias)
        else:
            self.register_parameter('bias', None)

        self.dropout = torch.nn.Dropout(dropout)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        x = torch.matmul(x, self.weight)

        if not cache_name in self.cache_dict:
            try:
                with open ('tmp/'+cache_name + str(x.device) +'.pkl', 'rb') as f:
                    self.cache_dict[cache_name] = pickle.load(f)
                edge_index, norm = self.cache_dict[cache_name]
            except: 
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                             improved=False, dtype=x.dtype)
                self.cache_dict[cache_name] = edge_index, norm
                with open ('tmp/'+cache_name + str(x.device) +'.pkl', 'wb') as f:
                    pickle.dump(self.cache_dict[cache_name], f)
        else:
            edge_index, norm = self.cache_dict[cache_name]

        x = self.propagate(edge_index, x=x, norm=norm)

        if self.concat:
            out = x.view(-1, self.num_heads, self.out_channels)
            return out
        else:
            return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return self.dropout(aggr_out)

    def __repr__(self):
        return '{}({}, {}, num_heads={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.num_heads)
