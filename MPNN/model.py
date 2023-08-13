
# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 23, 2019
# // PURPOSE     : contains network that uses GNN
# // SPECIAL NOTES:
# // ===============================
# // Change History: 1.0: a simple network including GatedGraphConv
# // Change History: 1.3: Added: model for soft constraints
# // Change History: 1.6: Added: Handling minibatches.
# // Change History: 2.0: Added: mean correction and gaussian correction
# //
# //==================================
__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

import torch
import torch.nn.functional as F

from torch.func import grad, vmap
from torch.distributions import MultivariateNormal
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
import torch_scatter as ts

device = torch.device('cuda')

k = 0
def multivariate_mean_variance(means, sigmas):
    n = len(sigmas)

    A = torch.inverse(torch.diag(torch.pow(sigmas[:-1], -1)))
    B = torch.ones(n-1, n-1).to(device) * torch.pow(sigmas[-1], -1)

    covariance_matrix = A - 1/(1 + torch.trace(torch.matmul(B, A))) * torch.matmul(A ,torch.matmul(B, A))

    c = (k - means[-1])/sigmas[-1]
    reduced_mean = torch.matmul(covariance_matrix, torch.ones(n-1).to(device)*c + torch.div(means[:-1], sigmas[:-1]))

    return reduced_mean, covariance_matrix

def conditional_marginal(means, sigmas, X, i):
    conditional_mean = means[i] + sigmas[i]*(k - torch.sum(means))/torch.sum(sigmas)
    conditional_var = sigmas[i] - torch.square(sigmas[i])/torch.sum(sigmas)
    return (2 * torch.pi * conditional_var)**(-1/2) * torch.exp(-1/(2 * conditional_var) * (X[i] - conditional_mean)**2)

def sample_multivariate(reduced_mean, covariance_matrix):
    m = MultivariateNormal(reduced_mean, covariance_matrix)
    X = m.sample()
    X = torch.cat((X, torch.tensor([k - torch.sum(X)]).to(device)), dim=0)

    return X

## Need to check
def calculate_grad(pred, mu, sigma, constraint_mu, constraint_covariance):
    n = len(sigma)
    dx_dmu = torch.zeros(n, n).to(device)
    dx_dsigma = torch.zeros(n, n).to(device)
    for i in range(0, n):
        dx_dmu[i] = grad(conditional_marginal, argnums = 0)(mu, sigma, pred, i)
        dx_dsigma[i] = grad(conditional_marginal, argnums = 1)(mu, sigma, pred, i)

    return dx_dmu, dx_dsigma

class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma):
        constraint_mu, constraint_covariance = multivariate_mean_variance(mu, sigma)
        pred = sample_multivariate(constraint_mu, constraint_covariance)
        ctx.save_for_backward(pred, mu, sigma, constraint_mu, constraint_covariance)
        return pred

    @staticmethod
    def backward(ctx, grad_output):
        pred, mu, sigma, constraint_mu, constraint_covariance = ctx.saved_tensors
        dx_dmu, dx_dsigma = calculate_grad(pred, mu, sigma, constraint_mu, constraint_covariance)
        dl_dmu = dx_dmu * grad_output
        dl_dsigma = dx_dsigma * grad_output
        return dl_dmu, dl_dsigma
    
class Net_gaussian_correction_with_sampling(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE):
        print("GNN_LAYERS = ", GNN_LAYERS)
        print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE = ", HIDDEN_FEATURES_SIZE)
        
        super(Net_gaussian_correction_with_sampling, self).__init__()
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE, bias=False) # for embedding
        self.conv1 = GatedGraphConv(HIDDEN_FEATURES_SIZE, GNN_LAYERS)
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.lin2 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.sigmoid(self.lin0(x)) # embedding
        x = self.conv1(x1, edge_index)
        x = F.relu(x)
        mu = self.lin1(x)
        sigma = self.softplus(self.lin2(x))

        mu = torch.squeeze(mu)
        sigma = torch.squeeze(sigma)

        pred = torch.empty_like(mu)
        for i in range(0, data.num_graphs):
            mu_sample = mu[data.batch == i]
            sigma_sample = sigma[data.batch == i]

            pred[data.batch == i] = Sample.apply(mu_sample, sigma_sample)

        return pred

class Net_gaussian_correction(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE):
        print("GNN_LAYERS = ", GNN_LAYERS)
        print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE = ", HIDDEN_FEATURES_SIZE)
        
        super(Net_gaussian_correction, self).__init__()
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE, bias=False) # for embedding
        self.conv1 = GatedGraphConv(HIDDEN_FEATURES_SIZE, GNN_LAYERS)
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.lin2 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.sigmoid(self.lin0(x)) # embedding
        x = self.conv1(x1, edge_index)
        x = F.relu(x)
        mu = self.lin1(x)
        sigma = self.softplus(self.lin2(x))
        uncorrected_mu = mu.clone()
        mu_all = ts.scatter_add(mu, data.batch, dim=0)
        sigma_all = ts.scatter_add(sigma, data.batch, dim=0)

        for i in range(0, data.num_graphs):
            mu[data.batch == i] = mu[data.batch == i] - mu_all[i] * (sigma[data.batch == i] / sigma_all[i])

        return mu.squeeze(1), x1.squeeze(1), sigma.squeeze(1), uncorrected_mu.squeeze(1)


# credit to https://github.com/rusty1s/pytorch_geometric/
class GatedGraphConv(MessagePassing):
    """The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            h = self.rnn(m, h)

        return h


    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)


