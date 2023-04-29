'''
This code is adapted from Variationally Regularized Graph-based
Representation Learning for Electronic Health Records (cited)
https://github.com/NYUMedML/GNN_for_EHR
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def clone_params(param, N):
    return nn.ParameterList([copy.deepcopy(param) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GraphLayer(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, num_of_nodes,
                 num_of_heads, dropout, alpha, concat=True):
        super(GraphLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_of_nodes = num_of_nodes
        self.num_of_heads = num_of_heads
        self.W = clones(nn.Linear(in_features, hidden_features), num_of_heads)
        self.a = clone_params(nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True), num_of_heads)
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        if not concat:
            self.V = nn.Linear(hidden_features, out_features)
        else:
            self.V = nn.Linear(num_of_heads * hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if concat:
            self.norm = LayerNorm(hidden_features)
        else:
            self.norm = LayerNorm(hidden_features)

    def initialize(self):
        for i in range(len(self.W)):
            nn.init.xavier_normal_(self.W[i].weight.data)
        for i in range(len(self.a)):
            nn.init.xavier_normal_(self.a[i].data)
        if not self.concat:
            nn.init.xavier_normal_(self.V.weight.data)
            nn.init.xavier_normal_(self.out_layer.weight.data)

    def attention(self, linear, a, N, data, edge):
        data = linear(data).unsqueeze(0)
        assert not torch.isnan(data).any()
        # edge: 2*D x E
        h = torch.cat((data[:, edge[0, :], :], data[:, edge[1, :], :]), dim=0)
        data = data.squeeze(0)
        # h: N x out
        assert not torch.isnan(h).any()
        # edge_h: 2*D x E
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        # edge: 2*D x E
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.hidden_features * self.num_of_heads))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        edge_e = torch.sparse_coo_tensor(edge.to(device), edge_e.to(device), torch.Size([N, N]))
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).to(device))
        # e_rowsum: N x 1
        row_check = (e_rowsum == 0)
        e_rowsum[row_check] = 1
        zero_idx = row_check.nonzero()[:, 0]
        edge_e = edge_e.add(
            torch.sparse.FloatTensor(zero_idx.repeat(2, 1), torch.ones(len(zero_idx)).to(device), torch.Size([N, N])))
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward(self, edge, data=None):
        N = self.num_of_nodes
        if self.concat:
            h_prime = torch.cat([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=1)
        else:
            h_prime = torch.stack([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=0).mean(
                dim=0)
        h_prime = self.dropout(h_prime)
        if self.concat:
            return F.elu(self.norm(h_prime))
        else:
            return self.V(F.relu(self.norm(h_prime)))


class VariationalGNN(nn.Module):
    """Graph-based neural network. Encoder-decoder (Enc-dec) or variationally regulated (VGNN).
    """

    def __init__(self, in_features, out_features, num_of_nodes, n_heads, n_layers,
                 dropout, alpha, variational=True, excluded_features=0):
        super(VariationalGNN, self).__init__()
        self.variational = variational
        self.num_of_nodes = num_of_nodes + 1 - excluded_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.excluded_features = excluded_features

        # Encoder
        self.embed = nn.Embedding(self.num_of_nodes, in_features, padding_idx=0)
        self.in_att = clones(
            GraphLayer(in_features, in_features, in_features, self.num_of_nodes,
                       n_heads, dropout, alpha, concat=True),
            n_layers)
        
        # Variational regularization
        self.parameterize = nn.Linear(out_features, out_features * 2)

        # Decoder
        self.out_att = GraphLayer(in_features, in_features, out_features, self.num_of_nodes,
                                  n_heads, dropout, alpha, concat=False)
        linear_out_input = out_features
        if excluded_features > 0:
            linear_out_input = out_features + out_features // 2
            self.features_ffn = nn.Sequential(
                nn.Linear(excluded_features, out_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout))
        self.out_layer = nn.Sequential(
            nn.Linear(linear_out_input, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1))
        
        # Initialize encoder graph layers
        for i in range(n_layers):
            self.in_att[i].initialize()

    @staticmethod
    def make_fc_graph_edges(nodes):
        """Creates a graph connectivity vector in COO format for a complete undirected graph from
           a given set of nodes.

        Args:
            nodes (torch.Tensor): nodes of shape (n)

        Returns:
            torch.Tensor: graph connectivity vector in COO format
        """
        n = nodes.size()[1]
        source = nodes.repeat(1, n) # non-zero nodes in order, repeated n times
        dest = nodes.repeat(n, 1).transpose(0, 1).contiguous().view((1, n ** 2))
        return torch.cat((source, dest), dim=0)

    def data_to_edges(self, codes):
        """Creates input and output graph connectivity vectors in COO format from a set of nodes.
           Both graphs are complete and undirected. The output graph consists of the input graph
           and one additional node for prediction.

        Args:
            codes (torch.Tensor): tensor of EHR codes for a single patient

        Returns:
            (torch.Tensor, torch.Tensor): input and output graph connectivity in COO format
        """
        n = codes.size()[0]
        nonzero = codes.nonzero() # nonzero: indices with non-zero codes, of shape (num_nonzero, 1)
        if nonzero.size()[0] == 0:
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[n + 1], [n + 1]])
        if self.training:
            # Exclude non-zero codes with 0.05 probability
            mask = torch.rand(nonzero.size()[0])
            mask = mask > 0.05
            nonzero = nonzero[mask]
            if nonzero.size()[0] == 0:
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[n + 1], [n + 1]])
        # Input edges
        # Nodes, of shape (1, n), incremented by 1
        nonzero = nonzero.transpose(0, 1) + 1
        input_edges = self.make_fc_graph_edges(nonzero)
        # Output edges
        # Adds a node at the end with value n + 1. Will act as prediction node.
        nonzero = torch.cat((nonzero, torch.LongTensor([[n + 1]]).to(device)), dim=1)
        output_edges = self.make_fc_graph_edges(nonzero)

        return input_edges.to(device), output_edges.to(device)

    def reparameterize(self, mu, logvar):
        """Sample latent variables from distribution to become input to decoder. Used in
           variational regularization.

        Args:
            mu (torch.Tensor): mean
            logvar (torch.Tensor): variance

        Returns:
            torch.Tensor: samples of latent distribution
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encoder_decoder(self, codes):
        """Passes batch through the Encoder-decoder network

            First, embeddings of medical concepts are made.
            Medical embeddings are processed by the encoder to form the graph representation. In
            each graph layer, the graph representation is updated via self-attention.
            If variational, a latent layer is added between the encoder and decoder to regularize
            the graph representation.
            The output of the encoder is then passed through the decoder, which consists of one
            layer and self-attention. The output is the representation of the predictive node.

        Args:
            codes (torch.Tensor): EHR codes for one patient of shape (num codes)

        Returns:
            (torch.Tensor, torch.Tensor): inference from decoder output of shape (embedding_dim)
                                          and KL divergence
        """
        # Embedding of medical concepts
        # h_prime: (num_of_nodes, embedding_dim)
        h_prime = self.embed(torch.arange(self.num_of_nodes).long().to(device))
        # Encoder
        # Observed nodes are masked with 95% probability and then made into complete undirected
        # graphs.
        # input_edges: input graph edge connection matrix of nodes after masking
        # output_edges: output graph edge connection matrix of input nodes and one additional node
        input_edges, output_edges = self.data_to_edges(codes)
        for attn in self.in_att:
            h_prime = attn(input_edges, h_prime)
        # If variational regularization is used, add linear layer and sample distribution
        if self.variational:
            h_prime = self.parameterize(h_prime).view(-1, 2, self.out_features)
            h_prime = self.dropout(h_prime)
            mu = h_prime[:, 0, :]
            # The variance is parameterized as an exponential to ensure non-negativity
            logvar = h_prime[:, 1, :]
            h_prime = self.reparameterize(mu, logvar)
            mu = mu[codes, :]
            logvar = logvar[codes, :]
        # Decoder
        # The last row of h_prime is the representation of an additional node for prediction
        h_prime = self.out_att(output_edges, h_prime)
        out = h_prime[-1]
        # If applying variational regularization, KL divergence is needed
        if self.variational:
            kld = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) / mu.size()[0]
        else:
            kld = torch.tensor(0.0).to(device)
        return out, kld

    def forward(self, batch):
        """Forward pass of VGNN or Enc-dec network.

           Steps:
           Pass nodes for each patient in batch through Encoder-decoder network, giving the
           prediction task embedding from the decoder output and the KL divergence.
           Determine logits (unnormalized predictions) concatenating the decoder output for all
           patients and passing through ReLU and a FFN (self.out_layer).
           Sum KL divergence for all patients.

           If any features are excluded from the graph, they are separated, and pass through
           self.features_ffn and then joined with the encoder-decoder output, which acts on the
           non-excluded features.

        Args:
            batch (torch.Tensor): batch of shape (batch size, num nodes)

        Returns:
            (torch.Tensor, torch.Tensor): logits and KL divergence
        """
        kld_batch = []
        included_batch = []
        if self.excluded_features == 0:
            for i in range(batch.size()[0]):
                out, kld = self.encoder_decoder(batch[i, :])
                included_batch.append(out)
                kld_batch.append(kld)
                out_batch = torch.stack(included_batch)
        else:
            excluded_batch = []
            for i in range(batch.size()[0]):
                excluded_nodes = torch.FloatTensor([batch[i, :self.excluded_features]]).to(device)
                excluded_batch.append(self.features_ffn(excluded_nodes))
                out, kld = self.encoder_decoder(batch[i, self.excluded_features:])
                included_batch.append(out)
                kld_batch.append(kld)
            out_batch = torch.cat((torch.stack(excluded_batch), torch.stack(included_batch)),
                                   dim=1)
        logits = self.out_layer(F.relu(out_batch))
        kld_sum = torch.sum(torch.stack(kld_batch))
        return logits, kld_sum
