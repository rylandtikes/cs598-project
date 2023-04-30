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
        # Attention
        # Linear layers for multi-head attention
        self.W = clones(
            nn.Linear(in_features, hidden_features),
            num_of_heads
            )
        # Parameters for multi-head attention
        self.a = clone_params(
            nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True),
            num_of_heads
            )
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # FFN applied after attention
        self.linear = nn.Linear(hidden_features, out_features)
        if concat:
            self.ffn = nn.Sequential(
                nn.Dropout(dropout),
                LayerNorm(hidden_features),
                nn.ELU()
                # linear is not present here in official code (should it be?)
            )
        else:
            self.ffn = nn.Sequential(
                nn.Dropout(dropout),
                LayerNorm(hidden_features),
                nn.ReLU(),
                self.linear
            ) 

    def initialize(self, init_method=nn.init.xavier_normal_):
        """Set initial weights in model.
        """
        for i in range(len(self.W)):
            init_method(self.W[i].weight.data)
        for i in range(len(self.a)):
            init_method(self.a[i].data)
        if not self.concat:
            init_method(self.linear.weight.data)

    def attention(self, linear, a, data, edges):
        """Updates representation by graph propagation with self-attention.

            Uses 'Method 2': the inner product of a learnable vector and the concatenation of two
            relevant representations.

        Args:
            linear (torch.Module): linear layer
            a (torch.Parameter): parameters of attention
            data (torch.Tensor): representation of EHR codes,
                                 of shape (self.num_of_nodes, embedding dim)
            edge (torch.Tensor): edge connectivity vector for observed codes, of shape (2, v_obs^2)

        Returns:
            torch.Tensor: updated representation of EHR codes after attenention from graph
        """
        data = linear(data).unsqueeze(0)
        # h: self.num_of_nodes x out
        h = torch.cat(
            (data[:, edges[0, :], :],
             data[:, edges[1, :], :]), dim=0)
        data = data.squeeze(0)
        # edge_h: 2*D x E
        edge_h = torch.cat(
            (h[0, :, :],
             h[1, :, :]), dim=1).transpose(0, 1)
        # d_h: dimensionality of embedding dimension
        d_h = np.sqrt(self.hidden_features * self.num_of_heads)
        # edge_e: E
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / d_h)
        edge_e = torch.sparse_coo_tensor(
            edges.to(device),
            edge_e.to(device),
            torch.Size([self.num_of_nodes, self.num_of_nodes])
            )
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(self.num_of_nodes, 1)).to(device))
        # e_rowsum: N x 1
        row_check = (e_rowsum == 0)
        e_rowsum[row_check] = 1
        zero_idx = row_check.nonzero()[:, 0]
        edge_e = edge_e.add(torch.sparse.FloatTensor(
                               zero_idx.repeat(2, 1),
                               torch.ones(len(zero_idx)).to(device),
                               torch.Size([self.num_of_nodes, self.num_of_nodes])
                               ))
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        return h_prime

    def forward(self, edges, data=None):
        """Uses fully connected graph to update representation of EHR codes via self-attention.

        Args:
            edge (torch.Tensor): edge connectivity vector for observed codes, of shape (2, v_obs^2)
            data (torch.Tensor): representation of EHR codes,
                                 of shape (self.num_of_nodes, embedding dim)

        Returns:
            torch.Tensor: updated representation of EHR codes
        """
        # TODO: separate forward for multi-head?

        # If using multi-head attention (self.num_of_heads > 1), compute attention coeffs multiple
        # times. If is input layer (self.concat), aggregate coeffs by concatenation.
        attn_list = []
        for w, a in zip(self.W, self.a):
            attn_out = self.attention(w, a, data, edges)
            attn_list.append(attn_out)
        if self.concat:
            h_prime = torch.cat(attn_list, dim=1)
        else:
            h_prime = torch.stack(attn_list, dim=0).mean(dim=0)
        # FFN
        return self.ffn(h_prime)


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
                nn.Dropout(dropout)
                )
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(linear_out_input, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1)
            )
        
        # Initialize encoder graph layers
        for i in range(n_layers):
            self.in_att[i].initialize()
        self.out_att.initialize() # Out attn intialization wasn't being performed previously

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
        observed = codes.nonzero() # observed EHR codes, of shape (num_nonzero, 1)
        if observed.size()[0] == 0:
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[n + 1], [n + 1]])
        if self.training:
            # Exclude observed codes with 0.05 probability
            mask = torch.rand(observed.size()[0])
            mask = mask > 0.05
            observed = observed[mask]
            if observed.size()[0] == 0:
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[n + 1], [n + 1]])
        # Input edges
        # Nodes, of shape (1, n), incremented by 1
        observed = observed.transpose(0, 1) + 1
        input_edges = self.make_fc_graph_edges(observed)
        # Output edges
        # Adds a node at the end with value n + 1. Will act as prediction node.
        observed = torch.cat((observed, torch.LongTensor([[n + 1]]).to(device)), dim=1)
        output_edges = self.make_fc_graph_edges(observed)

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
        logits = self.out_layer(out_batch)
        kld_sum = torch.sum(torch.stack(kld_batch))
        return logits, kld_sum
