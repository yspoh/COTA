import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from utils import *
import ot
from torch.distributions.normal import Normal
from info_nce import InfoNCE
import collections
# from torchsom.core import SOM
# from torchsom.visualization import SOMVisualizer, VisualizationConfig


# ----- Model Components -----
class ClusterOTRecommender(nn.Module):
    def __init__(self, sencoder, tencoder, cluster_size, lambda_e, maxiter, num_experts, tau, usepmap, device):
        super().__init__()
        self.device = device
        self.cluster_size = cluster_size
        self.lambda_e = lambda_e
        self.maxiter = maxiter
        self.sencoder = sencoder
        self.tencoder = tencoder
        self.num_experts = num_experts
        self.info_nce_loss = InfoNCE()
        self.usepmap = usepmap
        self.tau = tau # How many explicit hard negatives to sample per user
        for p in self.sencoder.parameters(): p.requires_grad = False
        for p in self.tencoder.parameters(): p.requires_grad = False
        # Latent Prototypes (Target Domain Anchors)
        su_all, _ = self.sencoder()
        tu_all, _ = self.tencoder()

        self.skmeans, self.sp = prepare(su_all, cluster_size, device)
        _, self.tp = prepare(tu_all, cluster_size, device)

        self.cluster_map = nn.Sequential(
            nn.Linear(sencoder.embed_dim, sencoder.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(sencoder.embed_dim // 2, sencoder.embed_dim)
        )
        
        # MOE Integration
        self.moe = MoE(
            input_size=sencoder.embed_dim * 4,
            output_size=sencoder.embed_dim,
            gate_input_size=sencoder.embed_dim * 4,
            num_experts=num_experts,
            hidden_size=sencoder.embed_dim // 2,
            k=2,
            noisy_gating=True
        )

    def forward(self, su_idx, tu_idx, ti_idx, cluster=False):
        # Alternative Training (Abandoned if, always go else)
        if cluster:
            # Check if we actually have clusters to transport
            if self.sp.size(0) == 0:
                return self.sp, torch.tensor(0.0).to(self.device)

            # mappedsp, moe_loss = self.moe(torch.cat([self.sp, self.sp, self.sp], 1), self.sp)
            mappedsp = self.cluster_map(self.sp)

            wdloss, plan = WDLOSS(mappedsp, self.tp, self.lambda_e, self.maxiter, self.device)
            self.saved_transport_plan = plan.detach()

            return None, wdloss
        else :
            su_all, _ = self.sencoder()
            tu_all, ti_all = self.tencoder()
            # Select Batch
            seu = su_all[su_idx]
            teu = tu_all[tu_idx]
            tei = ti_all[ti_idx]
            seup, seup_labels = assign_prototype(self.skmeans, seu, self.device)

            wdloss = torch.tensor(0.0, device=self.device)
            if self.cluster_size>0 and self.usepmap:
                mappedsp = self.cluster_map(self.sp)
                wdloss, plan = WDLOSS(mappedsp, self.tp, self.lambda_e, self.maxiter, self.device)
                self.saved_transport_plan = plan

                # Get the transport probabilities for these specific users
                # Shape: [batch_size, num_target_clusters]
                # transport_probs = self.saved_transport_plan[seup_labels.detach()]
                logits = self.saved_transport_plan[seup_labels]
                # Normalize so the probabilities sum to 1 for each user (L1 Norm)
                # target_cluster_indices = torch.argmax(transport_probs, dim=1)
                # transport_probs = torch.nn.functional.normalize(transport_probs, p=1, dim=1)
                transport_probs = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
                # Math: [batch, num_clusters] @ [num_clusters, embed_dim] -> [batch, embed_dim]
                # sdestination = self.tp[target_cluster_indices]
                sdestination = torch.matmul(transport_probs, self.tp)
                mappedsp_input = mappedsp[seup_labels]
            else:
                sdestination = seup.detach()
                mappedsp_input = seup.detach()

            mapped_source, moe_loss = self.moe(torch.cat([seu, seup, sdestination, mappedsp_input], -1))

            # s_norm = torch.nn.functional.normalize(mapped_source, p=2, dim=1)
            # t_norm = torch.nn.functional.normalize(teu, p=2, dim=1)
            # nce_loss = self.info_nce_loss(query=s_norm, positive_key=t_norm, negative_keys=None)

            return mapped_source, teu, tei, moe_loss, None, wdloss
    
    def getAll(self, su_idx, si_idx, tu_idx, ti_idx):
        # get GCN embeddings
        su_all, _ = self.sencoder()
        tu_all, ti_all = self.tencoder()
        seu = su_all[su_idx]
        sei = su_all[si_idx]
        teu = tu_all[tu_idx]
        tei = ti_all[ti_idx]

        mappedsp, moe_loss_prototypes = self.moe(torch.cat([self.sp, self.sp, self.sp], 1))

        wdloss, plan = WDLOSS(mappedsp, self.tp, self.lambda_e, self.maxiter, self.device)
        self.saved_transport_plan = plan.detach()
        seu = su_all[su_idx]
        teu = tu_all[tu_idx]
        tei = ti_all[ti_idx]
        seup, seup_labels = assign_prototype(self.skmeans, seu, self.device)

        # Get the transport probabilities for these specific users
        # Shape: [batch_size, num_target_clusters]
        transport_probs = self.saved_transport_plan[seup_labels.detach()]
        # Normalize so the probabilities sum to 1 for each user (L1 Norm)
        transport_probs = torch.nn.functional.normalize(transport_probs, p=1, dim=1) 
        # Calculate the "Ideal" destination by weighted sum of target prototypes
        # Math: [batch, num_clusters] @ [num_clusters, embed_dim] -> [batch, embed_dim]
        sdestination = torch.matmul(transport_probs, self.tp)

        mapped_source, moe_loss = self.moe(torch.cat([seu, seup, sdestination], -1))

        return mappedsp, self.tp, wdloss, mapped_source, seu, sei, teu, tei


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        # out = self.soft(out)
        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, gate_input_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(gate_input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(gate_input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, gate_x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = gate_x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = gate_x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, gate_x=None, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        gate_x: (Optional) Separate input for the gating network
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        # If no specific gate_x is provided, default to x
        if gate_x is None:
            gate_x = x

        gates, load = self.noisy_top_k_gating(gate_x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss



class MFEncoder(nn.Module):
    def __init__(self, n_users, m_items, embed_dim):
        super(MFEncoder, self).__init__()
        self.num_users  = n_users
        self.num_items  = m_items
        self.embed_dim = embed_dim
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)
        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()

    def forward(self):
        return self.embedding_user.weight, self.embedding_item.weight
    
class MF(nn.Module):
    def __init__(self, n_users, m_items, latent_dim):
        super(MF, self).__init__()
        self.num_users  = n_users
        self.num_items  = m_items
        self.latent_dim = latent_dim
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.f = nn.Sigmoid()

    def forward(self, users, items):
        users_emb = self.embedding_user.weight[users.long()]
        items_emb = self.embedding_item.weight[items.long()]
        rating = self.f(torch.sum(users_emb * items_emb, dim=1, keepdim=True))
        # rating = torch.sum(users_emb * items_emb, dim=1, keepdim=True)
        return rating
    


class EMCDR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Sigmoid(),
            nn.Linear(input_dim * 2, input_dim)
        )
    def forward(self, x):
        return self.fc(x)
    

class pretrain_CMF(nn.Module):
    def __init__(self, encoder, tn_users, tm_items, embed_dim, overlap_src2tgt: dict):
        super(pretrain_CMF, self).__init__()
        self.s_emb, _ = encoder()

        self.target_user = torch.nn.Embedding(num_embeddings=tn_users, embedding_dim=embed_dim)
        self.target_item = torch.nn.Embedding(num_embeddings=tm_items, embedding_dim=embed_dim)

        self.overlap_src2tgt = overlap_src2tgt
        
        # Initialize target user embeddings with source embeddings for overlapping users
        self._initialize_shared_embeddings()

    def _initialize_shared_embeddings(self):
        """Initialize target user embeddings with source embeddings for overlapping users"""
        with torch.no_grad():
            for s_uid, t_uid in self.overlap_src2tgt.items():
                self.target_user.weight.data[t_uid] = self.s_emb[s_uid]

    def forward(self, user_ids):
        return self.target_user(user_ids)


class CMF(nn.Module):
    def __init__(self, n_users, m_items, tn_users, tm_items, embed_dim, overlap_src2tgt: dict):
        super(CMF, self).__init__()

        self.source_user = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=embed_dim)
        self.source_item = torch.nn.Embedding(num_embeddings=m_items, embedding_dim=embed_dim)

        self.target_user = torch.nn.Embedding(num_embeddings=tn_users, embedding_dim=embed_dim)
        self.target_item = torch.nn.Embedding(num_embeddings=tm_items, embedding_dim=embed_dim)

        self.overlap_src2tgt = overlap_src2tgt
        self.f = nn.Sigmoid()
        
        # Initialize target user embeddings with source embeddings for overlapping users
        self._initialize_shared_embeddings()

    def _initialize_shared_embeddings(self):
        """Initialize target user embeddings with source embeddings for overlapping users"""
        with torch.no_grad():
            for s_uid, t_uid in self.overlap_src2tgt.items():
                self.target_user.weight.data[t_uid] = self.source_user.weight.data[s_uid]

    def forward(self, user_ids, item_ids, domain='source'):
        if domain == 'source':
            user_emb = self.source_user(user_ids)
            item_emb = self.source_item(item_ids)
        elif domain == 'target':
            user_emb = self.target_user(user_ids)
            item_emb = self.target_item(item_ids)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        
        # Dot product prediction
        preds = torch.sum(user_emb * item_emb, dim=1)
        return self.f(preds)

    def update_shared_embeddings(self):
        """Method to explicitly update shared embeddings, call this during training"""
        with torch.no_grad():
            for s_uid, t_uid in self.overlap_src2tgt.items():
                # Option 1: Copy source to target (source domain leads)
                self.target_user.weight.data[t_uid] = self.source_user.weight.data[s_uid]



class InterDomainAggregation(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(InterDomainAggregation, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Remove manual projection layers and rely on MultiheadAttention's built-in projections
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.global_vector = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)
    def forward(self, vanilla_emb, transferred_emb):
        # Input shapes: [batch_size, embed_dim]
        batch_size = vanilla_emb.size(0)
        # print("check 1")
        # Prepare sequence: [4, batch_size, embed_dim]
        global_vec = self.global_vector.expand(batch_size, -1).unsqueeze(0)  # [1, batch_size, embed_dim]
        M_u = torch.stack([
            vanilla_emb.unsqueeze(0),
            transferred_emb.unsqueeze(0),
            global_vec
        ]).view(3, batch_size, -1)
        # print("check 2")
        # Let MultiheadAttention handle all projections internally
        # Use global_vec as the query for attention pooling
        attn_output, _ = self.attention(
            query=M_u,  # [4, batch_size, embed_dim]
            key=M_u,                          # [4, batch_size, embed_dim]
            value=M_u                         # [4, batch_size, embed_dim]
        )
        final_user_rep = attn_output[-1]  # [batch_size, embed_dim]
        return final_user_rep
