"""
Neural_LP model (PyTorch)
A compact PyTorch implementation inspired by the Neural-LP architecture.
This model learns differentiable compositions of adjacency operators to answer queries.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse


def convert_matrix_db_to_torch_sparse(matrix_db, num_relation, num_entity, device):
    """Convert matrix_db dict to list of torch.sparse_coo_tensor on device."""
    mdb = []
    for r in range(num_relation):
        entry = matrix_db.get(r, ([[0, 0]], [0.], (num_entity, num_entity)))
        coords = entry[0]
        vals = entry[1]
        if len(coords) <= 1:
            indices = torch.LongTensor([[], []]).to(device)
            values = torch.FloatTensor([]).to(device)
            sp = torch.sparse_coo_tensor(indices, values, (num_entity, num_entity)).coalesce()
        else:
            rows = [c[0] for c in coords]
            cols = [c[1] for c in coords]
            indices = torch.LongTensor([rows, cols]).to(device)
            values = torch.FloatTensor(vals).to(device)
            sp = torch.sparse_coo_tensor(indices, values, (num_entity, num_entity)).coalesce()
        mdb.append(sp)
    return mdb

class NeuralLP(nn.Module):
    """
    Simple Neural-LP style model. Learns a set of composition weights and applies them to
    adjacency operator tensors to produce a score distribution over entities.

    Note: This is an initial implementation aiming for compatibility with the DRUM-style data loader.
    """
    def __init__(self, num_relation, num_entity, max_rule_len=3, num_rules=128, device=torch.device('cpu')):
        super(NeuralLP, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.max_rule_len = max_rule_len
        self.num_rules = num_rules
        self.device = device

        # We parameterize rule bodies using softmax weights over relations at each step for each rule
        # Shape: (num_rules, max_rule_len, num_relation)
        self.rule_logits = nn.Parameter(torch.randn(num_rules, max_rule_len, num_relation) * 0.1)

        # A small scalar temperature for sharper attention (optional)
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, heads, tails, mdb_torch):
        """
        queries: ignored in this simplified impl (we assume relational queries are encoded as relation indices in 'queries')
        heads, tails: tensors (batch,)
        mdb_torch: list of torch sparse adjacency matrices (num_relation entries), each size (num_entity, num_entity)
        Returns: scores (batch, num_entity)
        """
        batch = heads.size(0)
        device = self.device

        # Precompute per-relation sparse matrices as dense for matmul with entity vectors, if small enough
        # We'll create a function to apply one-step operator: vec -> A_r @ vec
        use_sparse = True

        def apply_op(r_idx, vec):
            A = mdb_torch[r_idx]
            if isinstance(A, torch.Tensor) and A.is_sparse:
                # A is (num_entity, num_entity) sparse, vec is (num_entity, batch) or (num_entity,)
                return torch.sparse.mm(A, vec)
            else:
                # dense
                return torch.matmul(A, vec)

        # For each rule, compute composed operator as weighted sum over relations at each step
        # We'll propagate batch of one-hot tail vectors through the composition to get predicted heads
        # tails_onehot: (batch, num_entity)
        tails_onehot = torch.zeros(batch, self.num_entity, device=device)
        tails_onehot.scatter_(1, tails.view(-1,1).to(device), 1.0)

        # Transpose to (num_entity, batch) for multiplication
        tails_vec = tails_onehot.t()

        # Rule attention weights (softmax over relations per step)
        logits = self.rule_logits / (torch.abs(self.temp) + 1e-6)
        att = F.softmax(logits, dim=2)  # (num_rules, max_rule_len, num_relation)

        # For each rule, start with tails_vec and sequentially apply weighted operator mixture
        # We'll aggregate rule outputs (num_entity, batch) across rules with equal weight
        rule_outputs = []
        for k in range(self.num_rules):
            vec = tails_vec.clone()
            for t in range(self.max_rule_len):
                # compute weighted sum of A_r @ vec over r
                # accumulate as dense vector (num_entity, batch)
                accum = None
                # iterate relations (could be optimized by batching dense matrices)
                att_r = att[k, t]  # (num_relation,)
                for r_idx in range(self.num_relation):
                    w = att_r[r_idx]
                    if w.item() == 0:
                        continue
                    step_res = apply_op(r_idx, vec)  # (num_entity, batch)
                    if accum is None:
                        accum = w * step_res
                    else:
                        accum = accum + w * step_res
                if accum is None:
                    # no relations? keep vec
                    accum = vec
                vec = accum
            rule_outputs.append(vec)

        # Sum over rules
        total = torch.stack(rule_outputs, dim=0).sum(dim=0)  # (num_entity, batch)

        # transpose back to (batch, num_entity)
        scores = total.t()
        # normalize to probabilities
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-12)

        # produce loss and top-k indicator similar to DRUM flavor: negative log-likelihood of true head
        heads_onehot = torch.zeros(batch, self.num_entity, device=device)
        heads_onehot.scatter_(1, heads.view(-1,1).to(device), 1.0)
        loss = -torch.sum(heads_onehot * torch.log(torch.clamp(scores, min=1e-12)), dim=1)

        # in_top using top-k
        top_k = min(10, self.num_entity)
        topk_vals, topk_idx = torch.topk(scores, top_k, dim=1)
        heads_exp = heads.view(-1,1).to(device)
        in_top = (topk_idx == heads_exp).any(dim=1)

        return loss, in_top, scores

    def get_rule_attention(self):
        return F.softmax(self.rule_logits, dim=2).detach().cpu().numpy()
