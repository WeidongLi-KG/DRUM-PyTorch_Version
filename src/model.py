# src/model.py
# PyTorch port of core DRUM model (with support for cached sparse/dense mdb entries)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse

def is_torch_sparse(t):
    return isinstance(t, torch.Tensor) and t.is_sparse

def is_torch_dense(t):
    return isinstance(t, torch.Tensor) and not t.is_sparse

def convert_scipy_to_torch_sparse(sp_mat, device=None, dtype=torch.float32):
    """Convert scipy.sparse matrix to torch.sparse_coo_tensor on device."""
    if not sparse.isspmatrix_coo(sp_mat):
        sp_mat = sp_mat.tocoo()
    rows = torch.LongTensor(sp_mat.row)
    cols = torch.LongTensor(sp_mat.col)
    if device is not None:
        rows = rows.to(device)
        cols = cols.to(device)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.FloatTensor(sp_mat.data)
    if device is not None:
        values = values.to(device)
    shape = sp_mat.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), dtype=dtype).coalesce()

class Learner(nn.Module):
    """
    PyTorch port of TF Learner from DRUM.
    Provides update/predict/get_predictions interfaces compatible with the original experiment driver.
    """
    def __init__(self, option, device=torch.device("cpu")):
        super(Learner, self).__init__()
        self.device = device

        # options
        self.seed = option['seed'] if isinstance(option, dict) else option.seed
        self.num_step = option['num_step'] if isinstance(option, dict) else option.num_step
        self.rank = option['rank'] if isinstance(option, dict) else option.rank
        self.num_layer = option['num_layer'] if isinstance(option, dict) else option.num_layer
        self.rnn_state_size = option['rnn_state_size'] if isinstance(option, dict) else option.rnn_state_size

        self.norm = not (option.get('no_norm') if isinstance(option, dict) else option.no_norm)
        self.thr = option['thr'] if isinstance(option, dict) else option.thr
        self.dropout = option['dropout'] if isinstance(option, dict) else option.dropout
        self.learning_rate = option['learning_rate'] if isinstance(option, dict) else option.learning_rate
        self.accuracy = option['accuracy'] if isinstance(option, dict) else option.accuracy
        self.top_k = option['top_k'] if isinstance(option, dict) else option.top_k

        self.num_entity = option['num_entity'] if isinstance(option, dict) else option.num_entity
        self.num_operator = option['num_operator'] if isinstance(option, dict) else option.num_operator
        self.query_is_language = option['query_is_language'] if isinstance(option, dict) else option.query_is_language

        if not self.query_is_language:
            self.num_query = option['num_query'] if isinstance(option, dict) else option.num_query
            self.query_embed_size = option['query_embed_size'] if isinstance(option, dict) else option.query_embed_size
        else:
            self.vocab_embed_size = option['vocab_embed_size'] if isinstance(option, dict) else option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option['num_vocab'] if isinstance(option, dict) else option.num_vocab
            self.num_word = option['num_word'] if isinstance(option, dict) else option.num_word

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Embeddings
        if not self.query_is_language:
            self.query_embedding = nn.Parameter(torch.tensor(self._random_uniform_unit(self.num_query + 1, self.query_embed_size), dtype=torch.float32))
        else:
            self.vocab_embedding = nn.Parameter(torch.tensor(self._random_uniform_unit(self.num_vocab + 1, self.vocab_embed_size), dtype=torch.float32))

        # Bi-LSTM per rank
        self.lstm_list = nn.ModuleList()
        for _ in range(self.rank):
            lstm = nn.LSTM(input_size=self.query_embed_size,
                           hidden_size=self.rnn_state_size,
                           num_layers=self.num_layer,
                           batch_first=True,
                           bidirectional=True)
            self.lstm_list.append(lstm)

        self.W_0 = nn.Linear(2 * self.rnn_state_size, self.num_operator + 1)

        self.optimizer = None

        self.to(self.device)

    def _random_uniform_unit(self, r, c):
        bound = 6.0 / math.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c)).astype(np.float32)
        norms = np.linalg.norm(init_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        init_matrix = init_matrix / norms
        return init_matrix

    def build_one_hot(self, indices, depth):
        batch = indices.size(0)
        one = torch.zeros(batch, depth, device=indices.device, dtype=torch.float32)
        one.scatter_(1, indices.view(-1, 1), 1.0)
        return one

    def forward(self, queries, heads, tails, mdb_torch, training=True):
        batch = tails.size(0)

        # embeddings
        if not self.query_is_language:
            q_embed = F.embedding(queries, self.query_embedding.to(queries.device))
            rnn_inputs = q_embed[:, :self.num_step, :]
        else:
            v = F.embedding(queries, self.vocab_embedding.to(queries.device))
            rnn_inputs = v.mean(dim=2)

        seq_len = max(1, self.num_step - 1)
        rnn_seq = rnn_inputs[:, :seq_len, :]

        # run Bi-LSTM per rank
        rnn_outputs_list = []
        for i in range(self.rank):
            out, _ = self.lstm_list[i](rnn_seq)
            rnn_outputs_list.append(out)

        # attention per time step
        attention_operators_list = []
        for i_rank in range(self.rank):
            att_per_t = []
            rnn_out = rnn_outputs_list[i_rank]
            for t in range(rnn_out.size(1)):
                logits = self.W_0(rnn_out[:, t, :])
                att = F.softmax(logits, dim=1)
                att_per_t.append(att)
            attention_operators_list.append(att_per_t)

        # initial memory
        memories_list = []
        tails_onehot = self.build_one_hot(tails.to(self.device), self.num_entity)
        for _ in range(self.rank):
            memories_list.append(tails_onehot.unsqueeze(1))  # (batch,1,num_entity)

        predictions = torch.zeros(batch, self.num_entity, device=self.device, dtype=torch.float32)

        for i_rank in range(self.rank):
            for t in range(self.num_step):
                memory_read = memories_list[i_rank][:, -1, :]
                if t < self.num_step - 1:
                    database_results = []
                    mem_t = memory_read.t()
                    for r in range(self.num_operator // 2):
                        entry = mdb_torch[r]
                        if isinstance(entry, tuple) and len(entry) == 2:
                            op_fw, op_tr = entry
                        else:
                            op_fw = entry
                            op_tr = None

                        if is_torch_sparse(op_fw):
                            prod = torch.sparse.mm(op_fw, mem_t)
                            prod_tb = prod.t()
                        elif is_torch_dense(op_fw):
                            prod = torch.matmul(op_fw, mem_t)
                            prod_tb = prod.t()
                        else:
                            prod = convert_scipy_to_torch_sparse(entry if not isinstance(entry, tuple) else entry[0], device=self.device)
                            prod = torch.sparse.mm(prod, mem_t)
                            prod_tb = prod.t()

                        att_fw = attention_operators_list[i_rank][t][:, r].unsqueeze(1).to(prod_tb.device)
                        database_results.append(prod_tb * att_fw)

                        if op_tr is not None:
                            if is_torch_sparse(op_tr):
                                prod2 = torch.sparse.mm(op_tr, mem_t)
                                prod2_tb = prod2.t()
                            elif is_torch_dense(op_tr):
                                prod2 = torch.matmul(op_tr, mem_t)
                                prod2_tb = prod2.t()
                            else:
                                if is_torch_dense(op_fw):
                                    prod2 = torch.matmul(op_fw.t(), mem_t)
                                    prod2_tb = prod2.t()
                                else:
                                    dense = op_fw.to_dense() if is_torch_sparse(op_fw) else op_fw
                                    prod2 = torch.matmul(dense.t(), mem_t)
                                    prod2_tb = prod2.t()
                            att_bw = attention_operators_list[i_rank][t][:, r + self.num_operator // 2].unsqueeze(1).to(prod2_tb.device)
                            database_results.append(prod2_tb * att_bw)

                    last_att = attention_operators_list[i_rank][t][:, -1].unsqueeze(1).to(memory_read.device)
                    database_results.append(memory_read * last_att)

                    added_database_results = torch.stack(database_results, dim=0).sum(dim=0)
                    if self.norm:
                        sums = torch.sum(added_database_results, dim=1, keepdim=True)
                        denom = torch.clamp(sums, min=self.thr)
                        added_database_results = added_database_results / denom

                    if self.dropout > 0.0 and training:
                        added_database_results = F.dropout(added_database_results, p=self.dropout, training=training)

                    memories_list[i_rank] = torch.cat([memories_list[i_rank], added_database_results.unsqueeze(1)], dim=1)
                else:
                    predictions = predictions + memory_read

        predictions_clamped = torch.clamp(predictions, min=self.thr)
        heads_onehot = self.build_one_hot(heads.to(self.device), self.num_entity)
        final_loss = - torch.sum(heads_onehot * torch.log(predictions_clamped), dim=1)

        topk_vals, topk_idx = torch.topk(predictions, self.top_k, dim=1)
        heads_exp = heads.view(-1, 1).to(self.device)
        in_top = (topk_idx == heads_exp).any(dim=1)

        return final_loss, in_top, predictions

    def set_optimizer(self, optim):
        self.optimizer = optim

    def update(self, queries, heads, tails, mdb_torch):
        self.train()
        if self.optimizer is None:
            self.set_optimizer(torch.optim.Adam(self.parameters(), lr=self.learning_rate if hasattr(self, 'learning_rate') else 1e-3))
        self.optimizer.zero_grad()
        loss_batch, in_top, preds = self.forward(queries, heads, tails, mdb_torch, training=True)
        loss = loss_batch.mean()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 5.0)
        self.optimizer.step()
        return loss_batch.detach().cpu().numpy(), in_top.detach().cpu().numpy()
