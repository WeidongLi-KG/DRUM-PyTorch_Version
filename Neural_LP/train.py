"""
Neural_LP training driver (PyTorch)
"""
import argparse
import torch
import numpy as np
from model import NeuralLP
from data import Data
from scipy import sparse


def convert_matrix_db_to_torch(matrix_db, num_relation, num_entity, device):
    from scipy import sparse as sp
    mdb = []
    for r in range(num_relation):
        entry = matrix_db.get(r, ([[0,0]], [0.], (num_entity, num_entity)))
        coords = entry[0]
        vals = entry[1]
        if len(coords) <= 1:
            indices = torch.LongTensor([[], []]).to(device)
            values = torch.FloatTensor([]).to(device)
            sp_t = torch.sparse_coo_tensor(indices, values, (num_entity, num_entity)).coalesce()
            mdb.append(sp_t)
            continue
        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]
        indices = torch.LongTensor([rows, cols]).to(device)
        values = torch.FloatTensor(vals).to(device)
        sp_t = torch.sparse_coo_tensor(indices, values, (num_entity, num_entity)).coalesce()
        mdb.append(sp_t)
    return mdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data = Data(args.datadir)
    data.reset(args.batch_size)

    model = NeuralLP(num_relation=data.num_relation, num_entity=data.num_entity, max_rule_len=3, num_rules=32, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # preconvert DB to torch sparse tensors on device
    mdb = convert_matrix_db_to_torch(data.matrix_db, data.num_relation, data.num_entity, device)

    for epoch in range(args.max_epoch):
        # simple epoch loop
        num_batches = int(np.ceil(len(data.train) / args.batch_size))
        epoch_loss = 0.0
        for b in range(num_batches):
            (qq, hh, tt), mdb_local = data.next_train()
            heads = torch.LongTensor(hh).to(device)
            tails = torch.LongTensor(tt).to(device)
            loss_batch, in_top, scores = model(None, heads, tails, mdb)
            loss = loss_batch.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.max_epoch} loss={epoch_loss/num_batches:.4f}")

    # write simple predictions file
    out_path = 'neural_lp_predictions.txt'
    with open(out_path, 'w') as f:
        for (qq, hh, tt), mdb_local in data.next_test():
            heads_t = torch.LongTensor(hh).to(device)
            tails_t = torch.LongTensor(tt).to(device)
            _, _, scores = model(None, heads_t, tails_t, mdb)
            scores = scores.cpu().detach().numpy()
            for i in range(scores.shape[0]):
                q = str(qq[i])
                h = data.number_to_entity[hh[i]]
                t = data.number_to_entity[tt[i]]
                preds_sorted = np.argsort(-scores[i])
                preds_entities = [data.number_to_entity[int(j)] for j in preds_sorted[:10]]
                line = ','.join([q, h, t] + preds_entities + [h]) + '\n'
                f.write(line)
    print('Wrote predictions to', out_path)

if __name__ == '__main__':
    main()
