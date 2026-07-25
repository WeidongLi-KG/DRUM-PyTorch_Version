"""
Neural_LP data loader and utilities.
This loader mirrors the matrix_db representation used in the DRUM port so the experiment driver can reuse similar code.
"""
import os
import numpy as np
from collections import Counter

class Data(object):
    def __init__(self, folder, seed=123):
        np.random.seed(seed)
        self.folder = folder
        self.relation_file = os.path.join(folder, 'relations.txt')
        self.entity_file = os.path.join(folder, 'entities.txt')
        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v:k for k,v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_entity = len(self.entity_to_number)

        self.train_file = os.path.join(folder, 'train.txt')
        self.valid_file = os.path.join(folder, 'valid.txt')
        self.test_file = os.path.join(folder, 'test.txt')
        self.facts_file = os.path.join(folder, 'facts.txt')

        self.train, self.num_train = self._parse_triplets(self.train_file)
        self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        self.test, self.num_test = self._parse_triplets(self.test_file)
        if os.path.isfile(self.facts_file):
            self.facts, self.num_fact = self._parse_triplets(self.facts_file)
        else:
            self.facts, self.num_fact = [], 0

        self.matrix_db = self._db_to_matrix_db(self.facts)

    def _numerical_encode(self):
        rel2num = {}
        with open(self.relation_file) as f:
            for line in f:
                r = line.strip()
                if r:
                    rel2num[r] = len(rel2num)
        ent2num = {}
        with open(self.entity_file) as f:
            for line in f:
                e = line.strip()
                if e:
                    ent2num[e] = len(ent2num)
        return rel2num, ent2num

    def _parse_triplets(self, file):
        out = []
        if not os.path.isfile(file):
            return out, 0
        with open(file) as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) != 3:
                    continue
                h, r, t = l[0], l[1], l[2]
                out.append((self.relation_to_number[r], self.entity_to_number[h], self.entity_to_number[t]))
        return out, len(out)

    def _db_to_matrix_db(self, db):
        matrix_db = {r: ([[0,0]], [0.], [self.num_entity, self.num_entity]) for r in range(self.num_relation)}
        for fact in db:
            rel, head, tail = fact
            matrix_db[rel][0].append([head, tail])
            matrix_db[rel][1].append(1.0)
        return matrix_db

    # Simple batch iterator for training/testing
    def reset(self, batch_size=32):
        self.batch_size = batch_size
        self.train_pos = 0

    def next_train(self):
        start = self.train_pos
        end = min(start + self.batch_size, len(self.train))
        batch = self.train[start:end]
        self.train_pos = 0 if end >= len(self.train) else end
        queries, heads, tails = zip(*batch)
        return (list(queries), list(heads), list(tails)), self.matrix_db

    def next_test(self):
        # simple sequential test iterator
        for i in range(0, len(self.test), self.batch_size):
            batch = self.test[i:i+self.batch_size]
            queries, heads, tails = zip(*batch)
            yield (list(queries), list(heads), list(tails)), self.matrix_db
