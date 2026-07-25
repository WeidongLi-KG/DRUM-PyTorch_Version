#!/usr/bin/env python3
# eval/get_truths.py
import sys
import os
import pickle
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python get_truths.py <folder>")
    sys.exit(1)

folder_name = sys.argv[1]
all_file = os.path.join(folder_name, "all.txt")

facts = []
with open(all_file, "r", encoding="utf-8") as f:
    for line in f:
        l = line.strip().split("\t")
        if len(l) == 0:
            continue
        assert(len(l) == 3)
        facts.append(l)
num_fact = len(facts)
print("Number of all facts %d" % num_fact)

query_head = defaultdict(list)
query_tail = defaultdict(list)
for h, r, t in facts:
    query_head[(r, h)].append(t)
    query_tail[(r, t)].append(h)

to_dump = {"query_head": query_head, "query_tail": query_tail}
truths_file = os.path.join(folder_name, "truths.pckl")
with open(truths_file, "wb") as out_f:
    pickle.dump(to_dump, out_f, protocol=pickle.HIGHEST_PROTOCOL)

print("Gather truths done. Wrote:", truths_file)
