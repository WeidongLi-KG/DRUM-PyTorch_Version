#!/usr/bin/env bash
# scripts/run_demo.sh
python src/main.py --datadir=datasets/family --exps_dir=exps --exp_name=demo --batch_size=8 --max_epoch=2
bash eval/collect_all_facts.sh datasets/family
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds exps/demo/test_predictions.txt --truths datasets/family/truths.pckl --top_k 10
