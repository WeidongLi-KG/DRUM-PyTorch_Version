# README.md

DRUM PyTorch Port (one-to-one structure with original DRUM)

This repository mirrors the original DRUM project structure but implements the model in PyTorch
with GPU-friendly sparse handling and dense fallback.

Quick start:
1. Clone repo
   git clone https://github.com/WeidongLi-KG/DRUM-PyTorch_Version.git
   cd DRUM-PyTorch_Version/src

2. Install dependencies (use appropriate torch for your CUDA):
   pip install -r requirements.txt

3. Run demo (from repo root):
   python src/main.py --datadir=datasets/family --exps_dir=exps --exp_name=demo --batch_size=8 --max_epoch=2

4. Eval:
   bash eval/collect_all_facts.sh datasets/family
   python eval/get_truths.py datasets/family
   python eval/evaluate.py --preds exps/demo/test_predictions.txt --truths datasets/family/truths.pckl --top_k 10

Notes:
- The code mirrors the original repository layout (src/, eval/, datasets/).
- Sparse matrices are converted once per DB and cached on device; conversion policy controlled by CLI args.
