# Neural_LP/README.md

This folder contains a PyTorch port of the Neural LP model, organized to mirror the original Neural-LP project structure.

Files
- model.py: PyTorch implementation of a differentiable rule learning model inspired by Neural LP.
- data.py: Data loader that reads dataset files in the same format used in the DRUM port (relations.txt, entities.txt, train.txt, valid.txt, test.txt, facts.txt).
- train.py: Training and evaluation driver with CLI options.
- requirements.txt: minimal requirements for this component.

Notes
- This is an initial one-to-one style port aiming for API parity with the rest of this repo. It uses the same matrix_db representation as the DRUM port and supports dense/sparse operator matrices.
- The implementation focuses on correctness and compatibility; further performance tuning and feature parity can be added on request.
