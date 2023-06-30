# PhysChem

<a href="https://bohrium.dp.tech/notebook/879b6a3a9d394448913ca1ce7de257ec" target="_blank"><img src="https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg" alt="Open In Bohrium"/></a>

**SEE `PhysChem.ipynb` FOR DETAILED INSTRUCTION**

Code for paper [Deep Molecular Representation Learning via Fusing Physical and Chemical Information](https://openreview.net/forum?id=Uxi7X1EqywV) in NeurIPS 2021.

Ways to reproduce:

1. If you want to work on QM9 dataset, please first download datasets of QM9 from [Dataset Collection](http://moleculenet.ai/datasets-1) to `data/QM9`.
2. Configure '--seed'/'--pos' and hyperparameters in `special_config` (with defaulf settings in `train/config.py`), then run the corresponding `xxx.py` or `xxx.slurm` in `scripts` directory

Note that there is a '--pos' option in configuration determining which variant is using:

1. `ConfType.NONE`: ChemNet (*s.a.*)
2. `ConfType.RDKIT`: ChemNet (*rdkit conf.*)
3. `ConfType.NEWTON`: PhysChem (for tasks in Table.2/3)
4. `ConfType.ONLY`: PhysNet (*s.a.*)
5. `ConfType.NEWTON_RGT`: PhysChem (for tasks in Table.4)
6. `ConfType.REAL`: ChemNet (*real conf.*)
7. `ConfType.SINGLE_CHANNEL`: ChemNet (*m.t.*)
