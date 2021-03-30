import json
import numpy as np

from train.utils.seed import DEFAULT_SEEDS


def eval_p(log: dict, higher_is_better=False) -> float:
    valid_p = [dic['validate_p_metric'] for dic in log]
    test_p = [dic['test_p_metric'] for dic in log]
    ps = zip(valid_p, test_p)
    ps = sorted(ps, key=lambda x: x[0], reverse=higher_is_better)
    return ps[0][1]


def eval_c(log: dict, higher_is_better=False) -> float:
    valid_c = [dic['validate_c_metric'] for dic in log]
    test_c = [dic['test_c_metric'] for dic in log]
    cs = zip(valid_c, test_c)
    cs = sorted(cs, key=lambda x: x[0], reverse=higher_is_better)
    return cs[0][1]


tuples = [
    ('Lipop', 'Lipop', False),
    ('Lipop', 'Lipop-RGT', False),
    ('Lipop', 'Lipop-Xconf', False),
    # ('TOX21', 'TOX21', True),
    # ('TOX21', 'TOX21-Xconf', True),
    ('ESOL', 'ESOL', False),
    ('ESOL', 'ESOL-RGT', False),
    ('ESOL', 'ESOL-Xconf', False),
    ('FreeSolv', 'FreeSolv', False),
    ('FreeSolv', 'FreeSolv-RGT', False),
    ('FreeSolv', 'FreeSolv-Xconf', False),
    ('QM7', 'HamEng', False),
    ('QM7', 'QM7', False),
    ('QM7', 'QM7-rdkit', False),
    ('QM7', 'QM7-Xconf', False),
    ('QM7', 'QM7-Oconf', False),
    ('QM7', 'QM7-real', False),
    ('QM7', 'CVGAE-QM7-rdkit', False),
    ('QM7', 'CVGAE-QM7-Xconf', False),
]

for d, f, h in tuples:
    p_results = []
    c_results = []
    for seed in DEFAULT_SEEDS:
        try:
            json_path = f'{d}/{f}@{seed}.json'
            with open(json_path) as fp:
                log = json.load(fp)
            try:
                p_result = eval_p(log, h)
                p_results.append(p_result)
            except KeyError:
                pass
            try:
                c_result = eval_c(log, h)
                c_results.append(c_result)
            except KeyError:
                pass
        except FileNotFoundError:
            pass
    if len(p_results):
        avg = np.mean(p_results)
        std = np.std(p_results)
        if d == 'QM7':
            print('{}: {:.4f} +- {:.4f}'.format(f, avg * 223.918853, std * 223.918853))
        else:
            print('{}: {:.4f} +- {:.4f}'.format(f, avg, std))
    if len(c_results):
        avg = np.mean(c_results)
        std = np.std(c_results)
        print('{}: {:.4f} +- {:.4f} (conf)'.format(f, avg, std))
