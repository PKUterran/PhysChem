import json
import numpy as np

from train.utils.seed import DEFAULT_SEEDS


def eval_pc(log: dict, higher_is_better=False) -> float:
    valid_p = [dic['validate_p_metric'] for dic in log]
    test_p = [dic['test_p_metric'] for dic in log]
    ps = zip(valid_p, test_p)
    ps = sorted(ps, key=lambda x: x[0], reverse=higher_is_better)
    return ps[0][1]


tuples = [
    # ('Lipop', 'Lipop', False),
    # ('Lipop', 'Lipop-Xconf', False),
    # ('TOX21', 'TOX21', True),
    # ('TOX21', 'TOX21-Xconf', True),
    # ('ESOL', 'ESOL', False),
    # ('ESOL', 'ESOL-Xconf', False),
    # ('FreeSolv', 'FreeSolv', False),
    # ('FreeSolv', 'FreeSolv-Xconf', False),
    ('QM7', 'QM7', False),
    ('QM7', 'QM7-rdkit', False),
    ('QM7', 'QM7-Xconf', False),
]

for d, f, h in tuples:
    results = []
    # for seed in DEFAULT_SEEDS[:2] + DEFAULT_SEEDS[3:]:
    for seed in DEFAULT_SEEDS:
        try:
            json_path = f'{d}/{f}@{seed}.json'
            with open(json_path) as fp:
                log = json.load(fp)
            result = eval_pc(log, h)
            results.append(result)
        except FileNotFoundError:
            pass
    if len(results) == 0:
        continue
    avg = np.mean(results)
    std = np.std(results)
    print('{}: {:.4f} +- {:.4f}'.format(f, avg, std))
    # print('{}: {:.4f} +- {:.4f}'.format(f, avg * 222, std * 222))
