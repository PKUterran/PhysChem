import os
import json
import matplotlib.pyplot as plt


def tendency_pc(log: dict, path: str, higher_is_better=False, show_conf=False):
    if 'ps-p1' in path or 'ps-p21' in path or 'ps-p21' in path:
        log = log[:250]
    if 'ps-p11' in path:
        log = log[:200]
    epochs = [dic['epoch'] for dic in log]
    fig, ax1 = plt.subplots()
    if 'CVGAE' in path or 'HamEng' in path:
        pass
    else:
        train_p = [dic['train_p_metric'] for dic in log]
        valid_p = [dic['validate_p_metric'] for dic in log]
        test_p = [dic['test_p_metric'] for dic in log]
        ps = zip(valid_p, test_p)
        ps = sorted(ps, key=lambda x: x[0], reverse=higher_is_better)
        print('{}: {:.4f}'.format(path, ps[0][1]))

        ax1.plot(epochs, train_p, color='red', linestyle='--')
        ax1.plot(epochs, test_p, color='red')
        # ax1.set_ylim(min(train_p) - 0.2, max(train_p) + 0.2)

    if show_conf:
        if ('TOX21' in path or 'sars' in path) and 'RGT' not in path:
            train_c = [dic['train_loss'] for dic in log]
            valid_c = [dic['validate_loss'] for dic in log]
            test_c = [dic['test_loss'] for dic in log]
        else:
            train_c = [dic['train_c_metric'] for dic in log]
            valid_c = [dic['validate_c_metric'] for dic in log]
            test_c = [dic['test_c_metric'] for dic in log]
        ps = zip(valid_c, test_c)
        ps = sorted(ps, key=lambda x: x[0], reverse=False)
        if ('TOX21' in path or 'sars' in path) and 'RGT' not in path:
            print('{}: {:.4f} (loss)'.format(path, ps[0][1]))
        else:
            print('{}: {:.4f} (conf)'.format(path, ps[0][1]))
        ax2 = ax1.twinx()
        ax2.plot(epochs, train_c, color='green', linestyle='--')
        ax2.plot(epochs, test_c, color='green')
        # ax2.set_ylim(min(train_c) - 0.1, max(train_c) + 0.1)

    plt.savefig(path)
    plt.close(fig)


tuples = [
    # ('QM9', 'QM9', False, True),
    # ('QM9', 'QM9-Xconf', False, True),
    # ('QM9', 'QM9-rdkit', False, True),
    # ('QM9', 'QM9-Oconf', False, True),
    # ('QM9', 'QM9-real', False, False),
    # ('QM9', 'QM9-single', False, True),
    # ('QM9', 'CVGAE-QM9-rdkit', False, True),
    # ('QM9', 'CVGAE-QM9-Xconf', False, True),
    # ('QM9', 'HamEng-QM9', False, True),
    # ('QM8', 'QM8@16880611', False, True),
    # ('QM8', 'QM8-rdkit@16880611', False, True),
    # ('QM8', 'QM8-Xconf@16880611', False, True),
    # ('QM8', 'QM8-Oconf@16880611', False, True),
    # ('QM8', 'QM8-real@16880611', False, False),
    # ('QM8', 'QM8-single', False, False),
    # ('QM8', 'CVGAE-QM8-rdkit@16880611', False, True),
    # ('QM8', 'CVGAE-QM8-Xconf@16880611', False, True),
    # ('QM8', 'HamEng@16880611', False, True),
    # ('QM8', 'AttentiveFP-QM8@16880611', False, False),
    # ('QM7', 'HamEng@16880611', False, True),
    # ('QM7', 'QM7@16880611', False, True),
    # ('QM7', 'QM7-rdkit@16880611', False, True),
    # ('QM7', 'QM7-Xconf@16880611', False, True),
    # ('QM7', 'QM7-Oconf@16880611', False, True),
    # ('QM7', 'QM7-real@16880611', False, False),
    # ('QM7', 'QM7-single', False, False),
    # ('QM7', 'CVGAE-QM7-rdkit@16880611', False, False),
    # ('QM7', 'CVGAE-QM7-Xconf@16880611', False, False),
    # ('QM7', 'AttentiveFP-QM7@16880611', False, False),
    #
    # ('Lipop', 'Lipop@16880611', False, False),
    # ('Lipop', 'Lipop-RGT@16880611', False, True),
    # ('Lipop', 'Lipop-Xconf@16880611', False, False),
    #
    # ('ESOL', 'ESOL@16880611', False, False),
    ('ESOL', 'ESOL@16880611-test', False, False),
    # ('ESOL', 'ESOL-RGT@16880611', False, True),
    # ('ESOL', 'ESOL-Xconf@16880611', False, False),
    #
    # ('FreeSolv', 'FreeSolv@16880611', False, False),
    # ('FreeSolv', 'FreeSolv-RGT@16880611', False, True),
    # ('FreeSolv', 'FreeSolv-Xconf@16880611', False, False),
    #
    # ('TOX21', 'TOX21@16880611', True, True),
    # ('TOX21', 'TOX21-RGT@16880611', True, True),
    # ('TOX21', 'TOX21-Xconf@16880611', True, True),
    #
    # ('sars', 'sars@16880611', True, True),
    # ('sars', 'sars-RGT@16880611', True, True),
    # ('sars', 'sars-Xconf@16880611', True, True),

    # ('QM9', 'ps-l0', False, True),  # lambda = 1
    # ('QM9', 'ps-l1', False, True),  # lambda = 10
    # ('QM9', 'ps-l3', False, True),  # lambda = 1000
    #
    # ('QM9', 'ps-p11', False, True),  # iteration=2, tau=1/8
    # ('QM9', 'ps-p12', False, True),  # iteration=2, tau=1/4
    # ('QM9', 'ps-p13', False, True),  # iteration=2, tau=1/2
    # ('QM9', 'ps-p21', False, True),  # iteration=4, tau=1/8
    # ('QM9', 'QM9', False, True),     # iteration=4, tau=1/4
    # ('QM9', 'ps-p23', False, True),  # iteration=4, tau=1/2
    # ('QM9', 'ps-p31', False, True),  # iteration=8, tau=1/8
    # ('QM9', 'ps-p32', False, True),  # iteration=8, tau=1/4
    # ('QM9', 'ps-p33', False, True),  # iteration=8, tau=1/2
]

for d, f, h, t in tuples:
    if not os.path.exists(d):
        os.mkdir(d)
    json_path = f'{d}/{f}.json'
    graph_path = f'{d}/{f}.png'
    try:
        with open(json_path) as fp:
            log = json.load(fp)
    except FileNotFoundError:
        continue
    tendency_pc(log, graph_path, h, t)
