import json
import matplotlib.pyplot as plt


def tendency_pc(log: dict, path: str, show_conf=False, higher_is_better=False):
    epochs = [dic['epoch'] for dic in log]
    train_p = [dic['train_p_metric'] for dic in log]
    valid_p = [dic['validate_p_metric'] for dic in log]
    test_p = [dic['test_p_metric'] for dic in log]
    ps = zip(valid_p, test_p)
    ps = sorted(ps, key=lambda x: x[0], reverse=higher_is_better)
    print('{}: {:.4f}'.format(path, ps[0][1]))

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_p, color='red', linestyle='--')
    ax1.plot(epochs, test_p, color='red')
    ax1.set_ylim(min(train_p) - 0.5, max(train_p) + 0.5)

    if show_conf:
        train_c = [dic['train_c_metric'] for dic in log]
        test_c = [dic['test_c_metric'] for dic in log]
        ax2 = ax1.twinx()
        ax2.plot(epochs, train_c, color='green', linestyle='--')
        ax2.plot(epochs, test_c, color='green')
        ax2.set_ylim(min(train_c) - 0.1, max(train_c) + 0.1)

    plt.savefig(path)
    plt.close(fig)


tuples = [
    ('Lipop', 'Lipop'),
    ('Lipop', 'Lipop-LR5'),
    # ('Lipop', 'Lipop-test'),
    ('QM9', 'QM9-Xconf'),
    ('QM9', 'QM9-rdkit'),
    # ('QM9', 'QM9', True),
    ('QM9', 'QM9-lrd', True),
    # ('QM9', 'QM9-10000-Xconf-naive'),
    # ('QM9', 'QM9-5000-naive'),
    # ('QM9', 'QM9-5000'),
    # ('QM9', 'QM9-5000-lambda1e-2'),
]

for d, f, *t in tuples:
    json_path = f'{d}/{f}.json'
    graph_path = f'{d}/{f}.png'
    with open(json_path) as fp:
        log = json.load(fp)
    tendency_pc(log, graph_path, t)
