import json
import matplotlib.pyplot as plt


def tendency_pc(log: dict, path: str):
    epochs = [dic['epoch'] for dic in log]
    train_p = [dic['train_p_metric'] for dic in log]
    train_c = [dic['train_c_metric'] for dic in log]
    test_p = [dic['test_p_metric'] for dic in log]
    test_c = [dic['test_c_metric'] for dic in log]

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_p, color='red', linestyle='--')
    ax1.plot(epochs, test_p, color='red')
    ax1.set_ylim(3, 11)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_c, color='green', linestyle='--')
    ax2.plot(epochs, test_c, color='green')
    ax2.set_ylim(0.7, 1.8)
    plt.savefig(path)
    plt.close(fig)


tuples = [
    ('QM9', 'QM9'),
    # ('QM9', 'QM9-5000-naive'),
    # ('QM9', 'QM9-5000'),
    # ('QM9', 'QM9-5000-lambda1e-2'),
]

for d, f in tuples:
    json_path = f'{d}/{f}.json'
    graph_path = f'{d}/{f}.png'
    with open(json_path) as fp:
        log = json.load(fp)
    tendency_pc(log, graph_path)
