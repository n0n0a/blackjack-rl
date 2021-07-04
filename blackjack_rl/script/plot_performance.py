from matplotlib import pyplot as plt
import os, pickle

# data dir
_base = os.path.dirname(os.path.abspath(__file__))#実行中のファイル(このファイル)の絶対パス
data_dir = os.path.join(_base, "../../data")#実行中のファイルからの相対パスでdataの出力先を決定
lspi_path = os.path.join(data_dir, "lspi_rewards.txt")
monte_path = os.path.join(data_dir, "monte_rewards.txt")
qlearning_path = os.path.join(data_dir, "qlearning_rewards.txt")
plot_path = os.path.join(data_dir, "performance.jpg")


def plot_performance():
    paths = [lspi_path, monte_path, qlearning_path]
    names = ["lspi", "monte", "qlearning"]
    fig = plt.figure()
    for idx, path in enumerate(paths):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                x = list(range(len(data)))
                print(data)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(x, data, label=names[idx])
                ax.set_xlabel('epoch')
                ax.set_ylabel('reward')
    plt.legend()
    plt.show()
    fig.savefig(plot_path)


if __name__ == '__main__':
    plot_performance()