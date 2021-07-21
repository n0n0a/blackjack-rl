from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import os, pickle
import numpy as np
from blackjack_rl.agent.lspi import LSPIAgent

# data dir
_base = os.path.dirname(os.path.abspath(__file__))#実行中のファイル(このファイル)の絶対パス
data_dir = os.path.join(_base, "../../data")#実行中のファイルからの相対パスでdataの出力先を決定
lspi_path = os.path.join(data_dir, "lspi_rewards.pkl")
monte_path = os.path.join(data_dir, "monte_rewards.pkl")
qlearning_path = os.path.join(data_dir, "qlearning_rewards.pkl")
plot_path = os.path.join(data_dir, "performance.jpg")
qplot_path = os.path.join(data_dir, "Qperformance.jpg")

lspi_weights_pass = os.path.join(data_dir, "lspi_weights.pkl")


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
    fig.legend()
    fig.savefig(plot_path)

def plot_Q():
    paths = [lspi_weights_pass]
    names = ["lspi"]
    fig2, ax2 = plt.subplots(1,2, figsize=(9,4))
    _max = -1
    _min = 1
    for idx, path in enumerate(paths):
        if os.path.exists(path):
            with open(path, "rb") as f:
                weights = pickle.load(f)
                print("weight_one : "+str(len(weights[0])))
                heatmap = np.zeros([9,10])
                for weight in weights:
                    for x in range(10):
                        for y in range(9):
                            stt = (x, y, False)
                            diff = weight[LSPIAgent._translate_weight_idx(state=stt, action=True)] - weight[LSPIAgent._translate_weight_idx(state=stt, action=False)]
                            heatmap[y,x] = diff
                            _max = diff if _max < diff else _max
                            _min = diff if _min > diff else _min
                
                mappable1 = ax2[0].pcolor(heatmap, cmap='seismic')
                ax2[0].set_xlabel("Upcard of Dealerr")
                ax2[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(-1))
                ax2[0].set_ylabel("Sum of Player's hand")
                ax2[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(-12))
                ax2[0].set_title('Hard')
                # ax.set_position([枠の左辺, 枠の下辺, 枠の横幅, 枠の高さ])単位は割合
                # ax2[0].set_position([0.1, 0.56, 0.8, 0.35])

                heatmap = np.zeros([9,10])
                for weight in weights:
                    for x in range(10):
                        for y in range(9):
                            stt = (x, y, True)
                            diff = weight[LSPIAgent._translate_weight_idx(state=stt, action=True)] - weight[LSPIAgent._translate_weight_idx(state=stt, action=False)]
                            heatmap[y,x] = diff
                            _max = diff if _max < diff else _max
                            _min = diff if _min > diff else _min
                
                mappable2 = ax2[1].pcolor(heatmap, cmap='seismic')
                ax2[1].set_xlabel("Upcard of Dealer")
                ax2[1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(-1))
                ax2[1].set_ylabel("Sum of Player's hand")
                ax2[1].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(-12))
                ax2[1].set_title('Soft')
                # ax2[1].set_position([0.1, 0.1, 0.8, 0.35])

    #カラーバーの設定
    axpos = ax2[1].get_position()
    cbar_ax = fig2.add_axes([0.87, axpos.y0, 0.02, axpos.height])
    norm = colors.Normalize(vmin=-max(_min,_max),vmax=max(_min,_max))
    mappable = ScalarMappable(cmap='seismic',norm=norm)
    mappable._A = []
    fig2.colorbar(mappable, cax=cbar_ax, label = "Diff b/w [hit] & [stand]")

    #余白の調整
    plt.subplots_adjust(right=0.8)
    plt.subplots_adjust(wspace=0.3)

    fig2.savefig(qplot_path)

                


if __name__ == '__main__':
    plot_Q()
    plot_performance()

    plt.show()