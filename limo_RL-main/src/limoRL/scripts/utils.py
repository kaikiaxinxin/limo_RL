# 导入 matplotlib.pyplot 模块，用于绘制和保存图表
import matplotlib.pyplot as plt
# 导入 numpy 模块，用于高效的数值计算
import numpy as np

def plotLearning(scores, filename, x=None, window=5):
    """
    绘制学习曲线，展示分数的移动平均。

    :param scores: 包含一系列分数的列表或数组
    :param filename: 保存图表的文件路径和文件名
    :param x: 可选参数，x 轴的数据点，默认为 None
    :param window: 计算移动平均时的窗口大小，默认为 5
    """
    # 获取分数列表的长度
    N = len(scores)
    # 初始化一个长度为 N 的数组，用于存储移动平均分数
    running_avg = np.empty(N)
    # 遍历分数列表，计算每个位置的移动平均分数
    for t in range(N):
        # 计算从 t - window 到 t 的分数的平均值，确保起始索引不小于 0
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    # 如果 x 未提供，则使用 0 到 N-1 的整数作为 x 轴数据
    if x is None:
        x = [i for i in range(N)]
    # 设置 y 轴标签为 'Score'
    plt.ylabel('Score')
    # 设置 x 轴标签为 'Step'
    plt.xlabel('Step')
    # 绘制 x 轴数据和移动平均分数的曲线
    plt.plot(x, running_avg)
    # 将绘制的图表保存到指定的文件
    plt.savefig(filename)

def plot_learning_curve(x, scores, figure_file):
    """
    绘制学习曲线，展示前 100 个分数的移动平均。

    :param x: x 轴的数据点
    :param scores: 包含一系列分数的列表或数组
    :param figure_file: 保存图表的文件路径和文件名
    """
    # 初始化一个长度为分数列表长度的数组，用于存储移动平均分数
    running_avg = np.zeros(len(scores))
    # 遍历移动平均分数数组，计算每个位置的移动平均分数
    for i in range(len(running_avg)):
        # 计算从 i - 100 到 i 的分数的平均值，确保起始索引不小于 0
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    # 绘制 x 轴数据和移动平均分数的曲线
    plt.plot(x, running_avg)
    # 设置图表标题为 'Running average of previous 100 scores'
    plt.title('Running average of previous 100 scores')
    # 将绘制的图表保存到指定的文件
    plt.savefig(figure_file)
