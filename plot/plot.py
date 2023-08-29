# -*- coding: utf-8 -*-
"""
Created on 06 Apr 4:05 PM

@Author: kingsley leung
@Email: kingsleyl0107@gmail.com

_description_:
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import transbigdata as tbd
import matplotlib.colors as colors

# from config.config import *
# set font chinese display
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_hist(df, field_name):
    plt.figure(figsize=(16, 9))
    # 计算均值和标准差
    mean = df[field_name].mean()
    std = df[field_name].std()
    sns.distplot(
        df[field_name], hist=True, kde=True, rug=False, fit=None, label="Distribution"
    )
    # 在图上添加均值和标准差的垂直线条和图例
    plt.axvline(mean, color="red", linestyle="--", label="Mean")
    plt.axvline(mean + std, color="green", linestyle="--", label="Standard Deviation")
    plt.axvline(mean - std, color="green", linestyle="--")
    # 添加图例
    plt.legend()
    # 在右上角添加均值和标准差的文本标注
    plt.text(
        0.8,
        0.7,
        f"Mean={mean:.2f}\nStd={std:.2f}",
        transform=plt.gca().transAxes,
        fontsize=16,
        weight="bold",
    )
    # 设置x轴和y轴的字体大小和加粗
    plt.xticks(fontsize=16, weight="bold")
    plt.yticks(fontsize=16, weight="bold")
    # 设置x轴和y轴标题，并加粗字体
    plt.xlabel("PSI", fontsize=16, weight="bold")
    plt.ylabel("Density(%)", fontsize=16, weight="bold")
    plt.ylim(0, 15)
    # 显示图像
    plt.show()


def plot_box(df, field_name, groupsize):
    # 绘制箱型图
    fig, ax = plt.subplots()

    # 计算每个组的大小
    group_size = int(len(df) / groupsize)

    # 将DataFrame均等分成20组
    groups = np.array_split(df, groupsize)
    for i, group in enumerate(groups):
        ax.boxplot(
            group[field_name],
            positions=[i],
            widths=[0.5],
            showfliers=False,
            notch=False,
            sym="",
        )
    # # 将DataFrame均等分成20组
    # groups = np.array_split(df, 20)
    # for i, group in enumerate(groups):
    #     ax.boxplot(group[field_name], positions=[i], widths=[0.5],
    #                showfliers=False, notch=False, sym='')

    # 设置x轴标签
    ax.set_xticks(np.arange(groupsize))
    # 设置y轴范围和虚线
    ax.axhline(y=0.5, linestyle="--", color="lightgray")
    ax.set_ylim([0.0, 1.2])
    # 显示图像

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_box2(df, field_name, width_scale_factor):
    ratios = []
    # 目标组价格之和
    target_sum = df["price"].sum() / 10

    # 初始化变量
    current_sum = 0
    group = []
    groups = []

    # 遍历价格列，将价格累加，直到达到目标总和
    for index, row in df.sort_values(["price"]).iterrows():
        current_sum += row["price"]

        # 当前价格累加达到目标总和时，将累加的组添加到groups中
        if current_sum >= target_sum:
            group.append(index)
            groups.append(group)
            group = []
            current_sum = 0
        else:
            group.append(index)

    # 如果最后一个组没有达到目标总和，将其添加到groups中
    if group:
        groups.append(group)

    print("分组结果：")
    for i, group in enumerate(groups, start=1):
        ratios.append(len(group) / len(df))
    # ratios.reverse()
    print(ratios)
    group_sizes = [int(ratio * len(df)) for ratio in ratios]
    group_sizes[-1] += len(df) - sum(group_sizes)
    groups = np.array_split(df, np.cumsum(group_sizes)[:-1])

    fig, ax = plt.subplots(figsize=(16, 9))

    # 计算累积宽度
    cumulative_widths = np.cumsum(
        [0] + [ratio * width_scale_factor + ratio * 2 for ratio in ratios]
    )

    # 计算每个箱子的中间位置
    box_positions = (cumulative_widths[:-1] + cumulative_widths[1:]) / 2

    for i, group in enumerate(groups):
        ax.boxplot(
            group[field_name],
            positions=[box_positions[i]],
            widths=[ratios[i] * width_scale_factor],
            showfliers=False,
            notch=False,
            sym="",
        )

    ax.set_xticks(box_positions)
    # ax.set_xticklabels([f"阶层{i + 1}" for i in range(len(groups))])
    ax.set_xticklabels(
        ["3.6", "6.4", "8.8", "11", "13", "15", "16.8", "18.4", "19.5", "20.4"]
    )
    ax.set_xlabel("rank (x10^3)", fontsize=16, weight="bold")
    ax.set_ylabel("PSI", fontsize=16, weight="bold")
    ax.axhline(y=0.5, linestyle="--", color="lightgray")
    ax.set_ylim([-0.1, 1.1])
    # 设置x轴和y轴的字体大小和加粗
    plt.xticks(fontsize=16, weight="bold")
    plt.yticks(fontsize=16, weight="bold")
    plt.show()


def plot_grid_psi(BOUNDARY, psi_grid_df):
    bd = BOUNDARY.bounds.iloc[0]
    fig = plt.figure(1, (8, 8), dpi=300)
    ax = plt.subplot(111)
    plt.sca(ax)
    # 加载地图底图
    tbd.plot_map(plt, bd, zoom=11, style=4)
    # 定义色条位置
    cax = plt.axes([0.05, 0.33, 0.02, 0.3])
    plt.title("PSI Value")
    plt.sca(ax)
    # 自定义cmap颜色列表
    colors_list = [
        (0, "#FFFFE0"),
        (0.01, "#1f78b4"),  # 深蓝色
        (0.25, "#7fcdbb"),  # 浅蓝色
        (0.5, "#d9d9d9"),  # 浅灰色
        (0.75, "#fc8d59"),  # 橙红色
        (1, "#d7191c"),
    ]  # 深红色

    # 创建cmap对象
    my_cmap = colors.LinearSegmentedColormap.from_list("my_cmap", colors_list)

    # 绘制数据
    psi_grid_df.plot(
        column="PSI",
        cmap=my_cmap,
        ax=ax,
        cax=cax,
        legend=True,
        alpha=0.5,
    )
    # 添加指北针和比例尺
    tbd.plotscale(
        ax,
        bounds=bd,
        textsize=10,
        compasssize=1,
        accuracy=2000,
        rect=[0.06, 0.03],
        zorder=10,
    )
    plt.axis("off")
    plt.xlim(bd[0], bd[2])
    plt.ylim(bd[1], bd[3])

    plt.show()
    # plt.savefig(rf"./result/TIME_WINDOW_NEW/pics/{tw}.png", dpi=300)
