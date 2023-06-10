import math
import numpy as np
from datetime import datetime

import pickle
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from matplotlib.cm import ScalarMappable


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def fill_nan(clinic_list):
    mean = np.nanmean(np.asarray(clinic_list))
    return [item if not math.isnan(item) else mean for item in clinic_list]


def my_min_max(data):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
    if isinstance(data, torch.Tensor):
        assert torch.min(data) != torch.max(data)
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data)), float(torch.min(data)), float(torch.max(data))
    else:
        assert np.min(data) != np.max(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data)), np.min(data), np.max(data)


def decode(data_encoded, data_min, data_max):
    return data_encoded * (data_max - data_min) + data_min


def calculate_scores(truth_list, prediction_list, f=None):
    assert len(truth_list) == len(prediction_list)
    TP = sum([1 for truth, prediction in zip(truth_list, prediction_list) if truth == 1 and prediction == 1])
    FP = sum([1 for truth, prediction in zip(truth_list, prediction_list) if truth == 0 and prediction == 1])
    FN = sum([1 for truth, prediction in zip(truth_list, prediction_list) if truth == 1 and prediction == 0])
    TN = sum([1 for truth, prediction in zip(truth_list, prediction_list) if truth == 0 and prediction == 0])

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * (precision * recall) / (precision + recall)

    # print("Length = {}".format(len(truth_list)))
    # print("TP = {}".format(TP))
    # print("FP = {}".format(FP))
    # print("FN = {}".format(FN))
    # print("TN = {}".format(TN))
    # print("Precision = TP / (TP + FP) = {:.4f}".format(precision))
    # print("Recall = TP / (TP + FN) = {:.4f}".format(recall))
    # print("F-score = 2 * (precision * recall) / (precision + recall) = {:.4f}".format(f_score))

    if f is not None:
        f.write("\n")
        f.write("Length = {}\n".format(len(truth_list)))
        f.write("TP = {}\n".format(TP))
        f.write("FP = {}\n".format(FP))
        f.write("FN = {}\n".format(FN))
        f.write("TN = {}\n".format(TN))
        f.write("Accuracy = (TP + TN) / (TP + FP + FN + TN) = {:.4f}\n".format(accuracy))
        f.write("Precision = TP / (TP + FP) = {:.4f}\n".format(precision))
        f.write("Recall = TP / (TP + FN) = {:.4f}\n".format(recall))
        f.write("F-score = 2 * (precision * recall) / (precision + recall) = {:.4f}\n".format(f_score))


def draw_3d_points(data_truth, data_prediction, data_error, save_path, title=None):
    # data_truth = data_truth
    # data_prediction = data_prediction
    np.random.shuffle(data_truth)
    np.random.shuffle(data_prediction)
    np.random.shuffle(data_error)
    fig = plt.figure(figsize=(16, 16))

    truth_y_min = np.min(data_truth[:, -1])
    truth_y_max = np.max(data_truth[:, -1])
    error_y_min = np.min(data_error[:, -1])
    error_y_max = np.max(data_error[:, -1])

    x_label = "k_hyz"
    y_label = "k_pyx"
    z_label = "k_smzx"

    ax1 = fig.add_subplot(221, projection='3d')

    data_truth_0 = data_truth[data_truth[:, -1] == 0]
    data_truth_1 = data_truth[data_truth[:, -1] == 1]

    x0 = [point[0] for point in data_truth_0]
    y0 = [point[1] for point in data_truth_0]
    z0 = [point[2] for point in data_truth_0]

    x1 = [point[0] for point in data_truth_1]
    y1 = [point[1] for point in data_truth_1]
    z1 = [point[2] for point in data_truth_1]

    scatter = ax1.scatter(x0, y0, z0, c="grey", label="Non-osci", alpha=0.2)
    scatter = ax1.scatter(x1, y1, z1, c="red", label="Osci", alpha=0.2)

    ax1.legend(fontsize=20)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel(z_label)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='z', labelsize=10)
    ax1.set_title("Truth of CYCLE_TIME", fontsize=20)


    ax2 = fig.add_subplot(222, projection='3d')

    data_prediction_0 = data_prediction[data_prediction[:, -1] == 0]
    data_prediction_1 = data_prediction[data_prediction[:, -1] == 1]

    x0 = [point[0] for point in data_prediction_0]
    y0 = [point[1] for point in data_prediction_0]
    z0 = [point[2] for point in data_prediction_0]

    x1 = [point[0] for point in data_prediction_1]
    y1 = [point[1] for point in data_prediction_1]
    z1 = [point[2] for point in data_prediction_1]

    scatter = ax2.scatter(x0, y0, z0, c="grey", label="Non-osci", alpha=0.2)
    scatter = ax2.scatter(x1, y1, z1, c="red", label="Osci", alpha=0.2)

    ax2.legend(fontsize=20)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel(z_label)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='z', labelsize=10)
    ax2.set_title("Prediction of CYCLE_TIME", fontsize=20)

    ax3 = fig.add_subplot(223, projection='3d')

    data_error_0 = data_error[data_error[:, -1] == 0]
    data_error_1 = data_error[data_error[:, -1] == 1]

    x0 = [point[0] for point in data_error_0]
    y0 = [point[1] for point in data_error_0]
    z0 = [point[2] for point in data_error_0]
    x1 = [point[0] for point in data_error_1]
    y1 = [point[1] for point in data_error_1]
    z1 = [point[2] for point in data_error_1]

    scatter = ax3.scatter(x0, y0, z0, c="purple", label=f"Truth=0 & Pred=1 ({len(data_error_0)})", alpha=0.5)
    scatter = ax3.scatter(x1, y1, z1, c="lime", label=f"Truth=1 & Pred=0 ({len(data_error_1)})", alpha=0.5)

    ax3.legend(fontsize=20)

    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_zlabel(z_label)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='z', labelsize=10)
    ax3.set_title(f"Error Cases ($n_{{E}}={len(data_error)}$)", fontsize=20)
    #
    # ax4 = fig.add_subplot(224, projection='3d')
    # x = [point[0] for point in data_error_remarkable]
    # y = [point[1] for point in data_error_remarkable]
    # z = [point[2] for point in data_error_remarkable]
    # val = [point[3] for point in data_error_remarkable]
    #
    # cmap = 'coolwarm'
    #
    # scatter = ax4.scatter(x, y, z, c=val, cmap=cmap, alpha=0.4)
    # colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=error_y_min, vmax=error_y_max)), ax=ax4,
    #                         shrink=0.5)
    #
    # ax4.set_xlabel(x_label)
    # ax4.set_ylabel(y_label)
    # ax4.set_zlabel(z_label)
    # ax4.tick_params(axis='x', labelsize=10)
    # ax4.tick_params(axis='y', labelsize=10)
    # ax4.tick_params(axis='z', labelsize=10)
    # ax4.set_title("Remarkable Error ($e>0.1$, $n_{{R}}={0:d}$) Distribution".format(len(data_error_remarkable)), fontsize=20)  # , len(data_error_remarkable) / len(data_error) * 100.0

    #remarkable

    # plt.show()
    if title:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("max", np.max(data_error[:, -1]))
    print("min", np.min(data_error[:, -1]))
    # plot_value_distribution(data_error[:, -1], save_path=save_path.replace(".png", "_distribution.png"))


def plot_value_distribution_bool(data, save_path):
    fig = plt.figure(figsize=(24, 6))
    bin_edges = np.arange(0.0, 2.0, 0.05)

    # Calculate the histogram of the data using the defined bins
    hist, _ = np.histogram(data, bins=bin_edges)

    # Calculate the frequencies as the relative count in each bin
    frequencies = hist / len(data)

    ax = fig.add_subplot(111)

    # Plot the bars with the frequencies
    bars = ax.bar(bin_edges[:-1], frequencies, width=0.05, align='edge', color="orange")

    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')

    # Set x-axis ticks to match the desired range [0.1, 0.2, ..., 0.9, 1.0]
    x_ticks = np.arange(0.0, 2.0, 0.05)
    plt.xticks(x_ticks)

    # Add count labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{hist[i]:d}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')

    plt.savefig(save_path, dpi=300)
    plt.close()


def one_time_draw_3d_points_from_txt_bool(txt_path, save_path, title=None, log_flag=False):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if "," in line and "x" not in line]
    data_truth = []
    data_prediction = []
    data_error = []
    for one_line in lines[:]:
        parts = one_line.split(",")
        if log_flag:
            parts[1] = np.log(float(parts[1]))
            parts[2] = np.log(float(parts[2]))
            parts[3] = np.log(float(parts[3]))
        data_truth.append((float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])))
        data_prediction.append((float(parts[1]), float(parts[2]), float(parts[3]), int(parts[5])))
        if int(parts[4]) != int(parts[5]):
            data_error.append((float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])))
    # print("data_truth:")
    # print(data_truth)
    # print("data_prediction:")
    # print(data_prediction)
    # print("data_error:")
    # print(data_error)
    data_truth = np.asarray(data_truth)
    data_prediction = np.asarray(data_prediction)
    data_error = np.asarray(data_error)
    draw_3d_points(data_truth, data_prediction, data_error, save_path, title)
    print(f"saved \"{title}\" to {save_path}")


def one_time_filter_data(data_path, filter_list):
    with open(data_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if len(line) > 10 and "k_" not in line]
    print(f"Initial: all {len(lines)} lines")

    for one_filter in filter_list:
        save_path = data_path.replace(".csv", f"_{'all' if one_filter > 1000 else one_filter}.csv")
        # with open(save_path, "w") as f_tmp:
        #     pass
        f_write = open(save_path, "w")

        count_inf = 0
        count_normal = 0
        count_normal_remain = 0
        count_bad = 0

        print(f"# filter: <{one_filter} or inf")
        for one_line in lines:
            parts = one_line.split(",")
            c1, c2, c3 = parts[3], parts[4], parts[5]
            if c1 == c2 == c3 == "inf":
                count_inf += 1
                f_write.write(one_line)
            elif c1 == "inf" or c2 == "inf" or c3 == "inf":
                count_bad += 1
            else:
                c1_f, c2_f, c3_f = float(c1), float(c2), float(c3)
                if max(c1_f, c2_f, c3_f) - min(c1_f, c2_f, c3_f) > 5:
                    count_bad += 1
                else:
                    count_normal += 1
                    if c1_f < one_filter:
                        count_normal_remain += 1
                        f_write.write(one_line)
        f_write.close()
        print(f"count_inf: {count_inf}")
        print(f"count_normal: {count_normal} ({count_normal_remain} remain for matching \"<{one_filter}\"))")
        print(f"count_bad: {count_bad}")






if __name__ == "__main__":
    # a = np.asarray([1.0, 2.0, 3.0])
    # a = torch.tensor([1.0, 2.0, 3.0])
    # print(my_min_max(a))
    # data = [(1, 2, 3, 0), (4, 5, 6, 1), (7, 8, 9, 0)]
    # draw_3d_points(data)
    # timestring = "20230603_073335_114703"  # "20230603_044727_785177"
    # one_time_draw_3d_points_from_txt_bool(f"record/output/output_{timestring}_best_train.txt",
    #                                  f"test/comparison_{timestring}_best_train_log.png",
    #                                  title="Results of the Train Set (n=101580) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt_bool(f"record/output/output_{timestring}_best_val.txt",
    #                                  f"test/comparison_{timestring}_best_test_log.png",
    #                                  title="Results of the Test Set (n=25396) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt_bool(f"record/output/output_{timestring}_last_train.txt",
    #                                  f"test/comparison_{timestring}_last_train_log.png",
    #                                  title="Results of the Train Set (n=101580) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt_bool(f"record/output/output_{timestring}_last_val.txt",
    #                                  f"test/comparison_{timestring}_last_test_log.png",
    #                                  title="Results of the Test Set (n=25396) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # with open("data/debug.txt", "w") as f:
    #     pass
    one_time_filter_data("data/dataset_0_1_2_v0604.csv", [999999, 200, 100])
