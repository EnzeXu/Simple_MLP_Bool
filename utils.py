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


def draw_3d_points(data_truth, data_prediction, save_path):
    data_truth = data_truth[:]
    data_prediction = data_prediction[:]
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(121, projection='3d')
    x = [point[0] for point in data_truth]
    y = [point[1] for point in data_truth]
    z = [point[2] for point in data_truth]
    val = [point[3] for point in data_truth]

    cmap = 'cool'

    scatter = ax.scatter(x, y, z, c=val, cmap=cmap)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap), ax=ax, shrink=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    ax.set_title("Truth of Test Set")


    ax = fig.add_subplot(122, projection='3d')
    x = [point[0] for point in data_prediction]
    y = [point[1] for point in data_prediction]
    z = [point[2] for point in data_prediction]
    val = [point[3] for point in data_prediction]

    cmap = 'cool'

    scatter = ax.scatter(x, y, z, c=val, cmap=cmap)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap), ax=ax, shrink=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    ax.set_title("Prediction of Test Set")

    # plt.show()
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8)
    plt.savefig(save_path, dpi=400)
    plt.close()


def one_time_draw_3d_points_from_txt(txt_path, save_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if "," in line and "x" not in line]
    data_truth = []
    data_prediction = []
    for one_line in lines:
        parts = one_line.split(",")
        data_truth.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        data_prediction.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])))
    print("data_truth:")
    print(data_truth)
    print("data_prediction:")
    print(data_prediction)
    # draw_3d_points(data_truth, data_prediction, save_path)


# def one_time_draw_3d_points_from_txt(txt_path, save_path):
#     with open(txt_path, "r") as f:
#         lines = f.readlines()
#     lines = [line for line in lines if "," in line and "x" not in line]
#     data_truth = []
#     data_prediction = []
#     for one_line in lines:
#         parts = one_line.split(",")
#         data_truth.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
#         data_prediction.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])))
#     print("data_truth:")
#     print(data_truth)
#     print("data_prediction:")
#     print(data_prediction)
#     # draw_3d_points(data_truth, data_prediction, save_path)



if __name__ == "__main__":
    # a = np.asarray([1.0, 2.0, 3.0])
    # a = torch.tensor([1.0, 2.0, 3.0])
    # print(my_min_max(a))
    # data = [(1, 2, 3, 0), (4, 5, 6, 1), (7, 8, 9, 0)]
    # draw_3d_points(data)
    one_time_draw_3d_points_from_txt("test/test.txt", "test/comparison.png")
