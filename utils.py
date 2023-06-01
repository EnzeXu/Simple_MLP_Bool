import math
import numpy as np
from datetime import datetime

import pickle
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader


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



if __name__ == "__main__":
    # a = np.asarray([1.0, 2.0, 3.0])
    a = torch.tensor([1.0, 2.0, 3.0])
    print(my_min_max(a))
