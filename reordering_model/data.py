import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import calc_multi_conv, calc_multi_conv_continuous, calc_penalty_score


class MatrixDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        if data_dir.split(".")[-1] == "npy":
            dataset = np.load(data_dir, allow_pickle=True).item()
        elif data_dir.split(".")[-1] == "npz":
            dataset = np.load(data_dir, allow_pickle=True)
            dataset = {_: dataset[_].item() for _ in dataset.files}
        else:
            raise NotImplementedError

        if indices is None:
            self.input_matrix = [_["swapped_noise_mat"] for _ in dataset.values()]
            self.target_matrix = [_["noise_mat"] for _ in dataset.values()]
        else:
            dataset_values = list(dataset.values())
            self.input_matrix = [
                dataset_values[_]["swapped_noise_mat"] for _ in indices
            ]
            self.target_matrix = [dataset_values[_]["noise_mat"] for _ in indices]

    def __len__(self):
        return len(self.input_matrix)

    def __getitem__(self, index):
        return (
            transforms.ToTensor()(self.input_matrix[index]).to(torch.float),
            transforms.ToTensor()(self.target_matrix[index]).to(torch.float).squeeze(),
        )


class DiffMatrixDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        if data_dir.split(".")[-1] == "npy":
            dataset = np.load(data_dir, allow_pickle=True).item()
        elif data_dir.split(".")[-1] == "npz":
            dataset = np.load(data_dir, allow_pickle=True)
            dataset = {_: dataset[_].item() for _ in dataset.files}
        else:
            raise NotImplementedError

        dataset = {"_".join(k.split("_")[:-4]): v for k, v in dataset.items()}
        dataset_keys = list(dataset.keys())
        dataset_values = list(dataset.values())
        n_mat = len(dataset_values)
        if indices is None:
            indices = np.arange(n_mat)

        self.input_matrix = [dataset_values[_]["swapped_noise_mat"] for _ in indices]
        self.target_matrix = []
        for _ in indices:
            key_list = dataset_keys[_].split("_")
            swap_count = eval(key_list[-1])
            key_list[-1] = str(
                (swap_count - 1 - (swap_count - 1) % 10) if swap_count > 0 else 0
            )
            self.target_matrix.append(dataset["_".join(key_list)]["swapped_noise_mat"])

    def __len__(self):
        return len(self.input_matrix)

    def __getitem__(self, index):
        return (
            transforms.ToTensor()(self.input_matrix[index]).to(torch.float),
            transforms.ToTensor()(self.target_matrix[index]).to(torch.float).squeeze(),
        )


class SwappedMatrixDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        if data_dir.split(".")[-1] == "npy":
            dataset = np.load(data_dir, allow_pickle=True).item()
        elif data_dir.split(".")[-1] == "npz":
            dataset = np.load(data_dir, allow_pickle=True)
            dataset = {_: dataset[_].item() for _ in dataset.files}
        else:
            raise NotImplementedError
        dataset = {"_".join(k.split("_")[:-4]): v for k, v in dataset.items()}
        dataset_keys = list(dataset.keys())
        dataset_values = list(dataset.values())
        n_mat = len(dataset_values)
        if indices is None:
            indices = np.arange(n_mat)

        self.input_matrix = []
        self.target_matrix = []
        for _ in indices:
            key_list = dataset_keys[_].split("_")
            swap_count = eval(key_list[-1])
            if swap_count > 0:
                self.input_matrix.append(dataset[dataset_keys[_]]["swapped_noise_mat"])
                self.target_matrix.append(dataset[dataset_keys[_]]["noise_mat"])

    def __len__(self):
        return len(self.input_matrix)

    def __getitem__(self, index):
        return (
            transforms.ToTensor()(self.input_matrix[index]).to(torch.float),
            transforms.ToTensor()(self.target_matrix[index]).to(torch.float).squeeze(),
        )


class TestMatrixDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        if data_dir.split(".")[-1] == "npy":
            dataset = np.load(data_dir, allow_pickle=True).item()
        elif data_dir.split(".")[-1] == "npz":
            dataset = np.load(data_dir, allow_pickle=True)
            dataset = {_: dataset[_].item() for _ in dataset.files}
        else:
            raise NotImplementedError
        dataset = {"_".join(k.split("_")[:-1]): v for k, v in dataset.items()}
        dataset_keys = list(dataset.keys())
        dataset_values = list(dataset.values())
        n_mat = len(dataset_values)
        if indices is None:
            indices = np.arange(n_mat)

        self.input_matrix = [dataset_values[_]["swapped_noise_mat"] for _ in indices]
        self.target_matrix = [dataset_values[_]["noise_mat"] for _ in indices]
        self.mat_pattern = [dataset_values[_]["mat_pattern"] for _ in indices]
        self.score_upb = []
        for _ in tqdm(indices):
            key_list = dataset_keys[_].split("_")
            key_list[key_list.index("swap") + 1] = "0"
            if "_".join(key_list) in dataset.keys():
                self.score_upb.append(dataset["_".join(key_list)]["score"])
            else:
                match_res = calc_multi_conv(
                    dataset_values[_]["noise_mat"].astype(float),
                    dataset_values[_]["mat_pattern"],
                    200,
                )
                match_res["score"], match_res["scores"] = calc_penalty_score(
                    dataset_values[_]["noise_mat"].astype(float), match_res
                )
                # return calc_penalty_score(mat, match_res, args.mat_size)
                self.score_upb.append(match_res["score"])
                # print("Warning: Calculating score in dataset, please ensure correct mat_size")
                # match_res = calc_multi_conv(dataset_values[_]['noise_mat'], dataset_values[_]['mat_pattern'], dataset_values[_]['noise_mat'].shape[-1])
                # self.score_upb.append(match_res['score'])

    def __len__(self):
        return len(self.input_matrix)

    def __getitem__(self, index):
        return (
            transforms.ToTensor()(self.input_matrix[index]).to(torch.float),
            transforms.ToTensor()(self.target_matrix[index]).to(torch.float).squeeze(),
            self.score_upb[index],
            self.mat_pattern[index],
        )


class TestContinuousMatrixDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        if data_dir.split(".")[-1] == "npy":
            dataset = np.load(data_dir, allow_pickle=True).item()
        elif data_dir.split(".")[-1] == "npz":
            dataset = np.load(data_dir, allow_pickle=True)
            dataset = {_: dataset[_].item() for _ in dataset.files}
        else:
            raise NotImplementedError
        dataset = {"_".join(k.split("_")[:-1]): v for k, v in dataset.items()}
        dataset_keys = list(dataset.keys())
        dataset_values = list(dataset.values())
        n_mat = len(dataset_values)
        if indices is None:
            indices = np.arange(n_mat)

        self.input_matrix = [dataset_values[_]["swapped_noise_mat"] for _ in indices]
        self.target_matrix = [dataset_values[_]["noise_mat"] for _ in indices]
        self.mat_pattern = [dataset_values[_]["mat_pattern"] for _ in indices]
        self.periods = [
            (
                dataset_values[_]["period"]
                if "period" in dataset_values[_].keys()
                else None
            )
            for _ in indices
        ]
        self.score_upb = []
        for _ in tqdm(indices):
            key_list = dataset_keys[_].split("_")
            key_list[key_list.index("swap") + 1] = "0"
            if "_".join(key_list) in dataset.keys():
                self.score_upb.append(dataset["_".join(key_list)]["score"])
            else:
                match_res = calc_multi_conv_continuous(
                    dataset_values[_]["noise_mat"].astype(float),
                    dataset_values[_]["mat_pattern"],
                    200,
                )
                # return calc_penalty_score(mat, match_res, args.mat_size)
                self.score_upb.append(match_res["score"])
                # print("Warning: Calculating score in dataset, please ensure correct mat_size")
                # match_res = calc_multi_conv(dataset_values[_]['noise_mat'], dataset_values[_]['mat_pattern'], dataset_values[_]['noise_mat'].shape[-1])
                # self.score_upb.append(match_res['score'])

    def __len__(self):
        return len(self.input_matrix)

    def __getitem__(self, index):
        return (
            transforms.ToTensor()(self.input_matrix[index]).to(torch.float),
            transforms.ToTensor()(self.target_matrix[index]).to(torch.float).squeeze(),
            self.score_upb[index],
            self.mat_pattern[index],
            self.periods[index],
        )
