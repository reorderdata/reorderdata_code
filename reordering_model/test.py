import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from data import TestContinuousMatrixDataset, TestMatrixDataset
from model import DAO_ResNet_Matmul_Sinkhorn, PairwiseDistanceLayer
from utils import (calc_multi_conv, calc_multi_conv_continuous,
                   calc_penalty_score, img_write_new)

parser = argparse.ArgumentParser()
# parser.add_argument("--epoches", type=int, default=300, help="epoches")
parser.add_argument("--output_dir", type=str, default='./output/',
                    help="model dir")
parser.add_argument("--model_path", type=str, default='.', help="model path")
parser.add_argument("--data_path", type=str, default='.', help="data path")
parser.add_argument("--scorer_path", type=str, default='.', help="scorer path")
parser.add_argument("--tta_iter", type=int, default=20)
parser.add_argument("--gpu", type=str, default='0', help="gpu id")
parser.add_argument("--model_dim", type=int, default=128)
parser.add_argument("--mat_size", type=int, default=200)
parser.add_argument("--pattern_type", type=str, choices=[None, "block","offblock","star","band"], default=None)
scorer_dim_map = {"block":0,"star":1,"offblock":2,"band":3}
parser.add_argument("--continuous_eval", action="store_true", default=False)

# add use_scheduler
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

print(args)
json.dump(vars(args), open(os.path.join(args.output_dir,'args.json'),'w'))

np.random.seed(101)
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = DAO_ResNet_Matmul_Sinkhorn(args.model_dim)
checkpoint = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(checkpoint)


scorer = torchvision.models.convnext_tiny(weights=None)
first_conv_layer = scorer.features[0][0]
scorer.features[0][0] = nn.Sequential(
    PairwiseDistanceLayer(),
    nn.Conv2d(2, 3, kernel_size=1, padding=0),
    first_conv_layer
)
fc_layer = nn.Linear(768, 1 if args.pattern_type is None else 4)
sigmoid = nn.Sigmoid()
scorer.classifier[2] = nn.Sequential(fc_layer, sigmoid)

checkpoint = torch.load(args.scorer_path, map_location='cpu')
scorer.load_state_dict(checkpoint)

if not args.continuous_eval:
    val_dataset = TestMatrixDataset(args.data_path)
else:
    val_dataset = TestContinuousMatrixDataset(args.data_path)


model.to(device)
model.eval()

scorer.to(device)
scorer.eval()

output_img_dir = os.path.join(args.output_dir, 'images')
os.makedirs(output_img_dir, exist_ok=True)
img_idx = 0
scores = []
time_1 = 0
time_tta = 0
def get_score(mat, patterns):
    if not args.continuous_eval:
        match_res = calc_multi_conv(mat, patterns, args.mat_size)
        match_res['score'], match_res['scores'] = calc_penalty_score(mat, match_res)
    else:
        match_res = calc_multi_conv_continuous(mat, patterns, args.mat_size)
        # return calc_penalty_score(mat, match_res, args.mat_size)
    return match_res['score']

with torch.no_grad():
    # for X, y, score_upb, patterns in tqdm(val_dataset):
    for data_tuple in tqdm(val_dataset):
        X = data_tuple[0]
        y = data_tuple[1]
        score_upb = data_tuple[2]
        patterns = data_tuple[3]
        if len(data_tuple) == 5:
            period = data_tuple[4]
        else:
            period = None

        X = X.to(device)

        start_time = time.time()

        pred = model.infer(X[None, :, :, :])

        time_1 += time.time() - start_time

        tta_pred = [X, pred]
        for _ in range(args.tta_iter - 1):
            tta_pred.append(model.infer(tta_pred[-1][:,None,:,:]))
        tta_scores = [scorer(_[:,None,:,:])[0,0 if args.pattern_type is None else scorer_dim_map[args.pattern_type]].item() for _ in tta_pred]

        time_tta += time.time() - start_time

        tta_best_mat = tta_pred[np.argmax(tta_scores)]

        input_score = get_score(X[0].squeeze(0).cpu().numpy(), patterns)
        pred_score = get_score(pred[0].squeeze(0).cpu().numpy(), patterns)
        tta_best_score = get_score(tta_best_mat[0].squeeze(0).cpu().numpy(), patterns)
        scores.append({'input_score': input_score, 'pred_score':pred_score, 'tta_pred_score':tta_best_score,'upb_score':score_upb})

        if img_idx % 256 == 0:
            img_write_new(os.path.join(output_img_dir, f'{img_idx}_input.png'),
                          X[0].detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir, f'{img_idx}_gt.png'),
                          y.detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir, f'{img_idx}_pred.png'),
                          pred[0].detach().cpu().numpy())
            img_write_new(os.path.join(output_img_dir, f'{img_idx}_tta_pred.png'),
                          tta_best_mat[0].detach().cpu().numpy())
        img_idx += 1

torch.save(scores, os.path.join(args.output_dir, "scores.pth"))

fp = open(os.path.join(args.output_dir, "results.txt"),"w")
fp.write(f"Mean score: {np.mean([np.minimum(_['tta_pred_score'] / _['upb_score'],1) for _ in scores])}\n")
fp.write(f"Mean inference time: {time_tta / len(val_dataset)}\n")
