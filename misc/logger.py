import os.path as osp

import torch
from scipy import stats


def save_checkpoint(state, path, name="neep1"):
    filename = osp.join(path, "%s.pth.tar" % name)
    torch.save(state, filename)


def load_checkpoint(name, path, model):
    filename = osp.join(path, "%s.pth.tar" % name)
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint '{}' (iteration {})".format(
            filename, checkpoint["iteration"]
        )
    )


def logging_metrics(seq_len, time_step, J1, J2, preds, ents=None):
    tmp = {}
    pred_rate = preds.mean() / (time_step * (seq_len - 1))
    tmp["J1"] = J1
    tmp["J2"] = J2
    pred_rate = (J2 - J1) / ((seq_len - 1) * time_step)
    tmp["pred_rate"] = pred_rate
    if ents is not None:
        _, _, r_value, _, _ = stats.linregress(preds.flatten(), ents.flatten())
        tmp["r_square"] = r_value ** 2
    return tmp


def logging_metrics_omj(J0, J1, J_wtd, mean_time_interval, preds=None, ents=None):
    tmp = {}
    tmp["J0"] = J0
    tmp["J1"] = J1
    tmp["J_wtd"] = J_wtd
    pred_rate = (J1 + J_wtd - 2 * J0) / mean_time_interval 
    tmp["pred_rate"] = pred_rate
    if preds is not None:
        _, _, r_value, _, _ = stats.linregress(preds.flatten(), ents.flatten())
        tmp["r_square"] = r_value ** 2
    return tmp
