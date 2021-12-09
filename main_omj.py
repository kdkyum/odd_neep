import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import model.net_omj as net
from data.dataset import odd_markov_jump_dataset
from misc.logger import load_checkpoint, logging_metrics_omj, save_checkpoint
from misc.sampler import CartesianSeqSampler


def train(opt, neep1, neep2, neep_wtd, optim, trajs, times, sampler):
    neep1.train()
    neep2.train()
    neep_wtd.train()
    batch = next(sampler)
    xs = trajs[batch].to(opt.device).transpose(0, 1)
    ts = times[batch].to(opt.device).transpose(0, 1)

    as0 = neep1(xs[:, 0])
    out1 = neep2(xs)
    out2 = neep_wtd(xs[:, 0], ts[:, 1] - ts[:, 0])
    optim.zero_grad()
    J0 = (1 + as0 - torch.exp(-as0)).mean()
    J1 = (1 + out1 - torch.exp(-out1)).mean()
    J_wtd = (1 + out2 - torch.exp(-out2)).mean()
    loss = -J0 - J1 - J_wtd
    loss.backward()
    optim.step()

    return J0.item(), J1.item(), J_wtd.item()


def validate(opt, neep1, neep2, neep_wtd, trajs, times, sampler):
    neep1.eval()
    neep2.eval()
    neep_wtd.eval()

    preds = []
    J0, J1, J_wtd = 0, 0, 0

    with torch.no_grad():
        for batch in sampler:
            xs = trajs[batch].to(opt.device).transpose(0, 1)
            ts = times[batch].to(opt.device).transpose(0, 1)

            as0 = neep1(xs[:, 0])
            as1 = neep1(xs[:, 1])
            out1 = neep2(xs)
            out2 = neep_wtd(xs[:, 0], ts[:, 1] - ts[:, 0])
            J0 += (1 + as0 - torch.exp(-as0)).sum().cpu().item()
            J1 += (1 + out1 - torch.exp(-out1)).sum().cpu().item()
            J_wtd += (1 + out2 - torch.exp(-out2)).sum().cpu().item()

            pred = (out1 + out2 - as0 - as1).squeeze().cpu().numpy()
            preds.append(pred)

    J0 = J0 / sampler.size
    J1 = J1 / sampler.size
    J_wtd = J_wtd / sampler.size
    preds = np.concatenate(preds)
    preds = preds.reshape(trajs.shape[0], -1)
    return J0, J1, J_wtd, preds


def main(opt):
    ##############################
    # Prepare dataset and models #
    ##############################
    opt.fmv = opt.hv
    opt.hmv = opt.hv + 0.1 * opt.c
    opt.fv = opt.hv + 0.2 * opt.c
    trainset = odd_markov_jump_dataset(opt, seed=0)
    validset = odd_markov_jump_dataset(opt, seed=1)

    trajs_t, times_t = trainset["jump_trajs"], trainset["jump_times"]
    val_trajs_t, val_times_t = validset["jump_trajs"], validset["jump_times"]

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    neep1, neep2, neep_wtd = net.__dict__[opt.arch](opt)
    optim = torch.optim.Adam(
        list(neep1.parameters())
        + list(neep2.parameters())
        + list(neep_wtd.parameters()),
        opt.lr,
    )
    train_sampler = CartesianSeqSampler(
        opt.trj_num, opt.trj_len, 2, opt.batch_size, device=opt.device
    )
    val_sampler = CartesianSeqSampler(
        opt.trj_num,
        opt.trj_len,
        2,
        opt.test_batch_size,
        device=opt.device,
        train=False,
    )

    ############
    # Training #
    ############
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    ret_train = []
    ret_val = []
    # mean time interval
    train_mti = (times_t[:, 1:] - times_t[:, :-1]).mean().cpu().item()
    val_mti = (val_times_t[:, 1:] - val_times_t[:, :-1]).mean().cpu().item()

    for i in tqdm(range(1, opt.n_iter + 1)):
        if i % opt.record_freq == 0 or i == 1:
            train_J0, train_J1, train_J_wtd, _ = validate(
                opt, neep1, neep2, neep_wtd, trajs_t, times_t, val_sampler
            )
            train_log = logging_metrics_omj(train_J0, train_J1, train_J_wtd, train_mti)
            train_log["iteration"] = i
            print(
                "Train  iter: %d  J0: %1.4e  J1: %1.4e  Jwtd: %1.4e  pred: %.5f"
                % (
                    i,
                    train_log["J0"],
                    train_log["J1"],
                    train_log["J_wtd"],
                    train_log["pred_rate"],
                )
            )

            val_J0, val_J1, val_J_wtd, _ = validate(
                opt, neep1, neep2, neep_wtd, val_trajs_t, val_times_t, val_sampler
            )
            val_log = logging_metrics_omj(val_J0, val_J1, val_J_wtd, val_mti)
            val_log["iteration"] = i
            print(
                "Valid  iter: %d  J0: %1.4e  J1: %1.4e  Jwtd: %1.4e  pred: %.5f"
                % (
                    i,
                    val_log["J0"],
                    val_log["J1"],
                    val_log["J_wtd"],
                    val_log["pred_rate"],
                )
            )

            if i == 1:
                best_val_J0 = val_J0
                best_val_J1 = val_J1
                best_val_J_wtd = val_J_wtd
                best_state_dict1 = neep1.state_dict()
                best_state_dict2 = neep2.state_dict()
                best_state_dict3 = neep_wtd.state_dict()
            else:
                if best_val_J0 < val_J0:
                    best_val_J0 = val_J0
                    best_state_dict1 = neep1.state_dict()
                    save_checkpoint(
                        {
                            "iteration": i,
                            "state_dict": best_state_dict1,
                            "best_J": best_val_J0,
                        },
                        opt.save,
                        "neep1",
                    )
                if best_val_J1 < val_J1:
                    best_val_J1 = val_J1
                    best_state_dict2 = neep2.state_dict()
                    save_checkpoint(
                        {
                            "iteration": i,
                            "state_dict": best_state_dict2,
                            "best_J": best_val_J1,
                        },
                        opt.save,
                        "neep2",
                    )
                if best_val_J_wtd < val_J_wtd:
                    best_val_J_wtd = val_J_wtd
                    best_state_dict2 = neep2.state_dict()
                    save_checkpoint(
                        {
                            "iteration": i,
                            "state_dict": best_state_dict3,
                            "best_J": best_val_J_wtd,
                        },
                        opt.save,
                        "neep_wtd",
                    )
            val_log["best_J0"] = best_val_J0
            val_log["best_J1"] = best_val_J1
            val_log["best_J_wtd"] = best_val_J_wtd
            ret_train.append(train_log)
            ret_val.append(val_log)

        train(opt, neep1, neep2, neep_wtd, optim, trajs_t, times_t, train_sampler)

    del trainset, validset

    ############################
    # Testing with best models #
    ############################
    load_checkpoint("neep1", opt.save, neep1)
    load_checkpoint("neep2", opt.save, neep2)
    load_checkpoint("neep_wtd", opt.save, neep_wtd)

    testset = odd_markov_jump_dataset(opt, seed=2)
    test_trajs_t, test_times_t, ents = (
        testset["jump_trajs"],
        testset["jump_times"],
        testset["EP"],
    )
    test_mti = (test_times_t[:, 1:] - test_times_t[:, :-1]).mean().cpu().item()

    J0, J1, J_wtd, preds = validate(
        opt, neep1, neep2, neep_wtd, test_trajs_t, test_times_t, val_sampler
    )
    test_logs = logging_metrics_omj(J0, J1, J_wtd, test_mti, preds, ents)
    print(
        "Test  J0: %1.4e  J1: %1.4e  Jwtd: %1.4e  pred: %.5f  R-square: %.5f"
        % (
            test_logs["J0"],
            test_logs["J1"],
            test_logs["J_wtd"],
            test_logs["pred_rate"],
            test_logs["r_square"],
        )
    )

    ##################################################
    # Save train, valid, test logs & hyperparameters #
    ##################################################
    train_df = pd.DataFrame(ret_train)
    val_df = pd.DataFrame(ret_val)
    test_df = pd.DataFrame([test_logs])

    train_df.to_csv(os.path.join(opt.save, "train_log.csv"), index=False)
    val_df.to_csv(os.path.join(opt.save, "val_log.csv"), index=False)
    test_df.to_csv(os.path.join(opt.save, "test_log.csv"), index=False)
    opt.device = "cuda" if use_cuda else "cpu"
    hparams = json.dumps(vars(opt))
    with open(os.path.join(opt.save, "hparams.json"), "w") as f:
        f.write(hparams)


if __name__ == "__main__":
    arch_names = sorted(
        name
        for name in net.__dict__
        if name.islower() and not name.startswith("__") and callable(net.__dict__[name])
    )
    parser = argparse.ArgumentParser(
        description="Neural Entropy Production Estimator for multi bead-spring model"
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="mlp",
        choices=arch_names,
        help="model architecture: " + " | ".join(arch_names) + " (default: mlp)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=3,
        metavar="N",
        help="number of input neuron = number of beads (default: 3)",
    )
    parser.add_argument(
        "--trj_num",
        "-M",
        type=int,
        default=1000,
        metavar="M",
        help="number of trajectories (default: 1000)",
    )
    parser.add_argument(
        "--trj_len",
        "-L",
        type=int,
        default=10000,
        metavar="L",
        help="number of step for each trajectory (default: 10000)",
    )
    parser.add_argument(
        "--hv",
        type=float,
        default=1.0,
        help="hv (default: 1.0)",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.1,
        help="c (default: 0.1)",
    )
    parser.add_argument(
        "--save",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        metavar="N",
        help="input batch size for training (default: 4096)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for testing (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        metavar="LR",
        help="learning rate (default: 0.00001)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100000,
        metavar="N",
        help="number of iteration to train (default: 100000)",
    )
    parser.add_argument(
        "--record_freq",
        type=int,
        default=1000,
        metavar="N",
        help="recording frequency (default: 1000)",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=1,
        metavar="N",
        help="number of MLP layer (default: 1)",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=200,
        metavar="N",
        help="number of hidden neuron (default: 200)",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )

    opt = parser.parse_args()
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")

    main(opt)
