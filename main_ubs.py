import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import model.net_ubs as net
from data.dataset import underdamped_beadspring_dataset
from misc.logger import load_checkpoint, logging_metrics, save_checkpoint
from misc.sampler import CartesianSeqSampler


def train(opt, neep1, neep2, optim, pos_trajs, vel_trajs, sampler):
    neep1.train()
    neep2.train()
    batch = next(sampler)
    xs = pos_trajs[batch].to(opt.device)
    vs = vel_trajs[batch].to(opt.device)

    ent1 = neep1(xs[-1], vs[-1])
    ent2 = neep2(xs, vs)

    optim.zero_grad()
    J1 = (ent1 - torch.exp(-ent1)).mean()
    J2 = (ent2 - torch.exp(-ent2)).mean()
    loss = -J1 - J2
    loss.backward()
    optim.step()

    return J1.item(), J2.item()


def validate(opt, neep1, neep2, pos_trajs, vel_trajs, sampler):
    neep1.eval()
    neep2.eval()

    ret1 = []
    ret2 = []

    J1 = 0
    J2 = 0
    with torch.no_grad():
        for batch in sampler:
            xs = pos_trajs[batch].to(opt.device)
            vs = vel_trajs[batch].to(opt.device)
            ent1 = neep1(xs[-1], vs[-1])
            ent2 = neep2(xs, vs)

            ret1.append(ent1.cpu().squeeze().numpy())
            ret2.append(ent2.cpu().squeeze().numpy())
            J1 += (ent1 - torch.exp(-ent1)).sum().cpu().item()
            J2 += (ent2 - torch.exp(-ent2)).sum().cpu().item()

    J1 = J1 / sampler.size
    J2 = J2 / sampler.size

    ret1 = np.concatenate(ret1)
    ret1 = ret1.reshape(pos_trajs.shape[0], -1)

    ret2 = np.concatenate(ret2)
    ret2 = ret2.reshape(pos_trajs.shape[0], -1)

    return ret1, ret2, J1, J2


def main(opt):
    ##############################
    # Prepare dataset and models #
    ##############################
    trainset = underdamped_beadspring_dataset(opt, seed=0)
    validset = underdamped_beadspring_dataset(opt, seed=1)

    pos_trajs_t, vel_trajs_t = trainset["position"], trainset["velocity"]
    val_pos_trajs_t, val_vel_trajs_t = validset["position"], validset["velocity"]

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    neep1, neep2 = net.__dict__[opt.arch](opt)
    optim = torch.optim.Adam(
        list(neep1.parameters()) + list(neep2.parameters()),
        opt.lr,
    )
    train_sampler = CartesianSeqSampler(
        opt.trj_num, opt.trj_len, opt.seq_len, opt.batch_size, device=opt.device
    )
    val_sampler = CartesianSeqSampler(
        opt.trj_num,
        opt.trj_len,
        opt.seq_len,
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

    for i in tqdm(range(1, opt.n_iter + 1)):
        if i % opt.record_freq == 0 or i == 1:
            preds1, preds2, train_J1, train_J2 = validate(
                opt, neep1, neep2, pos_trajs_t, vel_trajs_t, val_sampler
            )
            preds = preds2 - preds1
            train_log = logging_metrics(
                opt.seq_len, opt.time_step, train_J1, train_J2, preds
            )
            train_log["iteration"] = i
            print(
                "Train  iter: %d  J1: %1.4e  J2: %1.4e  pred: %.5f"
                % (i, train_log["J1"], train_log["J2"], train_log["pred_rate"])
            )

            preds1, preds2, val_J1, val_J2 = validate(
                opt, neep1, neep2, val_pos_trajs_t, val_vel_trajs_t, val_sampler
            )
            preds = preds2 - preds1
            val_log = logging_metrics(opt.seq_len, opt.time_step, val_J1, val_J2, preds)
            val_log["iteration"] = i
            print(
                "Valid  iter: %d  J1: %1.4e  J2: %1.4e  pred: %.5f"
                % (i, val_log["J1"], val_log["J2"], val_log["pred_rate"])
            )

            if i == 1:
                best_val_J1 = val_J1
                best_val_J2 = val_J2
                best_state_dict1 = neep1.state_dict()
                best_state_dict2 = neep2.state_dict()
            else:
                if best_val_J1 < val_J1:
                    best_val_J1 = val_J1
                    best_state_dict1 = neep1.state_dict()
                    save_checkpoint(
                        {
                            "iteration": i,
                            "state_dict": best_state_dict1,
                            "best_J": best_val_J1,
                        },
                        opt.save,
                        "neep1",
                    )
                if best_val_J2 < val_J2:
                    best_val_J2 = val_J2
                    best_state_dict2 = neep2.state_dict()
                    save_checkpoint(
                        {
                            "iteration": i,
                            "state_dict": best_state_dict2,
                            "best_J": best_val_J2,
                        },
                        opt.save,
                        "neep2",
                    )
            val_log["best_J1"] = best_val_J1
            val_log["best_J2"] = best_val_J2
            ret_train.append(train_log)
            ret_val.append(val_log)

        train(opt, neep1, neep2, optim, pos_trajs_t, vel_trajs_t, train_sampler)

    del trainset, validset

    ############################
    # Testing with best models #
    ############################
    load_checkpoint("neep1", opt.save, neep1)
    load_checkpoint("neep2", opt.save, neep2)

    testset = underdamped_beadspring_dataset(opt, seed=2)
    test_pos_trajs_t, test_vel_trajs_t, ents = (
        testset["position"],
        testset["velocity"],
        testset["EP"],
    )

    preds1, preds2, J1, J2 = validate(
        opt, neep1, neep2, test_pos_trajs_t, test_vel_trajs_t, val_sampler
    )
    preds = preds2 - preds1
    test_logs = logging_metrics(opt.seq_len, opt.time_step, J1, J2, preds, ents)
    print(
        "Test  J1: %1.4e  J2: %1.4e  pred: %.5f  R-square: %.5f"
        % (
            test_logs["J1"],
            test_logs["J2"],
            test_logs["pred_rate"],
            test_logs["r_square"],
        )
    )

    ##################################################
    # Save train, valid, test logs & hyperparameters #
    ##################################################
    # train_df = pd.DataFrame(ret_train)
    val_df = pd.DataFrame(ret_val)
    test_df = pd.DataFrame([test_logs])

    # train_df.to_csv(os.path.join(opt.save, "train_log.csv"), index=False)
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
        default=2,
        metavar="N",
        help="number of input neuron = number of beads (default: 2)",
    )
    parser.add_argument(
        "--trj_num",
        "-M",
        type=int,
        default=10000,
        metavar="M",
        help="number of trajectories (default: 10000)",
    )
    parser.add_argument(
        "--trj_len",
        "-L",
        type=int,
        default=1000,
        metavar="L",
        help="number of step for each trajectory (default: 1000)",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=1e-2,
        help="time step size of simulation (default: 0.01)",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=1e-2,
        help="mass (default: 0.01)",
    )
    parser.add_argument(
        "--Tc",
        type=float,
        default=1,
        metavar="T",
        help="Cold heat bath temperature (default: 1)",
    )
    parser.add_argument(
        "--Th",
        type=float,
        default=10,
        metavar="T",
        help="Hot heat bath temperature (default: 10)",
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
        default=0.005,
        metavar="LR",
        help="learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=80000,
        metavar="N",
        help="number of iteration to train (default: 80000)",
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
