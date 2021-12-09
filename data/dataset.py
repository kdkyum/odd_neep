import torch


def underdamped_beadspring_dataset(opt, seed=0):
    from data.underdamped_bead_spring import (
        simulation,
        del_medium_etpy,
        del_shannon_etpy,
    )

    pos_trajs, vel_trajs = simulation(
        opt.trj_num, opt.trj_len, opt.N, opt.Tc, opt.Th, opt.time_step, opt.m, seed=seed
    )

    pos_trajs_t = torch.from_numpy(pos_trajs).to(opt.device).float()
    vel_trajs_t = torch.from_numpy(vel_trajs).to(opt.device).float()

    ents_m = del_medium_etpy(pos_trajs, vel_trajs, opt.N, opt.Tc, opt.Th, opt.m)
    ents_s = del_shannon_etpy(pos_trajs, vel_trajs, opt.N, opt.Tc, opt.Th, opt.m)
    ents = ents_m + ents_s
    if opt.seq_len > 2:
        ents = torch.conv1d(
            torch.tensor(ents).float().view(opt.trj_num, 1, -1),
            torch.ones(1, 1, opt.seq_len - 1),
            stride=1,
            padding=0,
        ).squeeze(-1)

    pos_mean, pos_std = pos_trajs_t.mean(axis=(0, 1)), pos_trajs_t.std(axis=(0, 1))
    vel_mean, vel_std = vel_trajs_t.mean(axis=(0, 1)), vel_trajs_t.std(axis=(0, 1))
    pos_trajs_t = (pos_trajs_t - pos_mean) / pos_std
    vel_trajs_t = (vel_trajs_t - vel_mean) / vel_std

    return {"position": pos_trajs_t, "velocity": vel_trajs_t, "EP": ents}


def odd_markov_jump_dataset(opt, seed=0):
    from data.odd_markov_jump import simulation, analytic_etpy

    trajs, times = simulation(
        opt.trj_num, opt.trj_len, opt.hv, opt.hmv, opt.fv, opt.fmv, opt.N, seed=seed
    )
    ents, _ = analytic_etpy(trajs, times, opt.hv, opt.hmv, opt.fv, opt.fmv, opt.N)
    trajs = torch.from_numpy(trajs).long()
    times = torch.from_numpy(times).float()

    return {"jump_trajs": trajs, "jump_times": times, "EP": ents}