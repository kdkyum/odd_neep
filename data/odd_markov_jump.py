# Odd-parity Markov-jump process simulation
# Reference: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.050602
import numpy as np


def transition_rates(hv, hmv, fv, fmv, N=4):
    """outputs the matrix for the dynamics of Markovian jump process with odd-parity in a N-site ring.
    Parameters: hv : float
                     transition rate from (n,v) to (n+v,v)
                hmv: float
                     transition rate from (n,-v) to (n-v,-v)
                fv : float
                     transition rate from (n,v) to (n,-v)
                fmv: float
                     transition rate from (n,-v) to (n,+v)
                N  : float, optional
                     the number of sites (default: 4 )
    return : 2D array-like, of shape (2N, 2N)
        Matrix for the dynamics of a state vector [(1,v),...,(N,v),(1,-v),...,(N,-v) ]
    """
    Mv = np.zeros((N, N))

    for i in range(N):
        if i >= 0:  # 1 ~ N
            Mv[i][i - 1] = hv

    Mmv = np.zeros((N, N))

    for i in range(N):
        if i >= 0:  # 1 ~ N
            Mmv[i - 1][i] = hmv

    OFv = np.zeros((N, N))

    for i in range(N):
        if i >= 0:  # 1 ~ N
            OFv[i][i] = fv

    OFmv = np.zeros((N, N))

    for i in range(N):
        if i >= 0:  # 1 ~ N
            OFmv[i][i] = fmv

    # Final
    result_mat = np.zeros((2 * N, 2 * N))

    result_mat[:N, :N] = Mv
    result_mat[N:, N:] = Mmv
    result_mat[N:, :N] = OFv
    result_mat[:N, N:] = OFmv
    for i in range(2 * N):
        result_mat[i, i] = -sum(result_mat[:, i])

    return result_mat


def jump_probs(w):
    mask = 1 - np.eye(w.shape[0])
    w = mask * w
    mean_waiting_time = 1 / np.sum(w, axis=0)[np.newaxis, :]
    return w * mean_waiting_time


def visit_probs(w):
    jump_probabilities = jump_probs(w)
    eigenvalues, eigenvectors = np.linalg.eig(jump_probabilities)
    max_eigen = max(eigenvalues.real)
    max_index = np.where(eigenvalues == max_eigen)

    visit_state = eigenvectors[:, max_index]
    if sum(visit_state) < 0:
        visit_state = -visit_state
    visit_state = visit_state / sum(visit_state)
    assert visit_state.imag.any() == 0, "imaginary!"
    return visit_state.real.squeeze()


def p_ss(hv, hmv, fv, fmv, N):
    """outputs the steady state vector of Markovian jump process with odd-parity in a N-site ring.
    Parameters: hv : float
                     transition rate from (n,v) to (n+v,v)
                hmv: float
                     transition rate from (n,-v) to (n-v,-v)
                fv : float
                     transition rate from (n,v) to (n,-v)
                fmv: float
                     transition rate from (n,-v) to (n,+v)
                N  : float
                     the number of sites
    return : 1D array-like, of shape (2N)
        A state vector [(1,v),...,(N,v),(1,-v),...,(N,-v) ]
    """
    tran_mat = transition_rates(hv, hmv, fv, fmv, N=N)
    eigenvalues, eigenvectors = np.linalg.eig(tran_mat)
    max_eigen = max(eigenvalues.real)
    max_index = np.where(eigenvalues == max_eigen)

    assert int(max_eigen) == 0, "the dynamics diverges!"

    steady_state = eigenvectors[:, max_index]
    if sum(steady_state) < 0:
        steady_state = -steady_state
    steady_state = steady_state / sum(steady_state)
    assert steady_state.imag.any() == 0, "imaginary!"
    return steady_state.real


def simulation(num_trjs, trj_len, hv, hmv, fv, fmv, N, seed=0):
    """Simulation with Gillespie algorithm of an odd-parity Markov jump process
        each trajectory has own time series

    Args:
        num_trjs : Number of trajectories you want
        trj_len : length of trajectories
        hv : float
             transition rate from (n,v) to (n+v,v)
        hmv: float
             transition rate from (n,-v) to (n-v,-v)
        fv : float
             transition rate from (n,v) to (n,-v)
        fmv: float
             transition rate from (n,-v) to (n,+v)
        N  : float
             the number of sites
        seed : seed of random generator (default: 0)

    Returns: 2d-array of shape (num_trjs, trj_len)
        trajectories of an odd-parity Markov jump process
    """
    np.random.seed(seed)
    tran_mat = transition_rates(hv, hmv, fv, fmv, N=N)
    trajs = []
    times = []
    ti = np.zeros((num_trjs,))
    p_ss_sq = np.squeeze(p_ss(hv, hmv, fv, fmv, N))
    states = np.random.choice(np.size(p_ss_sq), size=(num_trjs,), p=p_ss_sq)
    trajs.append(np.copy(states))
    times.append(ti)

    for i in range(trj_len - 1):
        time_interval = np.random.uniform(0.0, 1.0, size=(num_trjs, 2 * N))
        # calculates time intervals !
        for (ens_idx, dest_idx), rand_val in np.ndenumerate(time_interval):
            tran_rate = tran_mat[dest_idx, states[ens_idx]]
            if tran_rate == 0 or states[ens_idx] == dest_idx:
                time_interval[ens_idx, dest_idx] = np.nan
            else:
                time_interval[ens_idx, dest_idx] = -np.log(rand_val) / tran_rate
        update_ti = np.copy(ti)
        next_states = np.copy(states)
        for i, each_intervals in enumerate(time_interval):
            smallest_interval = np.nanmin(each_intervals)
            small_idx = np.where(each_intervals == smallest_interval)
            next_states[i] = small_idx[0]
            update_ti[i] += smallest_interval
            assert (
                len(each_intervals[small_idx]) == 1
            ), "more than one... or none " + str(each_intervals[small_idx])

        assert len(time_interval) == num_trjs, "wrong! " + str(len(time_interval))

        trajs.append(np.copy(next_states))
        times.append(np.copy(update_ti))
        states = np.copy(next_states)
        ti = np.copy(update_ti)

    return np.array(trajs).T, np.array(times).T


def ep_rate(hv, hmv, fv, fmv, N):
    """Analytic average entropy production per step

    Args:
        hv : float
             transition rate from (n,v) to (n+v,v)
        hmv: float
             transition rate from (n,-v) to (n-v,-v)
        fv : float
             transition rate from (n,v) to (n,-v)
        fmv: float
             transition rate from (n,-v) to (n,+v)
        N  : float
             the number of sites

    Returns:
        analytic average entropy production per step
    """
    tran_mat = transition_rates(hv, hmv, fv, fmv, N=N)
    stationary = p_ss(hv, hmv, fv, fmv, N)
    ent_rate = 0
    for i, prob_i in enumerate(stationary):
        for j, tr in enumerate(tran_mat[:, i]):
            tr_rev = tran_mat[i - N, j - N]
            if tr != 0 and i != j:
                ent_rate += prob_i * tr * np.log(tr / tr_rev)
            if i == j:
                ent_rate += prob_i * (tr - tr_rev)
    return ent_rate[0, 0]


def analytic_etpy(traj, time_traj, hv, hmv, fv, fmv, N):
    """Analytic stochastic entropy production for given trajectories.
        this code is for simulationv2

    Args:
        traj : trajectories, shape=(num_trjs, trj_len)
        hv : float
             transition rate from (n,v) to (n+v,v)
        hmv: float
             transition rate from (n,-v) to (n-v,-v)
        fv : float
             transition rate from (n,v) to (n,-v)
        fmv: float
             transition rate from (n,-v) to (n,+v)
        N  : float
             the number of sites

    Returns:
        1d-array of shape (num_trjs, trj_len - 1)
    """

    tran_mat = transition_rates(hv, hmv, fv, fmv, N=N)
    mask = 1 - np.eye(tran_mat.shape[0])
    tau = 1 / np.sum(mask * tran_mat, axis=0)
    wtd = lambda t, n: np.exp(-t / tau[n]) / tau[n]
    P = jump_probs(tran_mat)
    R = visit_probs(tran_mat)

    waiting_time = time_traj[:, 1:] - time_traj[:, :-1]
    ent_wtd = np.log(
        wtd(waiting_time, traj[:, :-1]) / wtd(waiting_time, traj[:, :-1] - N)
    )
    ent_sys = np.log(R[traj[:, :-1]] / R[traj[:, 1:]])
    ent_aff = np.log(
        P[traj[:, 1:], traj[:, :-1]] / P[traj[:, :-1] - N, traj[:, 1:] - N]
    )
    ent = ent_aff + ent_wtd + ent_sys
    ent_rate = ent.sum(-1) / time_traj[:, -1]
    return ent, ent_rate
