# Velocity-Verlet algorithm
# Reference: http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import expm

kB = 1
k = 1
e = 1


def precision_ss(N, Tc, Th, m=1):
    Ti = np.linspace(Tc, Th, N)
    D = np.zeros((2 * N, 2 * N))

    for n, i in enumerate(np.arange(N, 2 * N)):
        D[i, i] = (e * kB * Ti[n]) / m

    A = np.zeros((N, N))

    for i in range(N):
        if i > 0:
            A[i][i - 1] = -k / m
        if i < N - 1:
            A[i][i + 1] = -k / m
        A[i][i] = 2 * k / m
    M = np.zeros((2 * N, 2 * N))
    for n, i in enumerate(np.arange(N, 2 * N)):
        M[n, i] = -1
    for n, i in enumerate(np.arange(N, 2 * N)):
        M[n + N, i] = e
    M[N:, :N] = A
    Q = solve_continuous_lyapunov(M, M @ D - D @ M.T)
    return np.linalg.inv(D + Q) @ M


def two_time_corr(tau, N, Tc, Th, m):
    """outputs two time correlation function in a steady state
    Parameters: tau : time interval
                N : the number of beads
                Tc : cold temperature
                Th : hot temperature
                m : float, optional
                    mass
    return : 2D array-like, of shape (2N, 2N)
        the two time correlation functions of phase variables
        scalar value
        Decaying threshold of memory effects
    """
    U_ss = precision_ss(N, Tc, Th, m=m)
    A = np.zeros((N, N))

    for i in range(N):
        if i > 0:  # 1 ~ N
            A[i][i - 1] = -k / m
        if i < N - 1:
            A[i][i + 1] = -k / m
        A[i][i] = 2 * k / m
    M = np.zeros((2 * N, 2 * N))
    for n, i in enumerate(np.arange(N, 2 * N)):
        M[n, i] = -1
    for n, i in enumerate(np.arange(N, 2 * N)):
        M[n + N, i] = e
    M[N:, :N] = A

    one_time_corr = np.linalg.inv(U_ss)
    # scipy.linalg.expm is required
    result = expm(-tau * M) @ one_time_corr
    eigenvalues, _ = np.linalg.eig(M)
    lowest_eig = min(eigenvalues.real)
    return result, lowest_eig ** (-1)


def sampling(N, num_trjs, Tc, Th, m=1):
    U = precision_ss(N, Tc, Th, m)
    U_inv = np.linalg.inv(U)
    cov = (U_inv + U_inv.T) / 2
    q = np.random.multivariate_normal(np.zeros((2 * N,)), cov, num_trjs)
    return cov, q


def simulation(num_trjs, trj_len, N, Tc, Th, dt, m=1, step=10, seed=0):
    T = np.linspace(Tc, Th, N)  # Temperatures linearly varies.
    Drift = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            Drift[i][i - 1] = k / m
        if i < N - 1:
            Drift[i][i + 1] = k / m
        Drift[i][i] = -2 * k / m

    rfc = np.zeros((N,))
    for i in range(N):
        rfc[i] = np.sqrt(2 * e * kB * T[i] / m)

    np.random.seed(seed)
    pos_trj = np.zeros((num_trjs, trj_len, N))
    vel_trj = np.zeros((num_trjs, trj_len, N))
    cov, q = sampling(N, num_trjs, Tc, Th, m)

    pos = q[:, :N]
    pos_trj[:, 0, :] = pos.copy()
    vel = q[:, N:]
    vel_trj[:, 0, :] = vel.copy()

    dt = dt / step
    dt2 = dt ** 2
    dt23 = dt ** (3 / 2)
    dt2r = np.sqrt(dt)

    prev_vel = vel_trj[:, 0]
    prev_pos = pos_trj[:, 0]

    for i in range(1, trj_len):
        for _ in range(step):
            theta = np.random.normal(0, 1.0, (num_trjs, N))
            xi = np.random.normal(0, 1.0, (num_trjs, N))
            eta = 1 / 2 * xi + 1 / (2 * np.sqrt(3)) * theta

            f_prev = np.einsum("ij,aj->ai", Drift, pos)
            C = dt2 * (f_prev - e * prev_vel) / 2 + rfc * dt23 * eta
            pos = prev_pos + dt * prev_vel + C
            f = np.einsum("ij,aj->ai", Drift, pos)
            vel = (
                prev_vel
                + dt * (f + f_prev) / 2
                - dt * e * prev_vel
                + rfc * dt2r * xi
                - e * C
            )
            prev_vel = vel
            prev_pos = pos

        pos_trj[:, i] = pos
        vel_trj[:, i] = vel

    return pos_trj, vel_trj


def del_shannon_etpy(pos_trj, vel_trj, N, Tc, Th, m=1):
    trj = np.concatenate((pos_trj, vel_trj), axis=2)
    U = precision_ss(N, Tc, Th, m)
    etpy = np.einsum("abi,ij,abj->ab", trj, U, trj) / 2
    return etpy[:, 1:] - etpy[:, :-1]


def del_medium_etpy(pos_trj, vel_trj, N, Tc, Th, m=1):
    Drift = np.zeros((N, N))
    Ti = np.linspace(Tc, Th, N)

    for i in range(N):
        if i > 0:
            Drift[i][i - 1] = k
        if i < N - 1:
            Drift[i][i + 1] = k
        Drift[i][i] = -2 * k

    x_prev = pos_trj[:, :-1, :]
    x_next = pos_trj[:, 1:, :]
    dx = x_next - x_prev
    f = np.einsum("ij,abj->abi", Drift, (x_next + x_prev) / 2)
    dQ1 = f * dx

    v_prev = vel_trj[:, :-1, :]
    v_next = vel_trj[:, 1:, :]
    dv = v_next - v_prev
    mv = m * ((v_next + v_prev) / 2)
    dQ2 = -mv * dv

    dQ = dQ1 + dQ2
    etpy = np.sum(dQ / Ti, axis=2)
    return etpy


def analytic_etpy(N, Tc, Th, m=1):
    Ti = np.linspace(Tc, Th, N)
    U = precision_ss(N, Tc, Th, m)
    U_inv = np.linalg.inv(U)
    cov = (U_inv + U_inv.T) / 2
    return m * e * np.sum([cov[N + i, N + i] / Ti[i] for i in range(N)]) - N * kB * e
