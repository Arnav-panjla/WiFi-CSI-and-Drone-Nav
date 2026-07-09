"""CSI-inertial fusion for a moving drone (proof of concept).

The static-fingerprinting experiment in `csi_localization.py` shows CSI alone
hits a wall once the receiver moves. This script demonstrates the architecture
we'd actually deploy instead:

    drone-as-transmitter  ->  fixed off-board anchors at known positions
    give CSI-based range (ToF) fixes  ->  fused with the onboard IMU in an
    EKF, starting from a known point.

Nobody sensor is enough on its own:
  * IMU        -> smooth, high-rate, but dead-reckoning drifts without bound
  * CSI ranges -> absolute (anchored) but noisy, and the noise GROWS with the
                  drone's speed (Doppler / vibration / tilt -- the exact effect
                  the README argues corrupts CSI under motion).
The EKF lets the CSI fixes bound the IMU drift while the IMU carries the state
smoothly between the sparse, noisy CSI updates.

We compare three estimators against ground truth:
    IMU-only  (dead reckoning)
    CSI-only  (per-epoch multilateration of the anchor ranges)
    Fused     (EKF: IMU prediction + CSI range update)

It is a simulation -- there is no real drone here -- but the noise models are
chosen to reflect the failure modes discussed in the README, and every place a
real CSI/IMU stream would plug in is marked.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results"
RNG = np.random.RandomState(0)

# ---- Scenario: a 10 x 10 m warehouse bay ----
ROOM = 10.0
DT = 0.02                      # 50 Hz IMU
T_END = 40.0                   # seconds of flight
CSI_EVERY = 25                 # CSI fix every 25 IMU steps -> 2 Hz

# Fixed anchors at known positions (the "off-board" infrastructure). The drone
# transmits; these receive and estimate range to it. Real deployment: surveyed
# AP coordinates.
ANCHORS = np.array([
    [0.0, 0.0],
    [ROOM, 0.0],
    [0.0, ROOM],
    [ROOM, ROOM],
    [ROOM / 2, ROOM],
])

# ---- IMU error model (cheap MEMS-grade) ----
ACCEL_BIAS = np.array([0.06, -0.05])   # m/s^2 constant bias -> quadratic drift
ACCEL_NOISE = 0.05                      # m/s^2 white noise (std)

# ---- CSI range error model ----
CSI_BASE_STD = 0.25            # m, ranging noise at rest
CSI_SPEED_STD = 0.6            # m per (m/s) -- noise grows with drone speed.
                              # This is the README's thesis made quantitative:
                              # motion (Doppler/vibration/tilt) corrupts CSI.


def ground_truth():
    """A smooth Lissajous flight path -> true position, velocity, acceleration.

    Returns arrays of shape (steps, 2). Having the closed form lets us hand the
    IMU the true acceleration (which it then corrupts) with no integration bias
    of our own.
    """
    t = np.arange(0.0, T_END, DT)
    cx, cy = ROOM / 2, ROOM / 2
    ax, ay = 3.5, 3.0          # path amplitudes (m)
    wx, wy = 2 * np.pi / 20.0, 2 * np.pi / 13.0   # angular rates (rad/s)
    px = cx + ax * np.sin(wx * t)
    py = cy + ay * np.sin(wy * t)
    vx = ax * wx * np.cos(wx * t)
    vy = ay * wy * np.cos(wy * t)
    accx = -ax * wx * wx * np.sin(wx * t)
    accy = -ay * wy * wy * np.sin(wy * t)
    pos = np.column_stack([px, py])
    vel = np.column_stack([vx, vy])
    acc = np.column_stack([accx, accy])
    return t, pos, vel, acc


def imu_only(pos0, vel0, acc_true):
    """Dead reckoning: integrate the *measured* (biased, noisy) acceleration.

    Real drone: this is the accelerometer stream after gravity removal in the
    world frame (attitude from the gyro/AHRS). Here we corrupt the true accel.
    """
    p = pos0.copy()
    v = vel0.copy()
    traj = np.empty_like(acc_true)
    for k in range(len(acc_true)):
        a_meas = acc_true[k] + ACCEL_BIAS + ACCEL_NOISE * RNG.randn(2)
        p = p + v * DT + 0.5 * a_meas * DT * DT
        v = v + a_meas * DT
        traj[k] = p
    return traj


def csi_ranges(pos_true, vel_true):
    """Per-step anchor range measurements from CSI (ToF), noise scaled by speed.

    Returns (ranges, std) where std is the per-epoch noise used both to corrupt
    the measurement and to weight it in the EKF. Real drone: replace with ToF /
    FTM ranging or AoA+range from each anchor's CSI.
    """
    speed = np.linalg.norm(vel_true, axis=1)
    std = CSI_BASE_STD + CSI_SPEED_STD * speed          # (steps,)
    true = np.linalg.norm(pos_true[:, None, :] - ANCHORS[None, :, :], axis=2)
    meas = true + std[:, None] * RNG.randn(*true.shape)
    return meas, std


def multilaterate(ranges, guess):
    """Gauss-Newton position fix from anchor ranges (the 'CSI-only' estimate)."""
    p = guess.copy()
    for _ in range(10):
        d = p - ANCHORS
        r = np.linalg.norm(d, axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        resid = r - ranges
        J = d / r[:, None]
        step, *_ = np.linalg.lstsq(J, resid, rcond=None)
        p = p - step
        if np.linalg.norm(step) < 1e-6:
            break
    return p


def ekf(pos0, vel0, acc_true, ranges, std):
    """Tightly-coupled EKF: IMU acceleration drives predict, CSI ranges update.

    State x = [px, py, vx, vy]. IMU enters as the control input in the predict
    step (so its bias is what the CSI updates must fight). CSI enters as a
    nonlinear range measurement per anchor in the update step.
    """
    x = np.array([pos0[0], pos0[1], vel0[0], vel0[1]], float)
    P = np.diag([0.1, 0.1, 0.1, 0.1])          # confident known start
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], float)
    B = np.array([[0.5 * DT * DT, 0],
                  [0, 0.5 * DT * DT],
                  [DT, 0],
                  [0, DT]], float)
    # Process noise: trust the IMU short-term, but leave room for its bias.
    q = 0.08
    Q = B @ (q * np.eye(2)) @ B.T + 1e-4 * np.eye(4)
    traj = np.empty((len(acc_true), 2))
    for k in range(len(acc_true)):
        # --- predict with the (corrupted) IMU acceleration ---
        a_meas = acc_true[k] + ACCEL_BIAS + ACCEL_NOISE * RNG.randn(2)
        x = F @ x + B @ a_meas
        P = F @ P @ F.T + Q
        # --- update with CSI ranges when an anchor fix is available ---
        if k % CSI_EVERY == 0:
            p = x[:2]
            d = p - ANCHORS
            r = np.linalg.norm(d, axis=1)
            r = np.where(r < 1e-6, 1e-6, r)
            H = np.zeros((len(ANCHORS), 4))
            H[:, 0] = d[:, 0] / r
            H[:, 1] = d[:, 1] / r
            R = (std[k] ** 2) * np.eye(len(ANCHORS))
            y = ranges[k] - r                     # innovation
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        traj[k] = x[:2]
    return traj


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t, pos, vel, acc = ground_truth()

    imu = imu_only(pos[0], vel[0], acc)
    ranges, std = csi_ranges(pos, vel)
    csi = np.array([multilaterate(ranges[k], pos[0]) for k in range(len(pos))])
    fused = ekf(pos[0], vel[0], acc, ranges, std)

    r_imu, r_csi, r_fused = rmse(imu, pos), rmse(csi, pos), rmse(fused, pos)

    # ---- trajectory plot ----
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(pos[:, 0], pos[:, 1], "k-", lw=2, label="ground truth")
    ax.plot(imu[:, 0], imu[:, 1], "C3-", lw=1, alpha=0.8, label=f"IMU only (RMSE {r_imu:.2f} m)")
    ax.scatter(csi[::5, 0], csi[::5, 1], s=6, c="C0", alpha=0.4, label=f"CSI only (RMSE {r_csi:.2f} m)")
    ax.plot(fused[:, 0], fused[:, 1], "C2-", lw=1.6, label=f"Fused EKF (RMSE {r_fused:.2f} m)")
    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], marker="^", s=90, c="k", label="anchors")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("CSI-inertial fusion vs. single sensors")
    # Zoom to the room: the IMU trail drifts far off-frame (see the error plot);
    # keeping the room in view is what makes truth/CSI/fused distinguishable.
    ax.set_xlim(-2, ROOM + 2); ax.set_ylim(-2, ROOM + 2)
    ax.annotate("IMU drifts off-frame", xy=(ROOM - 1.5, -1.4), fontsize=7,
                color="C3", ha="right")
    ax.legend(fontsize=7, loc="upper left"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fusion_trajectory.png"), dpi=120)
    plt.close(fig)

    # ---- error-over-time plot: IMU drifts, fusion stays bounded ----
    e_imu = np.linalg.norm(imu - pos, axis=1)
    e_csi = np.linalg.norm(csi - pos, axis=1)
    e_fused = np.linalg.norm(fused - pos, axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, e_imu, "C3", lw=1, label="IMU only")
    ax.plot(t, e_csi, "C0", lw=0.8, alpha=0.6, label="CSI only")
    ax.plot(t, e_fused, "C2", lw=1.4, label="Fused EKF")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("Error over time"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fusion_error.png"), dpi=120)
    plt.close(fig)

    lines = [
        "============ CSI-INERTIAL FUSION ============",
        f"trajectory: {T_END:.0f}s @ {1/DT:.0f}Hz IMU, CSI fix @ {1/(DT*CSI_EVERY):.0f}Hz, {len(ANCHORS)} anchors",
        f"IMU error model : bias={ACCEL_BIAS.tolist()} m/s^2, noise={ACCEL_NOISE} m/s^2",
        f"CSI error model : {CSI_BASE_STD} m + {CSI_SPEED_STD} m per (m/s) of speed",
        "",
        f"  IMU only  (dead reckoning)     RMSE = {r_imu:6.2f} m   final = {e_imu[-1]:6.2f} m",
        f"  CSI only  (multilateration)    RMSE = {r_csi:6.2f} m",
        f"  Fused     (EKF)                RMSE = {r_fused:6.2f} m",
        "",
        f"fusion improves on best single sensor by {(1 - r_fused / min(r_imu, r_csi)) * 100:.0f}%",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "fusion_results.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
