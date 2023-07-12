import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import hdbscan
# import umap


h5path = '/tigress/MMURTHY/usingla/gaitdata/X074_10_kalman.h5'

with h5py.File(h5path, "r") as f:
    print(f.keys())
    d = {}
    for k in f:
        d[k] = f[k][()]


SWING_THRESHOLD = 3  # if leg speed > 3 px/frame then it is in swing

leg_idx = np.array([6, 8, 10, 5, 7, 9])
print(leg_idx)

male_trx = d['trx_kalman'][..., 0]
print(male_trx.shape)

print(male_trx[:, leg_idx[0]].shape)


def get_position_change(t):
    return np.linalg.norm(t[1:] - t[:-1], axis=1)


leg_speed = {}
fwd_speed = get_position_change(male_trx[:, 0])
leg_state = [None]*len(leg_idx)
for i, l in enumerate(leg_idx):
    leg_speed[l] = get_position_change(male_trx[:, l])
    leg_state[i] = leg_speed[l] < SWING_THRESHOLD   # stance = 1, swing = 0

print("fwd_speed", fwd_speed.shape)
print(np.mean(fwd_speed), np.max(fwd_speed), np.min(fwd_speed), np.median(fwd_speed))

leg_state = np.array(leg_state)
print("leg_state", leg_state.shape)

n_legs_ground = np.sum(leg_state, axis=0)
print("n_legs_ground", n_legs_ground.shape)

gait = leg_state[:, fwd_speed > 5]
print("gait", gait.shape)

n_legs_ground = np.sum(gait, axis=0)
print(gait)
print(n_legs_ground)

plot_arr = gait

fig, ax = plt.subplots()
i = ax.imshow(plot_arr, cmap='Greys')
fig.colorbar(i)

plt.savefig('leg_state.png')
plt.show()

