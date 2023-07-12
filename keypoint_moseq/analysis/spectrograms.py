from keypoint_moseq.project.io import load_results
from keypoint_moseq.project.wtgold_utils_SR import fly_nodes

from scipy.stats import zscore
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


use_bodyparts = ['thorax', 'abdomen', 'wingL',
                 'wingR', 'forelegL4', 'forelegR4',
                 'midlegL4', 'midlegR4', 'hindlegL4',
                 'hindlegR4']


def fastWavelet_morlet_convolution(x, f, omega0, dt, useGPU=-1):
    if useGPU>=0:
        import cupy as np
        np.cuda.Device(useGPU).use()
    else:
        import numpy as np
    N = len(x)
    L = len(f)
    amp = np.zeros((L, N))

    if not N // 2:
        x = np.concatenate((x, [0]), axis=0)
        N = len(x)
        wasodd = True
    else:
        wasodd = False

    x = np.concatenate([np.zeros(int(N / 2)), x, np.zeros(int(N / 2))], axis=0)
    M = N
    N = len(x)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)

    if wasodd:
        idx = np.arange((M / 2), (M / 2 + M - 2)).astype(int)
    else:
        idx = np.arange((M / 2), (M / 2 + M)).astype(int)

    for i in range(L):
        m = (np.pi ** (-0.25)) * np.exp(-0.5 * (-Omegavals * scales[i] - omega0) ** 2)
        q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])

        q = q[idx]
        amp[i, :] = np.abs(q) * (np.pi ** -0.25) * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2) / np.sqrt(2 * scales[i])
    return amp


def get_wavelet_freq():

    numPeriods = 25     # %number of wavelet frequencies to use
    minF = 1    # minimum frequency for wavelet transform (Hz)
    maxF = 50   # maximum frequency for wavelet transform (Hz)

    minT = 1.0 / maxF
    maxT = 1.0 / minF
    Ts = minT * (2 ** ((np.arange(numPeriods) * np.log(maxT / minT)) / (np.log(2) * (numPeriods - 1))))
    freq = (1.0 / Ts)[::-1]

    # widths = omega0 * samplingFreq / (2 * freq * np.pi)
    # print("widths", widths)
    return freq


if __name__ == '__main__':

    samplingFreq = 150  # sampling frequency (Hz)
    dt = 1.0 / samplingFreq
    omega0 = 5  # dimensionless Morlet wavelet parameter
    freq = get_wavelet_freq()
    print("freq", freq)

    results_file = '/scratch/gpfs/us3519/fit_pair/project/2023_07_05-02_29_02/sweep_nlags/0/cv0/2023_07_05-04_49_02/results.h5'

    results = load_results(path=results_file)

    sessions = list(results.keys())
    print("sessions", sessions)
    s = sessions[0]
    print("Using session:", s)

    assigned_syllables = results[s]['syllables']
    print(assigned_syllables.shape)
    unique_syll, unique_indices, unique_counts = np.unique(assigned_syllables, return_index=True, return_counts=True)
    print(unique_syll, unique_counts)

    unique_counts = unique_counts/np.sum(unique_counts)
    print("unique_counts", unique_counts)
    unique_syll = unique_syll[unique_counts >= 0.01]
    print("unique_syll", unique_syll)

    coors = results[s]['estimated_coordinates']
    print("session dimensions", coors.shape)

    valid_syll_frames = np.where(np.isin(assigned_syllables, unique_syll))
    coors = coors[valid_syll_frames]
    syll_seq = assigned_syllables[valid_syll_frames]
    print("used session dimensions", coors.shape, syll_seq.shape)
    print(syll_seq)

    # indices = [i for i in range(1, len(syll_seq)) if syll_seq[i] != syll_seq[i - 1]]
    # syll_seqs = np.split(syll_seq, indices)

    thorax_idx = 0  # use_bodyparts.index('thorax')
    body_part_idx = np.arange(1, 10)    # exclude thorax idx
    for b in body_part_idx:
        b_coors = coors[:, b, :] - coors[:, thorax_idx, :]  # movement relative to thorax
        z_coors = zscore(b_coors)
        print(b, "z_coors", z_coors.shape)

        # cwtmatr_x = signal.cwt(z_coors[:, 0], signal.morlet2, widths)
        cwtmatr_x = fastWavelet_morlet_convolution(z_coors[:, 0], freq, omega0, dt, useGPU=-1)
        print(b, "cwtmatr_x", cwtmatr_x.shape)
        # cwtmatr_y = signal.cwt(z_coors[:, 1], signal.morlet2, widths)
        cwtmatr_y = fastWavelet_morlet_convolution(z_coors[:, 1], freq, omega0, dt, useGPU=-1)
        print(b, "cwtmatr_y", cwtmatr_y.shape)
        if b == 1:
            cwtmatr = np.concatenate([cwtmatr_x, cwtmatr_y], axis=0)
        else:
            cwtmatr = np.concatenate([cwtmatr, cwtmatr_x, cwtmatr_y], axis=0)
        print(cwtmatr_x.shape, cwtmatr.shape)

    # plot each syllable's mean spectrogram
    for syll in unique_syll:
        syll_mean = np.mean(cwtmatr[:, assigned_syllables == syll], axis=1)
        print(syll_mean.shape)

        syll_mean = syll_mean.reshape((len(body_part_idx)*2, -1))
        print(syll_mean.shape)

        plt.figure()
        plt.imshow(syll_mean.real, cmap='viridis', aspect='auto')
        plt.yticks(body_part_idx*2 - 1, use_bodyparts[1:])
        plt.xticks(ticks=[0, 4, 8, 13, 17, 21], labels=[1, 2, 4, 8, 16, 32], fontsize=10)
        plt.xlabel('frequency')
        # plt.xscale('log', base=2)
        plt.title(f'syllable {syll}')
        plt.colorbar()

        plt.savefig(f'me_spec/avgspectr{syll}.png', dpi=300, bbox_inches="tight")
        plt.close()
        break
