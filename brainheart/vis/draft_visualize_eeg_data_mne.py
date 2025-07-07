### Test blog

import matplotlib.pyplot as plt

info = mne.create_info(raw.info['ch_names'][:32], raw.info['sfreq'], ch_types=['eeg'] * 32)

# Create a montage from your positions
del ch_pos["ECG"]
del ch_pos["EOG"]
del ch_pos["RESP"]
# temp solution
value = ch_pos.pop("FP1")
ch_pos["Fp1"] = value
value = ch_pos.pop("FP2")
ch_pos["Fp2"] = value
value = ch_pos.pop("FPz")
ch_pos["Fpz"] = value
montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
info.set_montage(montage)

events = mne.make_fixed_length_events(raw, duration=40.0)  # 2-second epochs
epochs = mne.Epochs(raw, events, tmin=0, tmax=40.0, preload=True, baseline=(0, 0))
epochs_subset = epochs.pick_channels(raw.info['ch_names'][:32])
epochs_subset.set_montage(montage)
evoked = epochs_subset.average()

times = np.arange(0, 40, 0.10)  # Every 50ms for 2 seconds
anim = evoked.animate_topomap(times=times,
                              frame_rate=20,  # 20 fps
                              blit=False,
                              butterfly=False)
plt.show()
anim.save("/Users/yunmiaowang/Documents/GSoC_2025/data/eeglab_eeg_ecg/eeg_topomap_video.map", writer='ffmpeg')

data2plot = data[:32, :].mean(axis=1)
fig, ax = plt.subplots(figsize=(8, 6))
im, cm = mne.viz.plot_topomap(
    data2plot,
    info,
    axes=ax,
    show=False,
    contours=6,
    cmap='RdBu_r'
)

# Add colorbar
plt.colorbar(im, ax=ax, shrink=0.8)
plt.title('EEG Topoplot')
plt.show()