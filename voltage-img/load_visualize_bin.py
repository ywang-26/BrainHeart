import argparse
import glob
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import RectangleSelector
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd

def bandpass_filter(signal, fs=200, lowcut=6, highcut=20, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=order, Wn=[low, high], btype='band')
    return filtfilt(b, a, signal)


def estimate_hrv(signal, fs=200, height_threshold=0.1, apply_filter=True):
    # Apply bandpass filter if requested
    if apply_filter:
        signal = bandpass_filter(signal, fs)

    # Find peaks (beats)
    peaks, _ = find_peaks(signal, height=height_threshold * np.max(signal),
                          distance=int(0.05 * fs))  # min 600ms between beats

    # Calculate R-R intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs

    # HRV metrics
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # RMSSD
    sdnn = np.std(rr_intervals)  # SDNN
    mean_hr = 60 / np.mean(rr_intervals)  # Mean heart rate

    return {
        'rr_intervals': rr_intervals,
        'rmssd': rmssd,
        'sdnn': sdnn,
        'mean_hr': mean_hr,
        'peaks': peaks
    }

def load_img(file_path: str) -> np.ndarray:
    data = np.fromfile(file_path, dtype=np.float32)
    x_pixels = 50 # hard coded for now
    y_pixels = 50 # hard coded for now
    num_frames = data.shape[0] // (x_pixels * y_pixels)
    img_array = data.reshape(50, 50, num_frames, )

    return img_array


class ROISelector:
    def __init__(self, img_array):
        self.img_array = img_array
        # zscore the mean frame
        self.mean_frame = stats.zscore(np.mean(img_array, axis=2), axis=None)
        self.rois = []
        self.current_roi = None

    def interactive_selection(self):
        """Interactive ROI selection using matplotlib"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.mean_frame, cmap="RdBu", clim=(-1, 1))
        self.ax.set_title('Click and drag to select ROI (press Enter when done)')

        # Rectangle selector
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        plt.show()
        return self.fig

    def on_select(self, eclick, erelease):
        """Callback for rectangle selection"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Ensure proper ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        self.current_roi = (y_min, y_max, x_min, x_max)  # (y1, y2, x1, x2)
        self.add_current_roi()
        print(f"Selected and added ROI: y={y_min}:{y_max}, x={x_min}:{x_max}")

    def add_roi(self, roi_coords, name=None, verbose=False):
        """Add ROI manually with coordinates (y1, y2, x1, x2)"""
        if name is None:
            name = f"ROI_{len(self.rois) + 1}"
        self.rois.append({'coords': roi_coords, 'name': name})
        if verbose:
            print(f"Added {name}: {roi_coords}")

    def add_current_roi(self, name=None):
        """Add the currently selected ROI"""
        if self.current_roi is not None:
            self.add_roi(self.current_roi, name)
            self.current_roi = None
        else:
            print("No ROI currently selected")

    def extract_roi_traces(self):
        """Extract time traces for all ROIs"""
        traces = {}
        for roi in self.rois:
            coords = roi['coords']
            y1, y2, x1, x2 = coords
            trace = np.mean(self.img_array[y1:y2, x1:x2, :], axis=(0, 1))
            traces[roi['name']] = trace
        return traces

    def visualize_rois(self,  show_fig: bool = True, save_fig: bool = True, output_path: str = None):
        """Visualize all selected ROIs"""
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(self.mean_frame, cmap="RdBu", clim=(-1, 1))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.rois)))
        for i, roi in enumerate(self.rois):
            y1, y2, x1, x2 = roi['coords']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[i],
                facecolor='none', label=roi['name']
            )
            ax.add_patch(rect)

        ax.legend()
        ax.set_title('Selected ROIs')

        if save_fig:
            save_fig_path = os.path.join(output_path, f"selected_rois.png")
            plt.savefig(save_fig_path)
            print(f"Saved selected ROIs figure to {save_fig_path}")

        if show_fig:
            plt.show()

def visualize_traces(traces: dict, sample_rate: float = 200, show_fig: bool = True, save_fig: bool = False, output_path: str = None):
    """Visualize time traces for all ROIs"""
    # Plot all ROI traces
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 3 * len(traces)), sharex=True)
    if len(traces) == 1:
        axes = [axes]

    time = np.arange(len(list(traces.values())[0])) / sample_rate  # Default 200 Hz sampling

    for i, (name, trace) in enumerate(traces.items()):
        axes[i].plot(time, trace, label=name)
        axes[i].set_ylabel('dF/F')
        axes[i].set_title(f'{name} Time Trace')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_fig:
        save_fig_path = os.path.join(output_path, f"roi_traces.png")
        plt.savefig(save_fig_path)
        print(f"Saved ROI traces figure to {save_fig_path}")

    if show_fig:
        plt.show()

def hrv_analysis(traces: dict, verbose: bool=False) -> dict:
    """Perform HRV analysis on all ROI traces"""
    results = {}

    for name, trace in traces.items():
        results[name] = estimate_hrv(trace, fs=200)

        if verbose:
            print(f"\n{name} HRV Analysis:")
            print(f"  Mean HR: {results[name]['mean_hr']:.1f} BPM")
            print(f"  RMSSD: {results[name]['rmssd']:.4f} s")
            print(f"  SDNN: {results[name]['sdnn']:.4f} s")
            print(f"  Number of beats detected: {len(results[name]['peaks'])}")

    return results

def visualize_traces_hrv(traces, hrv_results, show_fig: bool = True, save_fig: bool = False, output_path: str = None):
    # Plot traces with detected peaks for each ROI
    fig, axes = plt.subplots(len(traces), 1, figsize=(15, 4 * len(traces)), sharex=True)
    if len(traces) == 1:
        axes = [axes]

    time = np.arange(len(list(traces.values())[0])) / 200

    for i, (name, trace) in enumerate(traces.items()):
        # Apply the same filtering as in HRV analysis
        filtered_trace = bandpass_filter(trace, fs=200)

        # Plot original and filtered traces
        axes[i].plot(time, trace, 'lightblue', alpha=0.7, label='Original')
        axes[i].plot(time, filtered_trace, 'blue', label='Filtered (6-20 Hz)')

        # Plot detected peaks
        peaks = hrv_results[name]['peaks']
        axes[i].plot(time[peaks], filtered_trace[peaks], 'ro', markersize=4, label=f'Beats ({len(peaks)})')

        axes[i].set_ylabel('dF/F')
        axes[i].set_title(
            f'{name} - HR: {hrv_results[name]["mean_hr"]:.1f} BPM, RMSSD: {hrv_results[name]["rmssd"]:.4f}s')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_fig:
        save_fig_path = os.path.join(output_path, f"roi_traces_hrv.png")
        plt.savefig(save_fig_path)
        print(f"Saved ROI traces with HRV figure to {save_fig_path}")

    if show_fig:
        plt.show()

def visualize_rr_intervals(traces, hrv_results, show_fig: bool = True, save_fig: bool = False, output_path: str = None):
    # Plot R-R intervals for each ROI
    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 4))
    if len(traces) == 1:
        axes = [axes]

    for i, (name, _) in enumerate(traces.items()):
        rr_intervals = hrv_results[name]['rr_intervals']

        # Plot R-R intervals over time
        axes[i].plot(rr_intervals, 'o-', markersize=3)
        axes[i].set_ylabel('R-R Interval (s)')
        axes[i].set_xlabel('Beat Number')
        axes[i].set_title(f'{name}\nMean: {np.mean(rr_intervals):.3f}s')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        save_fig_path = os.path.join(output_path, f"rr_intervals.png")
        plt.savefig(save_fig_path)
        print(f"Saved R-R intervals figure to {save_fig_path}")

    if show_fig:
        plt.show()

def create_summary_table(roi_selector, hrv_results):
    """Create a summary table with HRV analysis results for each ROI."""
    summary_data = []
    for name, results in hrv_results.items():
        summary_data.append({
            'ROI': name,
            'Mean HR (BPM)': f"{results['mean_hr']:.1f}",
            'RMSSD (s)': f"{results['rmssd']:.4f}",
            'SDNN (s)': f"{results['sdnn']:.4f}",
            'Beats Detected': len(results['peaks']),
            'ROI Coords': roi_selector.rois[[r['name'] for r in roi_selector.rois].index(name)]['coords'] if name in [
                r['name'] for r in roi_selector.rois] else 'Manual'
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nHRV Analysis Summary:")
    print(summary_df.to_string(index=False))

def visualize(local_path: str, show_fig: bool=True, save_figs: bool=True, output_path: str=None, verbose: bool=False):

    if os.path.isdir(local_path):
        local_files = glob.glob(os.path.join(local_path, "*.bin"))
    elif os.path.isfile(local_path):
        local_files = [local_path]
    else:
        raise ValueError(f"Invalid path: {local_path}. file_path should be a file or directory.")

    for local_file in local_files:
        name_prefix = os.path.splitext(os.path.basename(local_file))[0]

        if save_figs:
            if output_path is None:
                save_path = os.path.dirname(local_file) # Use dir where data is
            else:
                save_path = output_path

            save_path = os.path.join(save_path, name_prefix)
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = None

        img_array = load_img(local_file)

        roi_selector = ROISelector(img_array)
        # roi_selector.interactive_selection() # Uncomment to select ROIs manually

        roi_selector.add_roi((7, 10, 13, 17), "ROI_1", verbose=verbose)
        roi_selector.add_roi((17, 21, 6, 10), "ROI_2", verbose=verbose)
        roi_selector.add_roi((34, 38, 12, 15), "ROI_3", verbose=verbose)
        roi_selector.add_roi((15, 23, 23, 26), "ROI_4", verbose=verbose)
        roi_selector.add_roi((7, 10, 32, 35), "ROI_5", verbose=verbose)
        roi_selector.add_roi((17, 21, 39, 43), "ROI_6", verbose=verbose)
        roi_selector.add_roi((35, 39, 34, 38), "ROI_7", verbose=verbose)

        roi_selector.visualize_rois(show_fig=show_fig, save_fig=save_figs, output_path=save_path)

        # Extract traces for all ROIs
        roi_traces = roi_selector.extract_roi_traces()

        if verbose:
            print(f"Extracted traces for {len(roi_traces)} ROIs:")
            for name, trace in roi_traces.items():
                print(f"  {name}: {len(trace)} time points")

        # Visualize traces
        visualize_traces(roi_traces, sample_rate=200, show_fig=show_fig, save_fig=save_figs, output_path=save_path)

        # Perform HRV analysis on all ROI traces
        hrv_results = hrv_analysis(roi_traces, verbose=verbose)

        # Visualize results
        visualize_traces_hrv(roi_traces, hrv_results, show_fig=show_fig, save_fig=save_figs, output_path=save_path)
        visualize_rr_intervals(roi_traces, hrv_results,  show_fig=show_fig, save_fig=save_figs, output_path=save_path)

        # Print summary output
        create_summary_table(roi_selector, hrv_results)

if __name__ == "__main__":
    # Example command line
    # Example 1 -- run visualization and save figures to the desired path
    # python voltage-img/load_visualize_bin.py --file_path <LOCAL_FILE_PATH>
    #                                          --output_path <DEST_PATH>
    #                                          --save_figs
    # Example 2 -- run visualization and visualize figures without saving. Print out everything if --verbose flag is present
    # python voltage-img/load_visualize_bin.py --file_path <LOCAL_FILE_PATH>
    #                                          --show_figs
    #                                          --verbose

    parser = argparse.ArgumentParser(description='Load and visualize binary data. ')
    # Required arument
    parser.add_argument('--file_path', type=str, required=True, help='Path to local file or directory.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save figures.')
    parser.add_argument('--save_figs', action='store_true', help='Save figures to disk.')
    parser.add_argument('--show_figs', action='store_true', help='Show figures.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose messages.')
    args = parser.parse_args()

    visualize(local_path = args.file_path,
              save_figs=args.save_figs,
              output_path=args.output_path,
              show_fig=args.show_figs,
              verbose=args.verbose)
