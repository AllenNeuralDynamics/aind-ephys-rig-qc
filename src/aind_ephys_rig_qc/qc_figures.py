"""
Generates figures for checking ephys data quality
"""

from matplotlib.figure import Figure
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import spikeinterface.preprocessing as spre
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.node_pipeline import (
    ExtractDenseWaveforms,
    run_node_pipeline,
)
from spikeinterface.sortingcomponents.peak_detection import (
    DetectPeakLocallyExclusive,
)
from spikeinterface.sortingcomponents.peak_localization import (
    LocalizeCenterOfMass,
)


def plot_raw_data(
    data, start_frames, sample_rate, stream_name, chunk_size=1000
):
    """
    Plot a snippet of raw data as an image

    Parameters
    ----------
    data : np.ndarray
        The data to plot (samples x channels)
    start_frames : list
        The starting frame for each chunk
    sample_rate : float
        The sampling rate of the data
    stream_name : str
        The name of the stream
    chunk_size : int, default: 1000
        The size of each chunk

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """

    fig = Figure(figsize=(10, 4))

    num_chunks = len(start_frames)
    end_frames = start_frames + chunk_size

    axs = fig.subplots(ncols=num_chunks)

    fig.suptitle(f"{stream_name} Raw Data")
    for chunk_ind in range(num_chunks):
        if num_chunks == 1:
            ax = axs
        else:
            ax = axs[chunk_ind]
        start_frame = int(start_frames[chunk_ind])
        end_frame = int(end_frames[chunk_ind])
        ax.imshow(
            data[start_frame:end_frame, :].T,
            aspect="auto",
            cmap="RdBu",
            origin="lower",
        )
        ax.set_title(f"@ {np.round(start_frame / sample_rate, 2)}s")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Channels")

    fig.subplots_adjust(wspace=0.3)
    return fig


def plot_power_spectrum(
    data,
    start_frames,
    stream_name,
    sample_rate,
    chunk_size=10000,
):
    """
    Plot the power spectrum of the data

    Parameters
    ----------
    data : np.ndarray
        The data to plot (samples x channels)
    start_frames : list
        The starting frame for each chunk
    stream_name : str
        The name of the stream
    sample_rate : float
        The sampling rate of the data
    chunk_size : int, default: 10000
        The size of each chunk

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig = Figure(figsize=(10, 4))

    num_chunks = len(start_frames)
    axs = fig.subplots(
        nrows=2,
        ncols=num_chunks,
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )

    fig.suptitle(f"{stream_name} PSD")
    for chunk_ind in range(num_chunks):
        if num_chunks == 1:
            ax1 = axs[0]
            ax2 = axs[1]
        else:
            ax1 = axs[0, chunk_ind]
            ax2 = axs[1, chunk_ind]
        start_frame = int(start_frames[chunk_ind])
        end_frame = start_frame + chunk_size
        subset = data[start_frame:end_frame]
        p_channel = []
        for i in range(subset.shape[1]):
            f, p = welch(subset[:, i], fs=sample_rate)
            ax2.plot(f, p, color="gray", alpha=0.5)
            p_channel.append(p)

        p_channel = np.array(p_channel)
        p_mean = p_channel.mean(axis=0)
        ax2.plot(f, p_mean, color="k", lw=1)

        extent = [f.min(), f.max(), 0, subset.shape[1] - 1]
        ax1.imshow(
            p_channel,
            extent=extent,
            aspect="auto",
            cmap="inferno",
            origin="lower",
        )
        ax1.set_ylabel("Channels")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Power")
    fig.subplots_adjust(wspace=0.3)

    return fig


def plot_drift(diretory, stream_name):
    """
    Plot the drift of the data by spike localization

    Parameters
    ----------
    openephys_folder : str
        The path to the OpenEphys data folder

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    """set parameters for drift visualization"""
    visualization_drift_params = {
        "detection": {
            "peak_sign": "neg",
            "detect_threshold": 5,
            "exclude_sweep_ms": 0.1,
        },
        "localization": {
            "ms_before": 0.1,
            "ms_after": 0.3,
            "radius_um": 100.0,
        },
        "cmr": {"reference": "global", "operator": "median"},
        "highpass_filter": {"freq_min": 300.0, "margin_ms": 5.0},
        "n_skip": 30,
        "alpha": 0.15,
        "vmin": -200,
        "vmax": 0,
        "cmap": "Greys_r",
        "figsize": [10, 10],
    }

    """ get blocks/experiments and streams info """
    si.set_global_job_kwargs(n_jobs=-1)

    """ drift raster"""
    cmap = plt.get_cmap(visualization_drift_params["cmap"])
    norm = Normalize(
        vmin=visualization_drift_params["vmin"],
        vmax=visualization_drift_params["vmax"],
        clip=True,
    )
    n_skip = visualization_drift_params["n_skip"]
    alpha = visualization_drift_params["alpha"]

    stream_names, _ = se.get_neo_streams("openephys", diretory)
    spike_stream = [
        curr_stream_name
        for curr_stream_name in stream_names
        if stream_name in curr_stream_name
    ][0]

    recording = se.read_openephys(
        folder_path=diretory,
        stream_name=spike_stream,
    )
    # high-pass filter
    recording = spre.highpass_filter(
        recording, **visualization_drift_params["highpass_filter"]
    )
    # common reference
    recording = spre.common_reference(
        recording, **visualization_drift_params["cmr"]
    )
    """ Here we use the node pipeline implementation """
    peak_detector_node = DetectPeakLocallyExclusive(
        recording, **visualization_drift_params["detection"]
    )
    extract_dense_waveforms_node = ExtractDenseWaveforms(
        recording,
        ms_before=visualization_drift_params["localization"]["ms_before"],
        ms_after=visualization_drift_params["localization"]["ms_after"],
        parents=[peak_detector_node],
        return_output=False,
    )
    localize_peaks_node = LocalizeCenterOfMass(
        recording,
        radius_um=visualization_drift_params["localization"]["radius_um"],
        parents=[peak_detector_node, extract_dense_waveforms_node],
    )
    pipeline_nodes = [
        peak_detector_node,
        extract_dense_waveforms_node,
        localize_peaks_node,
    ]
    peaks, peak_locations = run_node_pipeline(
        recording, nodes=pipeline_nodes, job_kwargs=si.get_global_job_kwargs()
    )
    peak_amps = peaks["amplitude"]
    y_locs = recording.get_channel_locations()[:, 1]
    ylim = [np.min(y_locs), np.max(y_locs)]

    fig = Figure(figsize=visualization_drift_params["figsize"])
    axs_drift = fig.subplots(
        ncols=recording.get_num_segments(),
    )
    # for testing purposes
    if recording.get_total_duration() < 3:
        visualization_drift_params["n_skip"] = 1
        alpha = 0.5
    for segment_index in range(recording.get_num_segments()):
        segment_mask = peaks["segment_index"] == segment_index
        x = peaks[segment_mask]["sample_index"] / recording.sampling_frequency
        y = peak_locations[segment_mask]["y"]
        # subsample
        x_sub = x[::n_skip]
        y_sub = y[::n_skip]
        a_sub = peak_amps[::n_skip]
        colors = cmap(norm(a_sub))

    if recording.get_num_segments() == 1:
        ax_drift = axs_drift
    else:
        ax_drift = axs_drift[segment_index]  # pragma: no cover
    ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
    ax_drift.set_xlabel("time (s)", fontsize=12)
    ax_drift.set_ylabel("depth ($\\mu$m)", fontsize=12)
    ax_drift.set_xlim(
        0,
        recording.get_num_samples(segment_index=segment_index)
        / recording.sampling_frequency,
    )
    ax_drift.set_ylim(ylim)
    ax_drift.spines["top"].set_visible(False)
    ax_drift.spines["right"].set_visible(False)

    ax_drift.set_title(
        f"Drift map: {stream_name}\n# detected peaks: {len(peaks)}"
    )

    return fig


# def plot_timealign(streams, overwrite=False):
#     """
#     Plot the timealignment of the data

#     Parameters
#     ----------
#     data : streams
#         The recording streams to plot

#     Returns
#     -------
#     matplotlib.figure.Figure
#         Figure object containing the plot
#     """

#     fig = Figure(figsize=(10, 4))
#     ax = fig.subplots(1, 2)

#     sync_line = 1
#     main_name = "ProbeA-AP"

#     stream_time = []
#     stream_names = []
#     """extract time of data streams"""
#     for stream_ind in range(len(streams.continuous)):
#         stream_time.append(streams.continuous[stream_ind].timestamps)
#         stream_names.append(
#             streams.continuous[stream_ind].metadata["stream_name"]
#         )

#     """plot time alignment"""
#     for stream_ind in range(len(stream_time)):
#         ax[0].plot(stream_time[stream_ind], label=stream_names[stream_ind])
#     ax[0].legend()
#     ax[0].set_title("Time Alignment_original")
#     ax[0].set_xlabel("Samples")
#     ax[0].set_ylabel("Time (s)")

#     """plot time alignment after alignment"""
#     ignore_after_time = stream_time[0][-1] - np.min(stream_time[0])

#     stream_num = len(streams.continuous)
#     for stream_ind in range(stream_num):
#         stream_name = streams.continuous[stream_ind].metadata["stream_name"]
#         processor_id = streams.continuous[stream_ind].metadata[
#             "source_node_id"
#         ]
#         if stream_name == main_name:
#             main_or_not = True
#         else:
#             main_or_not = False

#         streams.add_sync_line(
#             sync_line,  # TTL line number
#             processor_id,  # processor ID
#             stream_name,  # stream
#             main=main_or_not,  # set as the main stream
#             ignore_intervals=[(ignore_after_time * 30000, np.inf)],
#         )

#     streams.compute_global_timestamps(overwrite=overwrite)
#     """extract time of data streams"""
#     stream_time_align = []
#     for stream_ind in range(len(streams.continuous)):
#         stream_time_align.append(streams.continuous[stream_ind].timestamps)

#     """plot time alignment"""
#     for stream_ind in range(len(stream_time)):
#         ax[1].plot(
#             stream_time_align[stream_ind], label=stream_names[stream_ind]
#         )
#     ax[1].legend()
#     ax[1].set_title("Time Alignment_aligned")
#     ax[1].set_xlabel("Samples")
#     ax[1].set_ylabel("Time (s)")

#     return fig
