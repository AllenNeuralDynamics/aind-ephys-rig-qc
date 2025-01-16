"""
Aligns timestamps across multiple data streams
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se
from harp.clock import align_timestamps_to_anchor_points, decode_harp_clock
from matplotlib.figure import Figure
from open_ephys.analysis import Session
from scipy.stats import chisquare


def clean_up_sample_chunks(sample_number):
    """
    Detect discontinuities in sample numbers
    and range of sample numbers in residual chunks

    Parameters
    ----------
    sample_number : np.array
        The sample numbers of the each recording event,
        normally increases by 1

    Returns
    -------
    realign : bool
        Whether the recording can be realigned
    residual_ranges : list
        List of ranges of sample numbers in residual chunks,
        to be removed in alignment
    """
    residual_ranges = []
    sample_intervals = np.diff(sample_number)
    discontinuities = np.where(sample_intervals != 1)[0]
    if len(discontinuities) == 0:
        realign = True
        return realign, residual_ranges

    if len(discontinuities) > 0:
        print(f"Found {len(discontinuities)} discontinuit(ies)")
        if len(discontinuities) >= 3:
            print(
                "Found more than 3 discontinuities."
                + "Please check quality of recording."
            )
            realign = False
        else:
            realign = True
            sample_ranges = []
            discontinuities = np.concatenate(
                (np.array([0]), discontinuities + 1)
            )
            for dis_ind in range(len(discontinuities)):
                if dis_ind == len(discontinuities) - 1:
                    range_curr = sample_number[[discontinuities[dis_ind], -1]]
                else:
                    range_curr = sample_number[
                        [
                            discontinuities[dis_ind],
                            discontinuities[dis_ind + 1] - 1,
                        ]
                    ]
                sample_ranges.append(range_curr)
            sample_ranges = np.array(sample_ranges)
            # detect major chunk of recording
            major_ind = np.argmax(sample_ranges[:, 1] - sample_ranges[:, 0])
            major_min = sample_ranges[major_ind, 0]
            major_max = sample_ranges[major_ind, 1]
            # range of other residual chunks
            residual_ranges = np.delete(sample_ranges, major_ind, axis=0)
            # check if residual samples can be removed.
            # (i.e. sample number does not overlap)
            # if can be removed without affecting major stream
            no_overlaps = np.logical_and(
                (residual_ranges[:, 0] - major_min)
                * (residual_ranges[:, 0] - major_max)
                > 0,
                (residual_ranges[:, 1] - major_min)
                * (residual_ranges[:, 1] - major_max)
                > 0,
            )
            if np.all(no_overlaps):
                print(
                    "Residual chunks can be removed"
                    + " without affecting major chunk"
                )
            else:
                main_range = np.arange(major_min, major_max)
                for res_ind in range(len(residual_ranges)):
                    if no_overlaps[res_ind]:
                        main_range = main_range.delete(
                            main_range,
                            np.where(
                                main_range <= residual_ranges[res_ind, 1]
                                and main_range >= residual_ranges[res_ind, 0]
                            ),
                        )
                overlap_perc = (
                    1 - (len(main_range) / (major_max - major_min))
                ) * 100
                print(
                    "Residual chunks cannot be removed without"
                    + f" affecting major chunk, overlap {overlap_perc}%"
                )

        return realign, residual_ranges


def search_harp_line(recording, directory, pdf=None):
    """
    Search for the Harp clock line in the NIDAQ stream

    Parameters
    ----------
    recording : SpikeInterface recording object
        The recording object to search for the Harp clock line
    directory : str
        The path to the Open Ephys data directory
    pdf : PdfReport
        Report for adding QC figures (optional)

    Returns
    -------
    harp_line : int
        The line number of the Harp clock in the NIDAQ stream
    """

    events = recording.events
    # find the NIDAQ stream
    # for stream_ind, stream in enumerate(recording.continuous):
    #     if "PXIe" in stream.metadata["stream_name"]:
    #         nidaq_stream_ind = stream_ind
    #         break
    # nidaq_stream_name = recording.continuous[nidaq_stream_ind].metadata[
    #     "stream_name"
    # ]
    # nidaq_stream_source_node_id = recording.continuous[
    #     nidaq_stream_ind
    # ].metadata["source_node_id"]
    # # list potential lines to scan on NIDAQ stream
    nidaq_stream_name = [stream_name for stream_name in events['stream_name'].unique().tolist() if 'PXIe' in stream_name][0]
    nidaq_stream_source_node_id = events[events['stream_name']==nidaq_stream_name].processor_id.unique()[0]

    lines_to_scan = events[
        (events.stream_name == nidaq_stream_name)
        & (events.processor_id == nidaq_stream_source_node_id)
        & (events.state == 1)
    ].line.unique()

    # stream_folder_names, _ = se.get_neo_streams("openephys", directory)
    # stream_folder_names = [
    #     stream_folder_name.split("#")[-1]
    #     for stream_folder_name in stream_folder_names
    # ]

    ncols = len(lines_to_scan)
    figure, axs = plt.subplots(
        nrows=2, ncols=ncols, figsize=(12, 5), layout="tight"
    )

    # check if distribution is uniform
    # and plot distribution of inter-event intervals
    # initialize p_value and p_short
    p_value = np.zeros(len(lines_to_scan))
    p_short = np.zeros(len(lines_to_scan))
    bin_size = 100  # bin size in s to count number of events
    for line_ind, curr_line in enumerate(lines_to_scan):
        if ncols == 1:
            ax1 = axs[0]
            ax2 = axs[1]
        else:
            ax1 = axs[0, line_ind]
            ax2 = axs[1, line_ind]
        curr_events = events[
            (events.stream_name == nidaq_stream_name)
            & (events.processor_id == nidaq_stream_source_node_id)
            & (events.state == 1)
            & (events.line == curr_line)
        ].copy()
        bin_size = 100  # bin size in s
        ts = curr_events.timestamp.values
        bins = np.arange(np.min(ts), np.max(ts), bin_size)
        bins_intervals = np.arange(0, 1.5, 0.1)
        event_intervals = np.diff(ts)
        ts = ts[np.where(event_intervals > 0.1)[0] + 1]
        ax1.hist(event_intervals, bins=bins_intervals)
        ax1.set_title(curr_line)
        ax1.set_xlabel("Inter-event interval (s)")
        ax2.hist(ts, bins=bins)
        ax2.set_xlabel("Time in session (s)")

        if line_ind == 0:
            ax1.set_ylabel("Number of events")
            ax2.set_ylabel("Number of events")

        # check if distribution is uniform
        ts_count, _ = np.histogram(ts, bins=bins)
        expected_data = np.full(len(ts_count), np.mean(ts_count))
        chi2_stat, p_value[line_ind] = chisquare(ts_count, expected_data)
        # check if there's inter-event interval < 0.1s
        p_short[line_ind] = np.sum(event_intervals < 0.05) / len(
            event_intervals
        )
        if line_ind == 0:
            ax2.set_title(
                f"p_uniform time {p_value[line_ind]:.2f}"
                + f"short interval perc {p_short[line_ind]:.2f}"
            )
        else:
            ax2.set_title(f"{p_value[line_ind]:.2f}, {p_short[line_ind]:.2f}")

    # pick the line with even distribution overtime
    # and has short inter-event interval
    candidate_lines = lines_to_scan[(p_short > 0.5) & (p_value > 0.95)]
    plt.suptitle(f"Harp line(s) {candidate_lines}")

    if pdf is not None:
        pdf.add_page()
        pdf.set_y(30)
        pdf.embed_figure(figure)
    harp_line = candidate_lines

    # figure.savefig(os.path.join(directory, "harp_line_search.png"))
    return harp_line, nidaq_stream_name, nidaq_stream_source_node_id, figure


def archive_and_replace_original_timestamps(
    directory,
    new_timestamps,
    timestamp_filename="timestamps.npy",
    archive_filename="original_timestamps.npy",
):
    """
    Replace the original timestamps with the synchronized timestamps.

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    new_timestamps : np.array
        The new synchronized timestamps
    timestamp_filename : str
        The name of the file in which the original timestamps are stored
    archive_filename : str
        The name of the file for archiving the original timestamps
    """

    if not os.path.exists(os.path.join(directory, archive_filename)):
        # rename the original timestamps file
        os.rename(
            os.path.join(directory, timestamp_filename),
            os.path.join(directory, archive_filename),
        )
    else:
        print(
            "Original timestamps already archived. Removed current timestamps."
        )
        os.remove(os.path.join(directory, timestamp_filename))

    # save the new timestamps
    np.save(os.path.join(directory, timestamp_filename), new_timestamps)


def align_timestamps(  # noqa
    directory,
    original_timestamp_filename="original_timestamps.npy",
    flip_NIDAQ=False,
    local_sync_line=1,
    main_stream_index=0,
    pdf=None,
):
    """
    Aligns timestamps across multiple Open Ephys data streams

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    original_timestamp_filename : str
        The name of the file for archiving the original timestamps
    local_sync_line : int
        The line number for the local sync signal on each stream
    main_stream_index : int
        The index of the main stream to align to
    pdf : PdfReport
        Report for adding QC figures (optional)
    """

    session = Session(directory, mmap_timestamps=False)
    stream_folder_names, _ = se.get_neo_streams("openephys", directory)
    stream_folder_names = [
        stream_folder_name.split("#")[-1]
        for stream_folder_name in stream_folder_names
    ]

    for recordnode in session.recordnodes:
        curr_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:
            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            events = recording.events
            main_stream = recording.continuous[main_stream_index]

            main_stream_name = main_stream.metadata["stream_name"]

            print("Processing stream: ", main_stream_name)
            main_stream_source_node_id = main_stream.metadata["source_node_id"]
            main_stream_sample_rate = main_stream.metadata["sample_rate"]
            if 'PXIe' in main_stream_name and flip_NIDAQ:
                # flip the NIDAQ stream if sync line is inverted between NIDAQ
                # and main stream
                print('Flipping NIDAQ stream as main stream...')
                main_stream_events = events[
                    (events.stream_name == main_stream_name)
                    & (events.processor_id == main_stream_source_node_id)
                    & (events.line == local_sync_line)
                    & (events.state == 0)
                ]
            else:
                main_stream_events = events[
                    (events.stream_name == main_stream_name)
                    & (events.processor_id == main_stream_source_node_id)
                    & (events.line == local_sync_line)
                    & (events.state == 1)
                ]
            # sort by sample number in case timestamps are not in order
            main_stream_events = main_stream_events.sort_values(
                by="sample_number"
            )

            # detect discontinuities from sample numbers
            # and remove residual chunks to avoid misalignment
            sample_numbers = main_stream.sample_numbers
            main_stream_start_sample = np.min(sample_numbers)
            main_stream_start_sample = np.min(sample_numbers)
            sample_intervals = np.diff(sample_numbers)
            sample_intervals_cat, sample_intervals_counts = np.unique(
                sample_intervals, return_counts=True
            )
            sample_intervals_cat = sample_intervals_cat.astype(str).tolist()
            sample_intervals_counts = sample_intervals_counts / len(
                sample_intervals
            )
            realign, residual_ranges = clean_up_sample_chunks(sample_numbers)
            if not realign:
                print(
                    "Recording cannot be realigned."
                    + "Please check quality of recording."
                )
                continue
            else:
                # remove events in residual chunks
                for res_ind in range(len(residual_ranges)):
                    condition = np.logical_and(
                        (
                            main_stream_events.sample_number
                            >= residual_ranges[res_ind, 0]
                        ),
                        (
                            main_stream_events.sample_number
                            <= residual_ranges[res_ind, 1]
                        ),
                    )
                    main_stream_events = main_stream_events.drop(
                        main_stream_events[condition].index
                    )

                main_stream_times = (
                    main_stream_events.sample_number.values
                    / main_stream_sample_rate
                )
                main_stream_times = (
                    main_stream_times - main_stream_times[0]
                )  # start at 0
                main_stream_event_sample = (
                    main_stream_events.sample_number.values
                )

                # align recording timestamps to main stream
                ts_main = align_timestamps_to_anchor_points(
                    sample_numbers,
                    main_stream_event_sample,
                    main_stream_times,
                )

                print(
                    f"Total events for {main_stream_name}: "
                    + f"{len(main_stream_events)}"
                )
                if pdf is not None:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", size=12)
                    pdf.set_y(30)
                    pdf.write(
                        h=12,
                        text=(
                            "Temporal alignment "
                            f"of Record Node {curr_record_node} - "
                            f"Experiment {current_experiment_index} - "
                            f"Recording {current_recording_index}"
                        ),
                    )
                    fig = Figure(figsize=(10, 10))
                    axes = fig.subplots(nrows=3, ncols=2)

                    axes[0, 0].plot(
                        main_stream.timestamps, label=main_stream_name
                    )
                    axes[0, 1].plot(ts_main, label=main_stream_name)
                    axes[2, 0].bar(
                        sample_intervals_cat, sample_intervals_counts
                    )
                    axes[2, 1].axis("off")

                # save the timestamps for continuous in the main stream
                stream_folder_name = [
                    stream_folder_name
                    for stream_folder_name in stream_folder_names
                    if main_stream_name in stream_folder_name
                ][0]
                print("Updating stream continuous timestamps...")
                archive_and_replace_original_timestamps(
                    os.path.join(
                        recording.directory, "continuous", stream_folder_name
                    ),
                    new_timestamps=ts_main,
                    timestamp_filename="timestamps.npy",
                    archive_filename=original_timestamp_filename,
                )

                # save timestamps for the events in the main stream
                # mapping to original events sample number
                # in case timestamps are not in order
                main_stream_events_folder = os.path.join(
                    recording.directory, "events", stream_folder_name, "TTL"
                )
                sample_filename_events = os.path.join(
                    main_stream_events_folder, "sample_numbers.npy"
                )
                sample_number_raw = np.load(sample_filename_events)
                ts_main_events = align_timestamps_to_anchor_points(
                    sample_number_raw,
                    main_stream_event_sample,
                    main_stream_times,
                )
                print("Updating stream event timestamps...")
                archive_and_replace_original_timestamps(
                    main_stream_events_folder,
                    new_timestamps=ts_main_events,
                    timestamp_filename="timestamps.npy",
                    archive_filename=original_timestamp_filename,
                )
                # archive the original main stream events to recover
                # after removing first or last event
                main_stream_events_archive = main_stream_events.copy()

            for stream_idx, stream in enumerate(recording.continuous):
                if stream_idx != main_stream_index:
                    main_stream_events = main_stream_events_archive.copy()
                    stream_name = stream.metadata["stream_name"]
                    print("Processing stream: ", stream_name)
                    source_node_id = stream.metadata["source_node_id"]
                    sample_rate = stream.metadata["sample_rate"]
                    if 'PXIe' in stream_name and flip_NIDAQ:
                        print('Flipping NIDAQ stream...')
                        # flip the NIDAQ stream if sync line is inverted
                        # between NIDAQ and main stream
                        events_for_stream = events[
                            (events.stream_name == stream_name)
                            & (events.processor_id == source_node_id)
                            & (events.line == local_sync_line)
                            & (events.state == 0)
                        ]
                    else:
                        events_for_stream = events[
                            (events.stream_name == stream_name)
                            & (events.processor_id == source_node_id)
                            & (events.line == local_sync_line)
                            & (events.state == 1)
                        ]

                    # sort by sample number in case timestamps are not in order
                    events_for_stream = events_for_stream.sort_values(
                        by="sample_number"
                    )
                    # remove sync events in residual chunks
                    sample_numbers = stream.sample_numbers
                    sample_intervals = np.diff(sample_numbers)
                    sample_intervals_cat, sample_intervals_counts = np.unique(
                        sample_intervals, return_counts=True
                    )
                    sample_intervals_cat = sample_intervals_cat.astype(
                        str
                    ).tolist()
                    sample_intervals_counts = sample_intervals_counts / len(
                        sample_intervals
                    )
                    realign, residual_ranges = clean_up_sample_chunks(
                        sample_numbers
                    )

                    if not realign:
                        print(
                            "Recording cannot be realigned."
                            + " Please check quality of recording."
                        )
                        continue
                    else:
                        # remove events in residual chunks
                        for res_ind in range(len(residual_ranges)):
                            condition = np.logical_and(
                                (
                                    events_for_stream.sample_number
                                    >= residual_ranges[res_ind, 0]
                                ),
                                (
                                    events_for_stream.sample_number
                                    <= residual_ranges[res_ind, 1]
                                ),
                            )
                            events_for_stream = events_for_stream.drop(
                                events_for_stream[condition].index
                            )

                        # remove inconstant events between main and curr stream

                        if len(main_stream_events) != len(events_for_stream):
                            print(
                                "Number of events in main and current stream"
                                + " are not equal"
                            )
                            first_main_event_ts = (
                                main_stream_events.sample_number.values[0]
                                - main_stream_start_sample
                            ) / main_stream_sample_rate
                            first_curr_event_ts = (
                                events_for_stream.sample_number.values[0]
                                - sample_numbers[0]
                            ) / sample_rate
                            offset = np.abs(
                                first_main_event_ts - first_curr_event_ts
                            )
                            if offset > 0.1:
                                # bigger than 0.1s so that
                                # it should not be the same event
                                print(
                                    "First event in main and current stream"
                                    + " are not aligned. Off by "
                                    + f"{offset:.2f} s"
                                )
                                # remove first event from the stream
                                # with the most events
                                if len(main_stream_events) > len(
                                    events_for_stream
                                ):
                                    print(
                                        "Removing first event in main stream"
                                    )
                                    main_stream_events = main_stream_events[1:]
                                else:
                                    print(
                                        "Removing first event in"
                                        + " current stream"
                                    )
                                    events_for_stream = events_for_stream[1:]
                            else:
                                print(
                                    "First event in main and current stream"
                                    " are aligned. Off by "
                                    f"{offset:.2f} s"
                                )
                                # remove last event from the stream
                                # with the most events
                                if len(main_stream_events) > len(
                                    events_for_stream
                                ):
                                    print("Removing last event in main stream")
                                    main_stream_events = main_stream_events[
                                        :-1
                                    ]
                                else:
                                    print(
                                        "Removing last event in current stream"
                                    )
                                    events_for_stream = events_for_stream[:-1]
                        else:
                            print(
                                "Number of events in main and current stream"
                                + " are equal"
                            )

                        print(
                            f"Total events for {stream_name}: "
                            + f"{len(events_for_stream)}"
                        )

                        if pdf is not None:
                            """Plot original timestamps"""
                            axes[0, 0].plot(
                                stream.timestamps, label=stream_name
                            )
                            axes[1, 0].plot(
                                (
                                    np.diff(events_for_stream.timestamp)
                                    - np.diff(main_stream_events.timestamp)
                                )
                                * 1000,
                                label=(stream_name),
                                linewidth=1,
                            )
                            axes[1, 0].set_ylim([-1.5, 1.5])
                            axes[2, 0].bar(
                                sample_intervals_cat, sample_intervals_counts
                            )

                        assert len(main_stream_events) == len(
                            events_for_stream
                        )

                        local_stream_times = (
                            events_for_stream.sample_number.values
                            / sample_rate
                        )

                        ts = align_timestamps_to_anchor_points(
                            local_stream_times,
                            local_stream_times,
                            main_stream_times,
                        )

                        ts_stream = align_timestamps_to_anchor_points(
                            sample_numbers,
                            events_for_stream.sample_number.values,
                            main_stream_times,
                        )

                        if pdf is not None:
                            """Plot aligned timestamps"""
                            axes[0, 1].plot(ts_stream, label=stream_name)
                            axes[1, 1].plot(
                                (np.diff(ts) - np.diff(main_stream_times))
                                * 1000,
                                label=stream_name,
                                linewidth=1,
                            )
                            axes[1, 1].set_ylim([-1.5, 1.5])

                        # write the new timestamps .npy files
                        stream_folder_name = [
                            stream_folder_name
                            for stream_folder_name in stream_folder_names
                            if stream_name in stream_folder_name
                        ][0]
                        print("Updating stream continuous timestamps...")
                        archive_and_replace_original_timestamps(
                            os.path.join(
                                recording.directory,
                                "continuous",
                                stream_folder_name,
                            ),
                            new_timestamps=ts_stream,
                            timestamp_filename="timestamps.npy",
                            archive_filename=original_timestamp_filename,
                        )

                        if pdf is not None:
                            axes[0, 0].set_title("Original alignment")
                            axes[0, 0].set_xlabel("Sample number")
                            axes[0, 0].set_ylabel("Time (ms)")
                            axes[0, 0].legend(loc="upper left")
                            axes[1, 0].set_xlabel("Sync event number")
                            axes[1, 0].set_ylabel("Time interval (ms)")
                            axes[2, 0].set_xlabel("Sample interval")
                            axes[2, 0].set_ylabel("Percentage")

                            axes[0, 1].set_title("After local alignment")
                            axes[0, 1].set_xlabel("Sample number")
                            axes[0, 1].set_ylabel("Time (ms)")
                            axes[0, 1].legend(loc="upper left")
                            axes[1, 1].set_xlabel("Sync event number")
                            axes[1, 1].set_ylabel("Time diff (ms)")

                            pdf.set_y(40)
                            pdf.embed_figure(fig)

                        # save timestamps for the events in the stream
                        # mapping original events sample number
                        # in case timestamps are not in order
                        stream_events_folder = os.path.join(
                            recording.directory,
                            "events",
                            stream_folder_name,
                            "TTL",
                        )
                        sample_filename_events = os.path.join(
                            stream_events_folder, "sample_numbers.npy"
                        )
                        sample_number_raw = np.load(sample_filename_events)

                        ts_events = align_timestamps_to_anchor_points(
                            sample_number_raw,
                            events_for_stream.sample_number.values,
                            main_stream_times,
                        )
                        print("Updating stream event timestamps...")
                        archive_and_replace_original_timestamps(
                            stream_events_folder,
                            new_timestamps=ts_events,
                            timestamp_filename="timestamps.npy",
                            archive_filename=original_timestamp_filename,
                        )

    fig.savefig(os.path.join(directory, "temporal_alignment.png"))

    return fig


def align_timestamps_harp(
    directory,
    pdf=None,
):
    """
    Aligns timestamps across multiple Open Ephys data streams

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    qc_report : PdfReport
        Report for adding QC figures (optional)
    """

    session = Session(directory, mmap_timestamps=False)
    stream_folder_names, _ = se.get_neo_streams("openephys", directory)
    stream_folder_names = [
        stream_folder_name.split("#")[-1]
        for stream_folder_name in stream_folder_names
    ]

    for recordnode in session.recordnodes:
        curr_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:

            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            events = recording.events

            # detect harp clock line
            harp_line, nidaq_stream_name, source_node_id = search_harp_line(
                recording, directory, pdf
            )
            if len(harp_line) > 1:
                print(f"Multiple Harp lines found. Select from {harp_line}")
                harp_line = int(input("Please select Harp line: "))
                print("Harp line selected: ", harp_line)
            elif len(harp_line) == 0:
                print("No Harp line found. Please check recording.")
                continue
            else:
                harp_line = harp_line[0]
                print("Harp line detected: ", harp_line)

            # align time to harp clock
            events = recording.events
            harp_events = events[
                (events.stream_name == nidaq_stream_name)
                & (events.processor_id == source_node_id)
                & (events.line == harp_line)
            ]

            harp_states = harp_events.state.values
            harp_timestamps_local = harp_events.timestamp.values
            start_times, harp_times = decode_harp_clock(
                harp_timestamps_local, harp_states
            )
            print("Total Harp events: ", len(harp_times))

            if pdf is not None:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", size=12)
                pdf.set_y(30)
                pdf.write(
                    h=12,
                    text=(
                        f"Harp alignment of Record Node {curr_record_node},"
                        f"Experiment {current_experiment_index},"
                        f"Recording {current_recording_index}"
                    ),
                )
            fig = Figure(figsize=(10, 10))
            axes = fig.subplots(nrows=2, ncols=2)

            axes[0, 0].plot(start_times, harp_times)
            axes[0, 1].plot(np.diff(start_times), label="start times")
            axes[0, 1].plot(np.diff(harp_times), label="harp times")
            axes[0, 1].legend(loc="upper left")
            axes[0, 1].set_ylabel("Intervals - 1s (ms)")
            # axes[2,0].bar(sample_intervals_cat, sample_intervals_counts)

            for stream_ind in range(len(recording.continuous)):

                stream_name = recording.continuous[stream_ind].metadata[
                    "stream_name"
                ]
                stream_folder_name = [
                    stream_folder_name
                    for stream_folder_name in stream_folder_names
                    if stream_name in stream_folder_name
                ][0]
                # continuous streams timestamps
                local_stream_times = recording.continuous[
                    stream_ind
                ].timestamps
                harp_aligned_ts = align_timestamps_to_anchor_points(
                    local_stream_times, start_times, harp_times
                )
                # plot harp timestamps vs local timestamps

                axes[1, 0].plot(local_stream_times, label=stream_name)
                axes[1, 1].plot(harp_aligned_ts, label=stream_name)

                archive_and_replace_original_timestamps(
                    os.path.join(
                        recording.directory, "continuous", stream_folder_name
                    ),
                    new_timestamps=harp_aligned_ts,
                    timestamp_filename="timestamps.npy",
                    archive_filename="local_timestamps.npy",
                )

                # events timestamps
                stream_events_times_folder = os.path.join(
                    recording.directory, "events", stream_folder_name, "TTL"
                )
                stream_events_times = np.load(
                    os.path.join(stream_events_times_folder, "timestamps.npy")
                )
                stream_events_harp_aligned_ts = (
                    align_timestamps_to_anchor_points(
                        stream_events_times, start_times, harp_times
                    )
                )

                archive_and_replace_original_timestamps(
                    stream_events_times_folder,
                    new_timestamps=stream_events_harp_aligned_ts,
                    timestamp_filename="timestamps.npy",
                    archive_filename="local_timestamps.npy",
                )

            axes[0, 0].set_title("Harp time vs local time")
            axes[0, 0].set_xlabel("Local time (s)")
            axes[0, 0].set_ylabel("Harp time (s)")
            axes[0, 1].set_title("Time intervals")
            axes[0, 1].legend(loc="upper left")
            axes[1, 0].set_title("Local timestamps (s)")
            axes[1, 0].set_xlabel("Samples")
            axes[1, 1].set_title("Harp timestamps (s)")
            axes[1, 1].set_xlabel("Samples")

            if pdf is not None:
                pdf.set_y(40)
                pdf.embed_figure(fig)

            fig.savefig(os.path.join(directory, "harp_temporal_alignment.png"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Two input arguments are required:")
        print(" 1. A data directory")
        print(" 2. A JSON parameters file")
    else:
        with open(
            sys.argv[2],
            "r",
        ) as f:
            parameters = json.load(f)

        directory = sys.argv[1]

        if not os.path.exists(directory):
            raise ValueError(f"Data directory {directory} does not exist.")

        align_timestamps(directory, **parameters)
