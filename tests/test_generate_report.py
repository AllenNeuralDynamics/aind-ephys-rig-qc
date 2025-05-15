"""Example test template."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure

from aind_ephys_rig_qc.generate_report import generate_qc_report
from aind_ephys_rig_qc.qc_figures import plot_drift
from aind_ephys_rig_qc.temporal_alignment import (
    align_timestamps,
    align_timestamps_harp,
)

test_folder = Path(__file__).parent / "resources" / "ephys_test_data"
test_dataset = "691894_2023-10-04_18-03-13_0.5s"


class TestGenerateReport(unittest.TestCase):
    """Generate Report Test Class"""

    @classmethod
    def setUpClass(cls):
        """Remove all generated timestamps files."""
        _clean_up_test_dir(test_folder / test_dataset)

    @classmethod
    def tearDownClass(cls):
        """Remove all generated timestamps files."""
        _clean_up_test_dir(test_folder / test_dataset)

    @patch("builtins.input", return_value="y")
    def test_generate_report_overwriting(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(directory, report_name, plot_drift_map=False)
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    @patch("builtins.input", return_value="n")
    def test_generate_report_not_overwriting(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(directory, report_name, plot_drift_map=False)
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    @patch("builtins.input", return_value="y")
    def test_generate_report_harp(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(
            directory,
            report_name,
            timestamp_alignment_method="harp",
            plot_drift_map=False,
        )
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    @patch("builtins.input", return_value="y")
    def test_generate_report_subsample(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(
            directory, report_name, plot_drift_map=False, subsample_plots=10
        )
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    @patch("builtins.input", return_value="n")
    def test_generate_report_harp_not_overwriting(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(
            directory,
            report_name,
            timestamp_alignment_method="harp",
            plot_drift_map=False,
        )
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))

    @patch("builtins.input", return_value="n")
    def test_generate_report_num_chunks(self, mock_input):
        """Check if output is pdf."""
        directory = str(test_folder / test_dataset)
        report_name = "qc.pdf"
        generate_qc_report(directory, report_name, num_chunks=1)
        self.assertTrue(os.path.exists(os.path.join(directory, report_name)))


class TestPlotDrift(unittest.TestCase):
    """Plot Drift Test Class"""

    def test_drift(self):
        """Check if output is figure."""
        directory = str(test_folder / test_dataset)
        stream_name = "ProbeA-AP"
        fig = plot_drift(directory, stream_name)
        self.assertIsInstance(fig, Figure)


class TestTimestampsAlignment(unittest.TestCase):
    """Timestamps Alignment Test Class"""

    @classmethod
    def setUpClass(cls):
        """Remove all generated timestamps files."""
        _clean_up_test_dir(test_folder / test_dataset)

    @classmethod
    def tearDownClass(cls):
        """Remove all generated timestamps files."""
        _clean_up_test_dir(test_folder / test_dataset)

    def setUp(self):
        """Remove all generated timestamps files."""
        _clean_up_test_dir(test_folder / test_dataset)

    def test_timestamps_alignment(self):
        """Check if output is figure."""
        directory = test_folder / test_dataset
        align_timestamps(directory)
        original_timestamps = [
            p for p in directory.glob("**/*original_timestamps.npy")
        ]
        assert len(original_timestamps) > 0

    @unittest.skip("The test dataset has not HARP times")
    def test_timestamps_alignment_harp(self):  # pragma: no cover
        """Check if output is figure."""
        directory = test_folder / test_dataset
        align_timestamps(directory)
        align_timestamps_harp(directory)
        original_timestamps = [
            p for p in directory.glob("**/*original_timestamps.npy")
        ]
        assert len(original_timestamps) > 0
        localsync_timestamps = [
            p for p in directory.glob("**/*localsync_timestamps.npy")
        ]
        assert len(localsync_timestamps) > 0

    def test_timestamps_alignment_no_plots(self):
        """Check if output is figure."""
        directory = test_folder / test_dataset
        align_timestamps(directory, make_plots=False)
        pngs = [p for p in directory.glob("**/*.png")]
        assert len(pngs) == 0

    def test_subsampling(self):
        """Check if output is figure."""
        directory = test_folder / test_dataset
        align_timestamps(directory)
        pngs_full_sizes = [
            p.stat().st_size for p in directory.glob("**/*.png")
        ]
        _clean_up_test_dir(directory)
        align_timestamps(directory, subsample_plots=5000)
        pngs_sub_sizes = [p.stat().st_size for p in directory.glob("**/*.png")]
        assert all(
            [sub < full for sub, full in zip(pngs_sub_sizes, pngs_full_sizes)]
        )


def _clean_up_test_dir(directory):
    """Remove all generated timestamps files."""
    directory = Path(directory)
    original_timestamps = [
        p for p in directory.glob("**/*original_timestamps.npy")
    ]

    for p in original_timestamps:
        timestamps_file = p.parent / "timestamps.npy"
        localsync_file = p.parent / "localsync_timestamps.npy"
        if timestamps_file.exists():
            timestamps_file.unlink()
        if localsync_file.exists():
            localsync_file.unlink()  # pragma: no cover
        p.rename(timestamps_file)

    # remove png and pdf files
    for p in directory.glob("*.png"):
        p.unlink()
    for p in directory.glob("*.pdf"):
        p.unlink()
