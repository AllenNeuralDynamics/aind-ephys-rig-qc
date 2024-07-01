"""Example test template."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch
from matplotlib.figure import Figure
from aind_ephys_rig_qc.generate_report import generate_qc_report
from aind_ephys_rig_qc.qc_figures import plot_drift


test_folder = Path(__file__).parent / "resources" / "ephys_test_data"
test_dataset = "691894_2023-10-04_18-03-13_0.5s"


class TestGenerateReport(unittest.TestCase):
    """Example Test Class"""

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

    def test_drift(self):
        """Check if output is figure."""
        directory = str(test_folder / test_dataset)
        stream_name = "ProbeA-AP"
        fig = plot_drift(directory, stream_name)
        self.assertIsInstance(fig, Figure)
