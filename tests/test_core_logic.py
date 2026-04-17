import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import imppatez


class TestCoreLogic(unittest.TestCase):
    def test_extract_entries_from_table(self):
        html = """
        <table>
            <thead>
                <tr>
                    <th>IMPPAT Phytochemical identifier</th>
                    <th>Phytochemical name</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>IMPHY0001</td><td>Chem A</td></tr>
                <tr><td>IMPHY0002</td><td>Chem B</td></tr>
            </tbody>
        </table>
        """
        entries = imppatez.extract_entries(html)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["IMPHY_ID"], "IMPHY0001")
        self.assertEqual(entries[1]["Phytochemical_Name"], "Chem B")

    def test_extract_entries_regex_fallback(self):
        html = "<td>IMPHY1234</td><td>Fallback Compound</td>"
        with patch("imppatez.pd.read_html", side_effect=ValueError("no tables")):
            entries = imppatez.extract_entries(html)
        self.assertEqual(entries, [{"IMPHY_ID": "IMPHY1234", "Phytochemical_Name": "Fallback Compound"}])

    def test_extract_smiles(self):
        detail_html = "<div>SMILES: CCO InChI: InChI=1S/...</div>"
        smiles = imppatez.extract_smiles(detail_html)
        self.assertEqual(smiles, "CCO")

    def test_ro5_from_smiles_statuses(self):
        self.assertEqual(imppatez.ro5_from_smiles(""), "No SMILES")
        self.assertEqual(imppatez.ro5_from_smiles("not-a-smiles"), "Invalid SMILES")
        self.assertEqual(imppatez.ro5_from_smiles("CCO"), "Passed")
        self.assertTrue(imppatez.ro5_from_smiles("CCCCCCCCCCCCCCCCCCCC").startswith("Failed"))

    def test_run_sdf_preconditions(self):
        msg, zip_file, missing_file = imppatez.run_sdf_from_previous("", "Lipinski Passed + Failed")
        self.assertIn("Please run Step 1 first.", msg)
        self.assertIsNone(zip_file)
        self.assertIsNone(missing_file)

        msg, zip_file, missing_file = imppatez.run_sdf_from_previous("/tmp/does_not_exist.csv", "Lipinski Passed + Failed")
        self.assertIn("was not found", msg)
        self.assertIsNone(zip_file)
        self.assertIsNone(missing_file)

    def test_run_sdf_csv_column_validation(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "bad.csv")
            pd.DataFrame({"X": ["1"]}).to_csv(csv_path, index=False)
            msg, zip_file, missing_file = imppatez.run_sdf_from_previous(csv_path, "Lipinski Passed + Failed")
            self.assertIn("IMPHY_ID", msg)
            self.assertIsNone(zip_file)
            self.assertIsNone(missing_file)

            csv_path2 = os.path.join(td, "no_ro5.csv")
            pd.DataFrame({"IMPHY_ID": ["IMPHY0001"]}).to_csv(csv_path2, index=False)
            msg, zip_file, missing_file = imppatez.run_sdf_from_previous(csv_path2, "Lipinski Passed Only")
            self.assertIn("RO5_Status", msg)
            self.assertIsNone(zip_file)
            self.assertIsNone(missing_file)

    def test_run_sdf_success_path(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "input.csv")
            pd.DataFrame(
                {
                    "IMPHY_ID": ["IMPHY0001", "BADID", "IMPHY0002"],
                    "RO5_Status": ["Passed", "Passed", "Failed (1 violation)"],
                }
            ).to_csv(csv_path, index=False)

            def fake_download(imphy_id, folder):
                if imphy_id == "IMPHY0001":
                    out = os.path.join(folder, f"{imphy_id}.sdf")
                    with open(out, "w", encoding="utf-8") as f:
                        f.write("fake sdf content")
                    return out
                return None

            with patch("imppatez.OUTPUT_DIR", td), patch("imppatez.timestamp_tag", return_value="20260417_003300"), patch(
                "imppatez.download_sdf", side_effect=fake_download
            ):
                msg, zip_file, missing_file = imppatez.run_sdf_from_previous(csv_path, "Lipinski Passed Only")

            self.assertIn("Total IDs selected:** 2", msg)
            self.assertIn("Downloaded SDF files:** 1", msg)
            self.assertIn("Missing / not available:** 1", msg)
            self.assertTrue(zip_file and os.path.exists(zip_file))
            self.assertTrue(missing_file and os.path.exists(missing_file))


if __name__ == "__main__":
    unittest.main()
