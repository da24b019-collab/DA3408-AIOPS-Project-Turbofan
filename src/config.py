"""
Shared configuration for the whole project.

Keeping column names and the params loader in one place means every stage
(ingest, features, training, the API) uses exactly the same definitions.
If two files each defined the column list, they could silently drift apart.
"""

from pathlib import Path

import yaml

# Project root = the folder that contains this "src" folder.
# Using an absolute path means scripts work no matter where they are run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# The raw files have no header row, so we name the 26 columns ourselves.
# Order comes from the readme that ships with the dataset:
# unit number, time in cycles, 3 operational settings, 21 sensor readings.
ID_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]  # s1 ... s21
RAW_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Physical meaning of each sensor (from Saxena et al., 2008).
# The dashboard will show these names so a non-technical operator
# sees "HPC outlet temperature" instead of a code like "s3".
SENSOR_DESCRIPTIONS = {
    "s1": "Fan inlet temperature (T2)",
    "s2": "LPC outlet temperature (T24)",
    "s3": "HPC outlet temperature (T30)",
    "s4": "LPT outlet temperature (T50)",
    "s5": "Fan inlet pressure (P2)",
    "s6": "Bypass-duct pressure (P15)",
    "s7": "HPC outlet pressure (P30)",
    "s8": "Physical fan speed (Nf)",
    "s9": "Physical core speed (Nc)",
    "s10": "Engine pressure ratio (epr)",
    "s11": "HPC outlet static pressure (Ps30)",
    "s12": "Fuel flow to Ps30 ratio (phi)",
    "s13": "Corrected fan speed (NRf)",
    "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)",
    "s16": "Burner fuel-air ratio (farB)",
    "s17": "Bleed enthalpy (htBleed)",
    "s18": "Demanded fan speed (Nf_dmd)",
    "s19": "Demanded corrected fan speed (PCNfR_dmd)",
    "s20": "HPT coolant bleed (W31)",
    "s21": "LPT coolant bleed (W32)",
}


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read params.yaml and return it as a nested dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
