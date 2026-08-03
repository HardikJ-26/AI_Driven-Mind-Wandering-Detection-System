"""Run the timestamp-windowed Exp4 pipeline with the multimodal LSTM model.

The shared loader, timestamp-derived 10-second windows, recording-level split,
and reporting remain in CrossValidation_80_20_Corrected.py.  This entry point
selects the LSTM architecture without duplicating that data logic.
"""

import os
import runpy
from pathlib import Path

os.environ["MODEL_TYPE"] = "lstm"
runpy.run_path(Path(__file__).with_name("CrossValidation_80_20_Corrected.py"), run_name="__main__")
