"""Moved to app/services/scale_fill.py (the app uses it); kept importable from eval/."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.services.scale_fill import *  # noqa: F401,F403,E402
from app.services.scale_fill import scale_fill, hold, window_keys, key_changes  # noqa: F401,E402
