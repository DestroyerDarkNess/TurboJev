"""Image and short-video decisions with SmolVLM2 and compiled TurboJev."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multimodal"))
from app import run

if __name__ == "__main__":
    run("vision")
