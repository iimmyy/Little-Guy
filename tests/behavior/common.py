"""Paths and the persona loader shared by the Mac-side tests."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get("LITTLEGUY_RUN", ROOT / ".run"))  # big local files (weights, logs, results), not in git
SERVER = Path(os.environ.get("LITTLEGUY_SERVER", RUN / "lcpp-src/build-mac/bin/llama-server"))  # the cloud job uses its own build
BRAINS = RUN / "weights/brains"
EXPERIMENTS = RUN / "weights/experiments"
DRAFTER = RUN / "weights/drafter"
LOGS = RUN / "logs"
RESULTS = RUN / "results/behavior"


def _settings():
    """littleguy.env at the repo root (copy littleguy.env.example); environment variables win."""
    values = {}
    env_file = ROOT / "littleguy.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    values.update({k: v for k, v in os.environ.items() if k.startswith("LITTLEGUY_")})
    return values


BOARD_SSH = _settings().get("LITTLEGUY_BOARD_SSH", "")  # ssh host for the board; empty means no board


def persona_path(name=None):
    """persona/<name>; by default your private persona if you have one, else the public example."""
    name = name or os.environ.get("PERSONA")
    if name:
        path = Path(name)
        return path if path.is_absolute() else ROOT / "persona" / name
    private = ROOT / "persona/little_guy.private.txt"
    return private if private.exists() else ROOT / "persona/little_guy.example.txt"


def load_persona(name=None):
    """The persona text Little Guy reads, without the # note lines. PERSONA=none: no persona at all (the raw model)."""
    if (name or os.environ.get("PERSONA")) == "none":
        return ""
    return "\n".join(l for l in persona_path(name).read_text().splitlines() if l.strip() and not l.startswith("#"))
