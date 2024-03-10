from pathlib import Path


def getRecentCheckpoint(folder):
    checkpoints = list(Path(folder).rglob("*.ckpt"))
    checkpoints.sort(key=lambda file: file.stat().st_mtime)
    # Get most recent checkpoint
    return checkpoints[-1]
