import logging
import os
import sys
from pathlib import Path
import uuid

import fire
from tqdm import tqdm

type StrPath = str | Path


def rename(path: StrPath = "."):
    path = Path(path)
    if not path.is_dir():
        logging.error(f"Path {path} is not a directory. Aborting...")
        return
    ldir = os.listdir(path)
    paths = [path.joinpath(p) for p in ldir]
    for p in paths:
        if p.is_dir():
            logging.warning(f"Path {path} contains directories. Aborting...")
            return
    paths.sort(key=lambda p: p.stem)
    progress_bar = tqdm(total=len(paths), desc="Renaming...", ncols=80, unit_scale=0.5)
    for i, p in enumerate(paths):
        paths[i] = p.rename(p.with_stem(uuid.uuid4().hex))
        progress_bar.update()
    for i, p in enumerate(paths):
        p.rename(p.with_stem(str(i)))
        progress_bar.update()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(rename)
    else:
        for path in ["resized/" + directory for directory in ("city", "fantasy", "landscapes")]:
            rename(path)
