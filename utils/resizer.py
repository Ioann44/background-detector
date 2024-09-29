import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import fire
from PIL import Image
from tqdm import tqdm

type StrPath = str | Path  # TODO: Update wsl to python 3.12


# p = Path(r'C:\Users\akrio\Desktop\Test').glob('**/*')  # possible better implementation
def listdir_recursive(path: StrPath = ".") -> Iterable[Path]:
    for entry in os.listdir(path):
        p = Path(path)
        full_path = p.joinpath(entry)
        if full_path.is_dir():
            yield from listdir_recursive(full_path)
        else:
            yield full_path


def resize_images(
    input_folder, output_folder, target_size=(224, 224), max_images=None, info=False, debug=False
):
    if debug:
        logging.getLogger().setLevel("DEBUG")
        info = True
    elif info:
        logging.getLogger().setLevel("INFO")

    paths = list(listdir_recursive(input_folder))
    logging.info(f"Found {len(paths)} images in {input_folder}")
    logging.debug(f"First 5 paths:\n{'\n'.join(p.__str__() for p in paths[:5])}") # don't use TODO Update to python 3.12
    # replace input_folder prefix to output_folder
    if max_images is None:
        max_images = len(paths)
    logging.info(f"Will save {max_images} images to {output_folder}")

    for path in tqdm(paths[:max_images], "Resizing images", ncols=80, disable=not info):
        new_name = f"{path.stem}.jpg"
        rel_path = path.relative_to(input_folder).parent
        new_dir = Path(output_folder).joinpath(rel_path)
        new_path = new_dir.joinpath(new_name)
        if new_path.is_file():
            continue
        img = Image.open(path)
        img = img.resize(target_size)
        os.makedirs(new_dir, exist_ok=True)
        img = img.convert("RGB")
        img.save(new_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(resize_images)
    else:
        pass
        resize_images("raw-data", "resized")
