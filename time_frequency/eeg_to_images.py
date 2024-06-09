import argparse
import io
import logging
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from tqdm.auto import tqdm
from PIL.Image import open

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("eeg_to_images")

parser = argparse.ArgumentParser(description="Create dataset of spectrograms from directory of EEG files.")
parser.add_argument("--input_dir", type=str)
parser.add_argument("--channel", type=int)
parser.add_argument("--output_dir", type=str, default="data")
args = parser.parse_args()

if args.input_dir is None:
    raise ValueError("You must specify an input directory for the EEG files.")

os.makedirs(args.output_dir, exist_ok=True)
eeg_image_files = []
for s in range(1,10):
    channel_path = os.path.join(args.input_dir, f'subject{s}', f'chan{args.channel}')
    for filename in os.listdir(channel_path):
        file_path = os.path.join(channel_path, filename)
        eeg_image_files.append(file_path)
examples = []
try:
    for image_file in tqdm(eeg_image_files):
        image = open(image_file).convert('L')
        with io.BytesIO() as output:
            image.save(output, format='PNG')
            # output.seek(0)
            examples.extend(
                [
                    {
                        "image": {"bytes": output.getvalue()},
                        "eeg_file": image_file,
                        "slice": 1,
                    }
                ]
            )
except Exception as e:
    print(e)
finally:
    if len(examples) == 0:
        raise FileNotFoundError("No valid EEG files were found.")
    ds = Dataset.from_pandas(
        pd.DataFrame(examples),
        features=Features(
            {
                "image": Image(),
                "eeg_file": Value(dtype="string"),
                "slice": Value(dtype="int16"),
            }
        ),
    )
    dsd = DatasetDict({"train": ds})
    dsd.save_to_disk(args.output_dir)
