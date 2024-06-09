import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str)
parser.add_argument("--channel", type=int)
args = parser.parse_args()
for s in os.listdir(args.root):
    data_path = os.path.join(args.root, s, f'chan{args.channel}')
    os.system(f'accelerate launch --config_file accelerate_local.yaml train_unet_eeg.py --dataset_path {data_path}')