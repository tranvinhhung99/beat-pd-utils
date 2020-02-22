import argparse
import pandas as pd

import os
from beat_pd.dataset.utils import interpolate_series

def parse_args():
    parser = argparse.ArgumentParser("Interpolate dataset then save it for later use")
    parser.add_argument("--data_folder")
    parser.add_argument('--output_folder')

    parser.add_argument("--log_step", default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_folder = args.data_folder
    log_step = args.log_step
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(data_folder):
        num_file = len(files)
        for i, fname in enumerate(files):
            output_path = os.path.join(output_folder, fname)
            input_path = os.path.join(root, fname)

            raw_data = pd.read_csv(input_path)
            synthesis_data = interpolate_series(raw_data)
            synthesis_data.to_csv(output_path, index=False)

            if i % log_step == 0:
                print(f"Finished: {i} / {num_file}")




    
    