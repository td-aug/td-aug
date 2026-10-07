import argparse
import os
import urllib.request

from tqdm import tqdm

DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
DEFAULT_DATASET_DIR = f"~/.ogbench/data"


def download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Download OGBench datasets.

    Args:
        dataset_names: List of dataset names to download.
        dataset_dir: Directory to save the datasets.
    """
    # Make dataset directory.
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download datasets.
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f"{dataset_name}.npz")
        dataset_file_names.append(f"{dataset_name}-val.npz")
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        if not os.path.exists(dataset_file_path):
            dataset_url = f"{DATASET_URL}/{dataset_file_name}"
            print("Downloading dataset from:", dataset_url)
            response = urllib.request.urlopen(dataset_url)
            tmp_dataset_file_path = f"{dataset_file_path}.tmp"
            with tqdm.wrapattr(
                open(tmp_dataset_file_path, "wb"),
                "write",
                miniters=1,
                desc=dataset_url.split("/")[-1],
                total=getattr(response, "length", None),
            ) as file:
                for chunk in response:
                    file.write(chunk)
            os.rename(tmp_dataset_file_path, dataset_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="antmaze-medium-stitch-v0")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)

    args = parser.parse_args()
    dataset_dir = os.path.expanduser(args.dataset_dir)
    download_datasets([args.dataset_name], dataset_dir)
