from typing import Generator
from pathlib import Path
import argparse
import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from original Hephaestus data.""")

parser.add_argument(
    "--data",
    default="/scratch/SFD25/LiCSAR-web-tools/",
    help="directory containing unzipped Hephaestus dataset",
)
args = parser.parse_args()

print(args)

def create_sample_generator(data_dir: str) -> Generator[str, None, None]:
    print('start')
    print(Path(data_dir))
    
    for index, image_file in enumerate(Path(data_dir).rglob(".png")):
        print('start')
        print(index)
        print(image_file)

sample = create_sample_generator(args.data)
