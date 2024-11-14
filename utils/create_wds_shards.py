from typing import Generator, Union
from pathlib import Path
import argparse
import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from original Hephaestus data.""")

parser.add_argument(
    "--data_dir",
    default="/scratch/SFD25/LiCSAR-web-tools/",
    help="directory containing unzipped Hephaestus dataset",
)
parser.add_argument(
    "--output_dir",
    default='/scratch/SDF25/Hephaestus_WDS/',
    help="diectory for tar shards in webdataset format"
)
args = parser.parse_args()

def main():
    samples = create_sample_generator(args.data_dir)
    write_webdataset(args.output_dir, samples)

def write_webdataset(output_dir: str, sample_gen: Generator[str, None, None]) -> None:
    with wds.ShardWriter(str(args.output_dir)+"hephaestus-%06d.tar", maxcount=1000) as sink:
        for sample in sample_gen:
            sink.write(sample)

def create_sample_generator(data_dir: str) -> Generator[str, None, None]:

    for index, image_file in enumerate(Path(data_dir).rglob("*diff.png")):
        sample = create_sample(image_file, index)
        if sample is None:
            continue
        # Yield optimizes memory by returning a generator that only generates samples as requested
        yield sample        

def create_sample(image_file: str, index: int) -> Union[dict, None]:
    date = str(image_file).split("/")[-1].split(".")[0]
    frame = str(image_file).split("/")[-4]
    label = frame+'_'+date
    try:
        with open(image_file, "rb") as f:
            ifg_data = f.read()
        instance_class = label
        with open(str(image_file).split(".")[0]+'.geo.cc.png', "rb") as f:
            coherence_data = f .read()
        if not ifg_data or not instance_class or not coherence_data:
            # Ignore incomplete records
            return None
        return {
            "__key__": "sample_%09d" % index,
            "ifg.png": ifg_data,
            "cc.png": coherence_data,
            "cls": instance_class
        }
    except FileNotFoundError as err:
        print(err)
        return None

if __name__ == '__main__':
    main()