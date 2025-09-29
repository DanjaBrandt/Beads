from segmenter import BeadSegmenter
from simulator import BeadSimulator
from comparator import ShapeComparator
from optimizer import ParameterOptimizer
from pipeline import BeadAnalysisPipeline
from config import Config
import argparse
import sys

from utils.process_functions import load_tiff_stacks, load_itk_tiff_stacks
# (optionally: load configs from YAML/JSON)
# import yaml
# with open("configs/segmentation.yaml") as f:
#     segmentation_config = yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser(description="Bead analysis pipeline")
    parser.add_argument("input_folder", type=str, help="Root folder containing timeseries_* subfolders")
    parser.add_argument("--timeseries", type=int, nargs="*", default=None,
                        help="Which timeseries to process (e.g., 0 1 2). Default: all")
    parser.add_argument("--timestamp", type=int, nargs="*", default=None,
                        help="Which timestamps to process (e.g., 0 1 2). Default: all")
    return parser.parse_args()


def main(input_folder, **kwargs):
    timeseries = kwargs.get("timeseries", None)
    timestamp = kwargs.get("timestamp", None)


    # Instantiate components
    microscope_config = Config(input_folder).as_dict()

    segmenter = BeadSegmenter(config=microscope_config)
    simulator = BeadSimulator(abaqus_config={})
    comparator = ShapeComparator(method="pointcloud")
    optimizer = ParameterOptimizer(simulator, comparator, optimizer_config={})

    # Build pipeline
    pipeline = BeadAnalysisPipeline(segmenter, simulator, comparator, optimizer)

    # Example inputs
    image_stacks = load_itk_tiff_stacks(input_folder)  # list of 3D numpy arrays
    initial_params = {"E": 1000, "nu": 0.45}  # guess for material properties
    timepoints = [0, 10, 20, 30]  # in seconds

    # Run pipeline
    best_params = pipeline.run(image_stacks, initial_params, timepoints)
    print("Optimized material parameters:", best_params)


if __name__ == "__main__":

    if __name__ == "__main__":
        if len(sys.argv) > 1:  # CLI mode
            args = get_args()
        else:  # Default mode (useful for testing / notebooks)
            args = argparse.Namespace(
                input_folder="./data",
                timeseries=0,
                timestamp=0,
                #save=0
            )

    main(input_folder=args.input_folder, timeseries=args.timeseries, timestamp=args.timestamp)