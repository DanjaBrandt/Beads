import os
import argparse
import sys
from pathlib import Path

from volume_render import VolumeDisplacement
from config import Config
from utils.process_functions import process_images_in_folder

class MainClass:
    """
        Main class to handle different processing types (Radon, Fourier, Display).
        """

    def __init__(self, input_config, process_key, save):
        self.microscope_config = input_config
        self.process_key = process_key
        self.save = save

        if process_key == "volume_displacement":
            self.processor = VolumeDisplacement(self.microscope_config).process
        else:
            # self.logger.error(f"Invalid process key: {process_key}")
            raise ValueError(f"Unknown process_key: {process_key}")

    def run(self, input_folder):

        process_images_in_folder(input_folder, self.processor, self.save)


def get_args():
    parser = argparse.ArgumentParser(description="Provide config values")

    parser.add_argument("--mode", type=str, choices=["napari", "imagej", "other"], required=True,
                        help="Choose processing type: 'radon', 'fourier', or 'display'")

    parser.add_argument("--save", type=str, default="True", choices=["True", "False"],
                        help="Choose if to save results")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:  # If there are command-line arguments
        args = get_args()
    else:  # No CLI arguments, use default values
        args = argparse.Namespace(
            mode="volume_displacement",
            input_folder="C:/Users/Danja Brandt/Desktop/FU/Beads/test_data",
            save=False
        )

    input_config = Config(args.input_folder).as_dict()
    app = MainClass(input_config, args.mode, args.save)
    app.run(args.input_folder)
