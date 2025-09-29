import os
import pprint
from typing import Callable
from pathlib import Path
import tifffile
import itk


def process_images_in_folder(
        input_folder: str, process_function: Callable, process_key: str, save: bool = True):
    """
    Processes all images in a given folder and saves the results.

    :param input_folder: The root folder containing input images.
    :param process_function: A function that processes a tifffarray stack and returns an output.
    :param process_key: The name of the process function (Napari, ImageJ...)
    :param save: Whether to save the output.
    """

    input_folder = Path(input_folder)

    if process_key == "display":
        print("Here display  results")
    else:
        process_stack(input_folder=input_folder, process_function=process_function, save=save)


def load_tiff_stacks(input_folder: Path) -> dict[str, "np.ndarray"]:
    """
    Load all 3D TIFF files in a folder (and subfolders).

    Args:
        input_folder (Path): Path to the folder containing TIFF files

    Returns:
        dict[str, np.ndarray]: dictionary of stacks indexed by filename order
    """
    tiff_files = sorted(
        [Path(root) / f
         for root, _, files in os.walk(input_folder)
         for f in files
         if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda f: f.name
    )

    stack_dict = {}
    for idx, file_path in enumerate(tiff_files):
        try:
            image_array = tifffile.imread(file_path)
            if image_array.ndim != 3:
                print(f"Skipping {file_path.name}: expected 3D TIFF, got {image_array.shape}")
                continue
            stack_dict[f"{idx:02d}"] = image_array
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue
    return stack_dict


def process_stack(input_folder: Path, process_function: Callable, save: bool = False):
    """
    Load 3D TIFF stacks and apply a processing function.

    Args:
        input_folder (Path): Folder containing TIFF files
        process_function (Callable): Function that takes a dict of stacks
        save (bool): Whether to save the processed results (not implemented here)
    """
    stack_dict = load_tiff_stacks(input_folder)
    if not stack_dict:
        print("No valid TIFF stacks found.")
        return

    process_function(stack_dict)


def load_itk_tiff_stacks(input_folder: Path) -> dict[str, dict]:
    """
    Load all 3D TIFF files in a folder (and subfolders) using ITK.

    Args:
        input_folder (Path): Path to the folder containing TIFF files

    Returns:
        dict[str, dict]: dictionary of stacks indexed by filename order,
                         each value contains 'array' and 'spacing'
                         e.g. {"00": {"array": np.ndarray, "spacing": (dx, dy, dz)}}
    """
    tiff_files = sorted(
        [Path(root) / f
         for root, _, files in os.walk(input_folder)
         for f in files
         if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda f: f.name
    )

    stack_dict = {}
    for idx, file_path in enumerate(tiff_files):
        try:
            itk_image = itk.imread(str(file_path), itk.F)  # read as float
            image_array = itk.GetArrayFromImage(itk_image)  # shape: [z, y, x]

            if image_array.ndim != 3:
                print(f"Skipping {file_path.name}: expected 3D TIFF, got {image_array.shape}")
                continue

            spacing = itk_image.GetSpacing()  # (dx, dy, dz)
            stack_dict[f"{idx:02d}"] = itk_image#{"array": image_array, "spacing": spacing}

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

    return stack_dict