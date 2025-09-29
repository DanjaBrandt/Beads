import napari
import pprint
import numpy as np
from cellpose import models, io
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, label
from scipy.ndimage import zoom
from skimage.restoration import richardson_lucy
import pandas as pd

from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt, center_of_mass#, binary_closing
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_tv_bregman
from scipy.ndimage import binary_fill_holes
from skimage.morphology import ball, closing
from skimage.morphology import ball, remove_small_objects, binary_closing
import trackpy as tp
from skimage.transform import rescale
from scipy.optimize import minimize
from itertools import product
from skimage.feature import blob_log
from csbdeep.utils import normalize
from skimage.transform import downscale_local_mean

import time

class VolumeDisplacement():
    def __init__(self, microscope_config):
        print("Starting analysis with Napari and Cellpose")
        self.microscope_config = microscope_config
        self.upsample = False
        pprint.pprint(microscope_config)

        pixel_size = self.microscope_config.get("pixel_size", 1)  # µm
        diameter_str = self.microscope_config.get("undeformed_diameter", 1)
        diameter = np.array(list(map(float, diameter_str.split(',')))) # µm
        print('diameter', diameter)
        self.diameter_px = diameter // pixel_size



    def process(self, input_stack_dict):
        """Main processing function (e.g., apply filtering)."""
        return self._process_stack(input_stack_dict)

    def _process_stack(self, input_stack_dict):
        upsampled_stack_dict = self._upsample_z_dimension(input_stack_dict) if self.upsample else input_stack_dict
        beads_position_dict = self._track_beads(upsampled_stack_dict)

        self._measure_bead(upsampled_stack_dict, beads_position_dict)

    def _upsample_z_dimension(self, input_stack_dict):
        """
        Upsample the z dimension of 3D image stacks stored in a dictionary.

        Args:
            input_stack_dict (dict): Dictionary with keys as identifiers and
                values as 3D numpy arrays with shape (Z, Y, X).

        Returns:
            dict: New dictionary with same keys and z-dimension upsampled arrays.
        """

        pixel_size_z = self.microscope_config.get("voxel_depth", 1)
        pixel_size_xy = self.microscope_config.get("pixel_size", 1)

        z_zoom = pixel_size_z / pixel_size_xy

        # Dictionary to hold upsampled stacks
        upsampled_dict = {}

        for key, volume in input_stack_dict.items():
            if not isinstance(volume, np.ndarray) or volume.ndim != 3:
                print(f"Skipping '{key}', not a 3D numpy array.")
                continue

            print(f"Upsampling key '{key}': original shape {volume.shape}")

            # Perform zoom on z dimension only, linear interpolation (order=1)
            upsampled_volume = zoom(volume, zoom=(z_zoom, 1, 1), order=1)

            print(f"Upsampled shape for '{key}' using 1-order interpolation: {upsampled_volume.shape}")

            upsampled_dict[key] = upsampled_volume

        return upsampled_dict

    def _track_beads(self, input_stack_dict):
        """Track beads in 3D stack using Laplacian of Gaussian blob detection."""
        input_stack_array = self._prepare_stack(input_stack_dict)
        scale, sigma = self._calculate_scale_and_sigma(input_stack_array)

        detections = {}
        for t, volume in enumerate(input_stack_array):
            frame_id = f"frame_{t:02d}"
            processed_vol = self._preprocess_volume(volume, scale)
            blobs = self._detect_blobs(processed_vol, sigma)

            for n, coord in enumerate(blobs[:, :3]):
                self._add_detection(detections, n, frame_id, coord, scale)
        pprint.pprint(detections)
        return detections

    def _prepare_stack(self, input_stack_dict):
        """Convert input dict to numpy array."""
        return np.stack([input_stack_dict[k] for k in sorted(input_stack_dict.keys())], axis=0)

    def _calculate_scale_and_sigma(self, stack):
        """Calculate scaling parameters based on microscope config."""

        scale = 1 if stack.shape[2] <= 100 else stack.shape[2] // 100
        scaled_diameter = self.diameter_px[2] // scale
        sigma = scaled_diameter // (np.sqrt(3) * 2)

        print(f'Scale: {scale}, Sigma: {sigma:.2f}')
        return scale, sigma

    def _preprocess_volume(self, volume, scale):
        """Normalize and filter volume."""
        vol_rescaled = rescale(volume, 1 / scale, anti_aliasing=True)
        return gaussian_filter(vol_rescaled, sigma=1)

    def _detect_blobs(self, volume, sigma):
        """Run blob detection with timing."""
        start_time = time.time()
        min_sigma = sigma * 0.5
        max_sigma = sigma * 1.5
        threshold = 0.5 * np.std(volume)
        blobs = blob_log(volume,
                         min_sigma=min_sigma,
                         max_sigma=max_sigma,
                         num_sigma=int(max_sigma - min_sigma),
                         threshold=threshold)
        print(f'Computation time: {time.time() - start_time:.2f}s')
        return blobs

    def _add_detection(self, detections, bead_num, frame_id, coord, scale):
        """Add detection to results dictionary."""
        bead_id = f"bead_{bead_num:02d}"
        if bead_id not in detections:
            detections[bead_id] = {}

        detections[bead_id][frame_id] = {
            'centroid': {
                'z': coord[0] * scale,
                'y': coord[1] * scale,
                'x': coord[2] * scale
            }
        }

    def _measure_bead(self, input_stack_dict, detection_dict):
        for bead_key, bead_params in detection_dict.items():
            print('bead_key:', bead_key)
            print('bead_params:', bead_params)
            for frame_key, frame_data in bead_params.items():
                frame_index = frame_key.split('_')[1]
                centroid = self._get_centroid(frame_data, bead_key, frame_key)
                stack_array = self._get_stack(input_stack_dict, frame_index)
                bead_roi = self._extract_roi(stack_array, centroid)

                if frame_index == '00':
                    calibration_parameters = self._find_calibration_parameters(bead_roi)
                else:
                    if calibration_parameters is None:
                        raise RuntimeError(f"Calibration parameters not found before frame {frame_index}")
                    axes = self._calculate_axes(bead_roi, calibration_parameters)

    def _get_centroid(self, frame_data, bead_key, frame_key):
        centroid = frame_data.get('centroid')
        if centroid is None:
            raise ValueError(f"Missing centroid for frame '{frame_key}' in bead '{bead_key}'")
        try:
            return tuple(map(int, (centroid['x'], centroid['y'], centroid['z'])))
        except KeyError as e:
            raise KeyError(f"Missing coordinate '{e.args[0]}' in centroid for frame '{frame_key}'")

    def _get_stack(self, input_stack_dict, frame_index):
        stack_array = input_stack_dict.get(frame_index)
        if stack_array is None:
            raise ValueError(f"Missing image data for frame '{frame_index}'")
        return stack_array

    def _extract_roi(self, stack_array, centroid):
        x, y, z = centroid
        dz, dy, dx = int(self.diameter_px[0]), int(self.diameter_px[1]), int(self.diameter_px[2])
        shape_z, shape_y, shape_x = stack_array.shape

        y_start, y_end = max(int(y - dy), 0), min(int(y + dy), shape_y)
        x_start, x_end = max(int(x - dx), 0), min(int(x + dx), shape_x)
        z_start, z_end = max(int(z - dz), 0), min(int(z + dz), shape_z)

        return stack_array[z_start:z_end, y_start:y_end, x_start:x_end]

    def _find_calibration_parameters(self, input_bead_stack):

        best_params = self._grid_search_calibration(input_bead_stack, self.diameter_px)
        print('input diam', self.diameter_px)
        print("Best calibration:", best_params)

    def _grid_search_calibration(self, bead_stack, target_diameter_px):
        best_loss = float('inf')
        best_params = None
        best_radii = None

        # Step 1: Full range
        denoising_weights = np.round(np.arange(0.1, 1.01, 0.1), 2)  # [0.1, ..., 1.0]
        otsu_percentages = np.round(np.arange(0.5, 2.01, 0.1), 2)  # [0.5, ..., 2.0]

        print(f"Total combinations: {len(denoising_weights) * len(otsu_percentages)}")

        # Cache denoised stacks
        denoised_cache = {
            weight: denoise_tv_bregman(bead_stack, weight=weight)
            for weight in denoising_weights
        }

        for weight, percentage in product(denoising_weights, otsu_percentages):
            print(f"Trying weight={weight}, percentage={percentage}")
            calibration_parameters = {
                'denoising_weight': weight,
                'otsu_percentage': percentage
            }

            try:
                radii = self._calculate_radii_only(denoised_cache[weight], calibration_parameters)
                diameters_px = 2 * radii
                loss = np.sum(np.abs(diameters_px - target_diameter_px))
            except Exception as e:
                print(f"Skipping ({weight}, {percentage}): {e}")
                continue

            if loss < best_loss:
                best_loss = loss
                best_params = calibration_parameters
                best_radii = radii
                print(f"New best loss={loss:.3f}, params={best_params}")

        print(f"\nBest overall loss: {best_loss:.3f}")
        print(f"Best parameters: {best_params}")
        print(f"Estimated diameters (px): {2 * best_radii}")

        # Step 2: Local refinement
        if best_params is not None:
            w_best, p_best = best_params['denoising_weight'], best_params['otsu_percentage']

            fine_weights = np.round(np.arange(max(0.1, w_best - 0.1), min(1.0, w_best + 0.11), 0.05), 3)
            fine_percentages = np.round(np.arange(max(0.5, p_best - 0.1), min(2.0, p_best + 0.11), 0.05), 3)

            for weight, percentage in product(fine_weights, fine_percentages):
                print(f"Refining weight={weight}, percentage={percentage}")
                calibration_parameters = {
                    'denoising_weight': weight,
                    'otsu_percentage': percentage
                }

                try:
                    if weight in denoised_cache:
                        denoised = denoised_cache[weight]
                    else:
                        denoised = denoise_tv_bregman(bead_stack, weight=weight)
                        denoised_cache[weight] = denoised

                    radii = self._calculate_radii_only(denoised, calibration_parameters)
                    diameters_px = 2 * radii
                    loss = np.sum(np.abs(diameters_px - target_diameter_px))
                except Exception as e:
                    print(f"Skipping ({weight}, {percentage}): {e}")
                    continue

                if loss < best_loss:
                    best_loss = loss
                    best_params = calibration_parameters
                    best_radii = radii
                    print(f"Refined best loss={loss:.3f}, params={best_params}")

            print(f"\nFinal best parameters: {best_params}")
            print(f"Final diameters (px): {2 * best_radii}")
            print(f"Final loss: {best_loss:.3f}")

        return best_params

    def _calculate_radii_only(self, bead_stack, calibration_parameters):
        denoising_weight = calibration_parameters.get('denoising_weight')
        otsu_percentage = calibration_parameters.get('otsu_percentage')

        denoised = denoise_tv_bregman(bead_stack, weight=denoising_weight)
        thresh = threshold_otsu(denoised)
        bead_mask = denoised > (otsu_percentage * thresh)

        labeled_mask = label(bead_mask)
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        bead_mask_clean = (labeled_mask == largest_component_label)

        bead_mask_closing = binary_closing(bead_mask_clean, footprint=ball(1))
        bead_mask_filled = binary_fill_holes(bead_mask_closing)
        bead_mask_closed = closing(bead_mask_filled, ball(1))

        center = center_of_mass(bead_mask_closed)
        coords = np.argwhere(bead_mask_closed)
        radii = np.max(np.abs(coords - center), axis=0)

        return radii






















    def _find_calibration_parameterss(self, input_bead_stack, centroid):

        calibration_parameters = {
            'denoising_weight': 0.2,
            'otsu_percentage': 1.7
        }
        self._calculate_axes(input_bead_stack, calibration_parameters, centroid)

    def _calculate_axes(self, bead_stack, calibration_parameters, centroid):
        print('START MORPH OPERATIONS')
        denoising_weight = calibration_parameters.get('denoising_weight')
        otsu_percentage = calibration_parameters.get('otsu_percentage')

        denoised = denoise_tv_bregman(bead_stack, weight=denoising_weight)

        thresh = threshold_otsu(denoised)
        bead_mask = denoised > (otsu_percentage * thresh)  # Start with 90% of Otsu

        labeled_mask = label(bead_mask)

        component_sizes = np.bincount(labeled_mask.ravel())

        component_sizes[0] = 0

        largest_component_label = np.argmax(component_sizes)
        bead_mask_clean = (labeled_mask == largest_component_label)

        bead_mask_closing = binary_closing(bead_mask_clean, footprint=ball(1))#binary_closing(bead_mask_clean, structure=np.ones((3, 3, 3)))
        bead_mask_filled = binary_fill_holes(bead_mask_closing)
        bead_mask_closed = closing(bead_mask_filled, ball(1))  # Ball radius controls strength

        bead_mask = bead_mask_closing

        # 4. Measure
        center = center_of_mass(bead_mask)
        points_data = np.array([center])
        coords = np.argwhere(bead_mask)
        radii = np.max(np.abs(coords - center), axis=0)  # Approximate XYZ radii
        print('center', center)
        print('radii', radii)
        axes = np.eye(3) * radii * 2
        diameters_px = 2 * radii
        print('input diameter pixel', self.diameter_px)
        print('found diameter pixel', diameters_px)
        #diameters_um = diameters_px * np.array([pixel_size_xy, pixel_size_xy, new_pixel_size_z])
        viewer = napari.Viewer()
        viewer.add_image(bead_stack, name="stack")
        viewer.add_image(denoised, name="denoised")
        viewer.add_labels(bead_mask.astype(np.uint16), name="bead_mask")

        napari.run()

    def _process_stack3(self, input_stack):
        pixel_size_z = self.microscope_config.get("voxel_depth", 1)  # µm
        pixel_size_xy = self.microscope_config.get("pixel_size", 1)  # µm
        fwhm_z = self.microscope_config.get("psf_fwhm_z", 1)  # µm
        fwhm_xy = self.microscope_config.get("psf_fwhm_xy", 1)  # µm

        z_zoom = pixel_size_z / pixel_size_xy
        new_pixel_size_z = pixel_size_z / z_zoom

        # Convert FWHM to standard deviation in voxel units
        sigma_xy_vox = (fwhm_xy / 2.355) / pixel_size_xy
        sigma_z_vox = (fwhm_z / 2.355) / new_pixel_size_z

        print(f"Sigma XY: {sigma_xy_vox:.2f} px, Sigma Z: {sigma_z_vox:.2f} px")

        def generate_psf(sigma_z, sigma_xy, size_z=None, size_xy=None):
            if size_z is None:
                size_z = 2 * int(np.ceil(3 * sigma_z)) + 1
            if size_xy is None:
                size_xy = 2 * int(np.ceil(3 * sigma_xy)) + 1

            shape = (size_z, size_xy, size_xy)
            z, y, x = np.indices(shape)
            cz, cy, cx = np.array(shape) // 2

            psf = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_xy ** 2)
                           + ((z - cz) ** 2) / (2 * sigma_z ** 2)))
            print('PSF sum before:', psf.sum())
            psf /= psf.sum()
            print('PSF sum after:', psf.sum())
            return psf

        psf = generate_psf(sigma_z_vox, sigma_xy_vox)

        print('psf shape', psf.shape)
        print('psf mean', np.mean(psf))

        upsampled_stack = zoom(input_stack, zoom=(z_zoom, 1, 1), order=3)
        print('input_stack shape', input_stack.shape)
        print('z_zoom', z_zoom)
        print('upsampled_stack shape', upsampled_stack.shape)


        print('upsampled_stack min', np.min(upsampled_stack))
        print('upsampled_stack max', np.max(upsampled_stack))
        normalized_stack = upsampled_stack.astype(np.float32) / upsampled_stack.max()

        deconvolved = richardson_lucy(normalized_stack, psf, num_iter=20)
        gauss_stack = gaussian_filter(normalized_stack, sigma=1.0)

        viewer = napari.Viewer()
        viewer.add_image(upsampled_stack, name="psf")
        viewer.add_image(deconvolved, name="deconvolved")
        viewer.add_image(gauss_stack, name="gauss_stack")

        napari.run()

    def _process_stack2d(self, input_stack):
        print(f"imported stack shape: {input_stack.shape}")
        model = models.Cellpose(gpu=False, model_type='cyto')

        # Original voxel size
        z_res = self.microscope_config.get("voxel_depth", 1)  # µm
        xy_res = self.microscope_config.get("pixel_size", 1)  # µm

        diameter_um = self.microscope_config.get("undeformed_diameter", 1)  # µm
        diameter_px = diameter_um / xy_res

        # Compute zoom factor to resample z dim
        z_zoom = z_res / xy_res

        stack_isotropic = resize(
            input_stack,
            (int(input_stack.shape[0] * z_zoom) // 2, input_stack.shape[1] // 2, input_stack.shape[2] // 2),
            order=1,
            anti_aliasing=True,
            preserve_range=True
        ).astype(input_stack.dtype)

        gauss_1 = gaussian_filter(stack_isotropic, sigma=1)
        gauss_2 = gaussian_filter(stack_isotropic, sigma=2)
        #gauss_3 = gaussian_filter(stack_isotropic, sigma=3)

        print('here ok')
        '''sphere, sphere_mask = self._create_sphere()
        sphere_props = regionprops(sphere_mask)
        print('sphere', sphere.shape)
        print("Equivalent diameter (in voxels):", sphere_props[0].equivalent_diameter)
        print('diameter_px', diameter_px)
        sphere = sphere.astype(np.float32)  # ensure float for proper noise addition
        gaussian_noise = np.random.normal(loc=0, scale=0.1, size=sphere.shape)
        noisy_image = sphere + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 1)  # stay within [0, 1]
        scaled = noisy_image * 255
        poisson_noisy = np.random.poisson(scaled).astype(np.uint8)
        poisson_noisy = np.clip(poisson_noisy, 0, 255)'''

        masks, flows, styles, diams = model.eval(
            stack_isotropic.astype(np.float32),
            diameter=diameter_px,
            channels=[0, 0],
            do_3D=True
        )
        print('model ok')
        #print(masks.shape)
        #print(masks.mean())
        #print('diams', diams)
        # View
        viewer = napari.Viewer()
        viewer.add_image(stack_isotropic, name="sphere")
        viewer.add_labels(masks.astype(np.uint16), name="Cellpose Segmentation")
        napari.run()

    def _measures_notes(self, mask):


        props = regionprops(label(masks > 0))[0]
        voxel_volume = z_um * xy_um * xy_um  # in µm³
        volume_um3 = props.area * voxel_volume

        print(f"Segmentation Volume: {volume_um3:.2f} µm³")
        print(f"Centroid (µm): {[round(c * s, 2) for c, s in zip(props.centroid, voxel)]}")

    def _create_sphere(self):

        # --- PARAMETERS ---
        xy_res = 0.09  # target resolution in all directions (µm)
        z_res = xy_res  # make it isotropic directly

        # Physical size of your image volume (match previous one)
        z_um = 52  # from original stack (52 slices at 1 µm)
        xy_um = 416 * 0.09  # ~37.4 µm in X/Y

        # Compute voxel counts based on target resolution
        z_size = int(np.round(z_um / z_res))
        y_size = int(np.round(xy_um / xy_res))
        x_size = int(np.round(xy_um / xy_res))

        print(f"Target shape (isotropic): ({z_size}, {y_size}, {x_size})")

        # Grid in physical units (µm)
        z, y, x = np.meshgrid(
            np.linspace(-z_um / 2, z_um / 2, z_size),
            np.linspace(-xy_um / 2, xy_um / 2, y_size),
            np.linspace(-xy_um / 2, xy_um / 2, x_size),
            indexing='ij'
        )

        # --- Define sphere ---
        radius_um = 17 / 2  # 17 µm diameter
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sphere = (dist <= radius_um).astype(np.uint8)
        #print('sphere', sphere.shape)
        # --- Display (no scaling needed, voxels are isotropic) ---
        #viewer = napari.Viewer()
        #viewer.add_labels(sphere, name="Sphere, built in physical space")
        #napari.run()
        sphere_image = (sphere * 255).astype(np.uint8)
        return sphere_image, sphere