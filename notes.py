def _track_beadss(self, input_stack_dict):
    input_stack_list = [input_stack_dict[k] for k in sorted(input_stack_dict.keys())]
    input_stack_array = np.stack(input_stack_list, axis=0)
    print(
        f"input image array max: {input_stack_array.max()}, min: {input_stack_array.min()}, mean: {np.mean(input_stack_array)}")

    pixel_size_xy = self.microscope_config.get("pixel_size", 1)  # µm
    diameter_um = self.microscope_config.get("undeformed_diameter", 1)  # µm
    diameter_pixel = diameter_um // pixel_size_xy

    t_dim, z_dim, y_dim, x_dim = input_stack_array.shape

    scale = 1 if y_dim <= 100 else y_dim // 100

    print('scale', scale)

    scaled_diameter = diameter_pixel // scale
    # print('a', round(time.time(), 2))
    sigma = scaled_diameter // (np.sqrt(3) * 2)
    # print('b', round(time.time(), 2))
    detections = {}

    for t, volume in enumerate(input_stack_array):
        print(f"volume max: {volume.max()}, min: {volume.min()}, mean: {np.mean(volume)}, shape: {volume.shape}")
        frame_id = f"frame_{t:02d}"  # Formats with leading zeros
        volume_rescaled = rescale(volume, 1 / scale, anti_aliasing=True)
        print(
            f"volume_rescaled max: {volume_rescaled.max()}, min: {volume_rescaled.min()}, mean: {np.mean(volume_rescaled)}, shape: {volume_rescaled.shape}")

        volume_norm = volume_rescaled  # normalize(volume_rescaled, 1, 99.8, clip=True)
        print('mean norm', np.mean(volume_norm))
        volume_gauss = gaussian_filter(volume_norm, sigma=1)

        threshold = 0.5 * np.std(volume_norm)

        # Run prediction
        start_time = time.time()
        min_sigma = sigma - sigma * 0.5
        max_sigma = sigma + sigma * 0.5
        num_sigma = int(max_sigma - min_sigma)
        blobs = blob_log(volume_gauss, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                         threshold=threshold)
        coords = blobs[:, :3]

        elapsed_time = time.time() - start_time
        print(f'computation time: {elapsed_time}')
        #

        for n, coord in enumerate(coords):
            bead_id = f"bead_{n:02d}"  # Formats with leading zeros (00, 01,...)

            if bead_id not in detections:
                detections[bead_id] = {}

            detections[bead_id][frame_id] = {
                'centroid': {
                    'z': coord[0] * scale,
                    'y': coord[1] * scale,
                    'x': coord[2] * scale
                }
            }

    pprint.pprint(detections)
    # coords = detections_df[['frame', 'z', 'y', 'x']].to_numpy()

    viewer = napari.Viewer()
    # viewer.add_image(input_stack_array, name="binary")
    viewer.add_image(volume_rescaled, name='rescaled')
    # viewer.add_image(volume_norm, name='norm')
    # viewer.add_points(coords, name="Beads", size=20, face_color='red', ndim=4)

    napari.run()


def _track_beads2(self, input_stack_dict):
    input_stack_list = [input_stack_dict[k] for k in sorted(input_stack_dict.keys())]
    input_stack_array = np.stack(input_stack_list, axis=0)
    print(f'input array shape {input_stack_array.shape}')

    pixel_size_xy = self.microscope_config.get("pixel_size", 1)  # µm
    diameter_um = self.microscope_config.get("undeformed_diameter", 1)  # µm

    diameter_pixel = diameter_um // pixel_size_xy

    print(f'Approximized diam in pixel: {diameter_pixel}')

    t_dim, z_dim, y_dim, x_dim = input_stack_array.shape
    scale_factors = (5, 5, 5)
    down_z = (z_dim + scale_factors[0] - 1) // scale_factors[0]  # + (1 if z_dim % 2 != 0 else 0)
    down_y = (y_dim + scale_factors[1] - 1) // scale_factors[1]  # + (1 if y_dim % 2 != 0 else 0)
    down_x = (x_dim + scale_factors[2] - 1) // scale_factors[2]  # + (1 if x_dim % 2 != 0 else 0)

    scaled_diameter = diameter_pixel // scale_factors[0]
    print(f'scaled diam in pixel: {scaled_diameter}')

    # For the Laplacian of Gaussian (LoG) method, The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    #     a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.

    sigma = scaled_diameter // (np.sqrt(3) * 2)

    print(f'approximated needed sigma: {sigma}')

    # Pre-allocate downscaled 4D array
    downscaled_stack = np.zeros((t_dim, down_z, down_y, down_x), dtype=np.float32)
    print('downscaled_stack shape', downscaled_stack.shape)

    detections = []

    for t, volume in enumerate(input_stack_array):
        volume = downscale_local_mean(volume, scale_factors)  # here better to use skimage.transform import rescale
        # image_small = rescale(image, 0.5, anti_aliasing=False)
        downscaled_stack[t] = volume
        volume_norm = normalize(volume, 1, 99.8, clip=True)
        volume_norm = gaussian_filter(volume_norm, sigma=1)

        # Run prediction
        start_time = time.time()
        print(f'start stack {t}')

        blobs = blob_log(volume_norm, min_sigma=5, max_sigma=15, num_sigma=10, threshold=0.2)
        coords = blobs[:, :3]

        elapsed_time = time.time() - start_time
        print(f'computation time: {elapsed_time}')
        # coords = details['points']  # shape: (N, 3) = (z, y, x)

        for coord in coords:
            detections.append({'frame': t, 'z': coord[0] * scale_factors[0],
                               'y': coord[1] * scale_factors[1],
                               'x': coord[2] * scale_factors[2]})

    detections_df = pd.DataFrame(detections)
    print(detections_df)
    coords = detections_df[['frame', 'z', 'y', 'x']].to_numpy()

    viewer = napari.Viewer()
    viewer.add_image(input_stack_array, name="binary")
    viewer.add_points(coords, name="Beads", size=5, face_color='red', ndim=4)

    napari.run()


def _track_beads2(self, input_stack_dict):
    input_stack_list = [input_stack_dict[k] for k in sorted(input_stack_dict.keys())]
    input_stack_array = np.stack(input_stack_list, axis=0)
    print(input_stack_array.shape)

    ####################
    # model = StarDist3D.from_pretrained('3D_demo')
    # labels, details = model.predict_instances(volume_norm, n_tiles=(1, 4, 4))

    #################
    def segment_beads_3d(volume):
        smoothed = gaussian_filter(volume, sigma=1)
        threshold = 0.6 * smoothed.max()
        binary = volume > threshold
        binary = binary_closing(binary, ball(3))
        binary = remove_small_objects(binary, min_size=3)
        labeled = label(binary)
        return labeled

    def extract_centroids(stack_4d):
        detections = []
        for t in range(stack_4d.shape[0]):
            labeled = segment_beads_3d(stack_4d[t])
            for region in regionprops(labeled):
                z, y, x = region.centroid
                detections.append({'frame': t, 'z': z, 'y': y, 'x': x})
        return pd.DataFrame(detections)

    timepoints = input_stack_array.shape[0]
    z, y, x = input_stack_array.shape[1:]

    # Prepare output stack
    thresholded_stack = np.zeros_like(input_stack_array, dtype=np.uint16)  # or bool if binary

    for t in range(timepoints):
        volume = input_stack_array[t]
        cleaned = segment_beads_3d(volume)  # returns labeled 3D or binary
        thresholded_stack[t] = cleaned

    centroids = extract_centroids(input_stack_array)
    print('centroids \n', centroids)
    coords = centroids[['z', 'y', 'x']].to_numpy()  # shape: (N, 3)

    # Optionally, get a timepoint if you want to handle 4D later
    timepoints = centroids['frame'].to_numpy()
    coords_4d = np.column_stack([timepoints, coords])

    #################

    def segment_beads_3d(volume):
        thresh = threshold_otsu(volume)
        binary = volume > thresh

        labeled = label(binary)
        return labeled

    def detect_beads_log(volume, min_sigma=1, max_sigma=3, threshold=0.01):
        """
        Detects 3D beads using Laplacian of Gaussian.
        Returns centroids in (z, y, x).
        """
        blobs = blob_log(volume, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold, overlap=0.5)
        # blob_log returns z, y, x, sigma
        return blobs[:, :3]

    ### --- Extract centroids --- ###
    def extract_centroids(stack_4d):
        detections = []
        for t in range(stack_4d.shape[0]):
            labeled = segment_beads_3d(stack_4d[t])
            for region in regionprops(labeled):
                z, y, x = region.centroid
                detections.append({'frame': t, 'z': z, 'y': y, 'x': x})
        return pd.DataFrame(detections)

    ### --- Track beads using TrackPy --- ###
    def track_beads(detections_df):
        tracked_df = tp.link_df(detections_df, search_range=5, memory=1)
        return tracked_df

    ### --- Visualize with Napari --- ###
    def show_in_napari(stack_4d, tracked_df):
        viewer = napari.Viewer()
        viewer.add_image(stack_4d, name="Bead Stack", scale=(1, 1, 1, 1))

        tracks_data = []
        for _, row in tracked_df.iterrows():
            tracks_data.append([row['frame'], row['z'], row['y'], row['x'], row['particle']])
        tracks_data = np.array(tracks_data)

        viewer.add_tracks(tracks_data, name="Bead Tracks", scale=(1, 1, 1, 1))
        napari.run()

    detections = extract_centroids(input_stack_array)
    tracked = track_beads(detections)
    show_in_napari(input_stack_array, tracked)




def _process_stack4(self, input_stack):
    pixel_size_z = self.microscope_config.get("voxel_depth", 1)  # µm
    pixel_size_xy = self.microscope_config.get("pixel_size", 1)  # µm
    diameter_um = self.microscope_config.get("undeformed_diameter", 1)  # µm

    print(f'Original diameter {diameter_um} µm')

    z_zoom = pixel_size_z / pixel_size_xy
    new_pixel_size_z = pixel_size_z / z_zoom

    upsampled_stack = zoom(input_stack, zoom=(z_zoom, 1, 1), order=3)

    denoised = denoise_tv_bregman(upsampled_stack, weight=0.3)
    print('denoised stack shape', denoised.shape)

    # 2. Threshold
    thresh = threshold_otsu(denoised)
    bead_mask = denoised > (1.7 * thresh)  # Start with 90% of Otsu

    # 3. Cleanup
    # bead_mask = binary_closing(bead_mask, structure=np.ones((3, 3, 3)))
    labeled_mask = label(bead_mask)

    # 2. Get sizes of each connected component
    component_sizes = np.bincount(labeled_mask.ravel())

    # 3. Ignore the background (label 0)
    component_sizes[0] = 0

    # 4. Keep only the largest component
    largest_component_label = np.argmax(component_sizes)
    bead_mask_clean = (labeled_mask == largest_component_label)

    # 5. Now apply binary closing
    bead_mask_closing = binary_closing(bead_mask_clean, structure=np.ones((3, 3, 3)))
    bead_mask_filled = binary_fill_holes(bead_mask_closing)
    bead_mask_closed = closing(bead_mask_filled, ball(1))  # Ball radius controls strength

    bead_mask = bead_mask_closing

    # 4. Measure
    distances = distance_transform_edt(bead_mask)
    center = center_of_mass(bead_mask)
    points_data = np.array([center])
    coords = np.argwhere(bead_mask)
    radii = np.max(np.abs(coords - center), axis=0)  # Approximate XYZ radii
    print('center', center)
    print('radii', radii)
    axes = np.eye(3) * radii * 2
    diameters_px = 2 * radii
    diameters_um = diameters_px * np.array([pixel_size_xy, pixel_size_xy, new_pixel_size_z])

    print(f"Measured diameters (XY, XY, Z): {diameters_um}")
    viewer = napari.Viewer()
    viewer.add_image(upsampled_stack, name="stack")
    viewer.add_image(denoised, name="denoised")
    viewer.add_labels(bead_mask.astype(np.uint16), name="bead_mask")
    viewer.add_labels(bead_mask_filled.astype(np.uint16), name="bead_mask_filled")
    viewer.add_labels(bead_mask_closed.astype(np.uint16), name="bead_mask_closed")

    viewer.add_points(
        points_data,
        name='Bead Center',
        size=10,  # Point size
        face_color='red',
        opacity=0.7
    )
    vectors_data = np.array([
        [center, axes[0]],  # X-axis
        [center, axes[1]],  # Y-axis
        [center, axes[2]]  # Z-axis
    ])

    viewer.add_vectors(
        vectors_data,
        name='Ellipsoid Axes',
        edge_color=['red', 'green', 'blue'],
        length=1.0
    )

    napari.run()


##THIS WORKS GOOD

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



def process_stack(input_folder: Path, process_function: Callable, save: bool):
    """
        Process all TIFF files in a folder and its subfolders, ignoring other file types.

        Args:
            input_folder (str): Path to the folder containing TIFF files
        """
    '''
    for root, _, files in os.walk(input_folder):
        root_path = Path(root)

        for file in files:
            if not file.lower().endswith(('.tif', '.tiff')):
                continue
            file_path = root_path / file
            try:
                image_array = tifffile.imread(file_path)
                if image_array is None:
                    continue

                process_function(image_array)
            except tifffile.TiffFileError as e:
                print(f"TIFF read error in {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing {file_path}: {e}")
                continue'''
    tiff_files = sorted(
        [Path(root) / f
         for root, _, files in os.walk(input_folder)
         for f in files
         if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda f: f.name
    )
    #print(tiff_files)

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

    process_function(stack_dict)