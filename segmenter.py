import napari
import pprint
import numpy as np
import itk
import time
from scipy import ndimage
from skimage.filters import threshold_otsu

class BeadSegmenter:
    def __init__(self, config):
        self.config = config  # thresholds, filters, etc.

        self.xy_spacing = self.config.get("pixel_size", 1)  # µm
        self.z_spacing = self.config.get("voxel_depth", 1)  # µm

    def segment(self, image_stack):
        # return 3D mesh or point cloud of bead

        #print("Importet Stack  of shape: ", image_stack["array"].shape)
        #print("itk spacing", image_stack["spacing"])

        #image_stack = image_stack[:, ::2, ::2]  # [z, y, x]

        image = itk.GetImageFromArray(image_stack.astype(np.float32))
        image.SetSpacing((self.xy_spacing, self.xy_spacing, self.z_spacing))  # (dx, dy, dz)
        start = time.time()

        # Gaussian smoothing (spacing-aware)
        gaussian = itk.SmoothingRecursiveGaussianImageFilter.New(image)
        gaussian.SetSigma(1)  # in physical units (microns)
        gaussian.Update()
        smoothed = gaussian.GetOutput()

        end = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] smoothed done in {end - start:.2f} seconds")

        smoothed_array = itk.GetArrayFromImage(smoothed)  # shape: [z, y, x]
        otsu_thresh = threshold_otsu(smoothed_array)

        print('max', smoothed_array.max())
        print('min', smoothed_array.min())
        print('mean', smoothed_array.mean())
        print('otsu', otsu_thresh)
        # Thresholding
        binary = itk.binary_threshold_image_filter(
            smoothed,
            lower_threshold=int(otsu_thresh),
            upper_threshold=1293,
            inside_value=1,
            outside_value=0
        )

        CastFilterType = itk.CastImageFilter[type(binary), itk.Image[itk.UC, 3]]
        cast_filter = CastFilterType.New(binary)
        cast_filter.Update()
        binary_uint8 = cast_filter.GetOutput()

        connected = itk.ConnectedComponentImageFilter.New(binary_uint8)
        connected.Update()
        labeled = connected.GetOutput()
        labeled_array = itk.GetArrayFromImage(labeled)  # shape: [z, y, x]

        print('labeled_array.shape', labeled_array.shape)
        spacing = labeled.GetSpacing()  # (dx, dy, dz)
        z_spacing, y_spacing, x_spacing = labeled.GetSpacing()
        print('spacing', spacing)
        origin = labeled.GetOrigin()  # in physical space
        print('origin', origin)
        labels = np.unique(labeled_array)
        print('labels', labels)
        labels = labels[labels != 0]  # drop background

        measurements = {}

        for label in labels:
            coords = np.argwhere(labeled_array == label)  # voxel indices (z, y, x)

            # Convert to physical coordinates
            phys_coords = coords.astype(np.float32)
            phys_coords[:, 0] = phys_coords[:, 0] * spacing[2] + origin[2]  # z
            phys_coords[:, 1] = phys_coords[:, 1] * spacing[1] + origin[1]  # y
            phys_coords[:, 2] = phys_coords[:, 2] * spacing[0] + origin[0]  # x



            # Volume = voxel count × voxel volume
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            volume = coords.shape[0] * voxel_volume

            # Centroid
            centroid = phys_coords.mean(axis=0)  # (z, y, x)
            voxel_centroid = coords.mean(axis=0)

            # Bounding box extents
            min_z, min_y, min_x = phys_coords.min(axis=0)
            max_z, max_y, max_x = phys_coords.max(axis=0)
            bbox = (min_x, max_x, min_y, max_y, min_z, max_z)

            # Equivalent spherical radius
            r_equiv = ((3 * volume) / (4 * np.pi)) ** (1 / 3)

            # Axis-aligned radii from centroid
            diffs = np.abs(phys_coords - centroid)  # distances from centroid
            radii_axis = diffs.max(axis=0)  # (rz, ry, rx) in µm

            measurements[label] = {
                "volume": volume,
                "centroid": centroid,
                "voxel_centroid": voxel_centroid,
                "bbox": bbox,
                "equivalent_radius": r_equiv,
                "extents": (max_x - min_x, max_y - min_y, max_z - min_z),  # size in μm
                "radii_axis": radii_axis
            }

        for label, m in measurements.items():
            print(f"Label {label}")
            print(f"  Volume: {m['volume']:.2f} µm³")
            print(f"  Centroid: (z={m['centroid'][0]:.2f}, y={m['centroid'][1]:.2f}, x={m['centroid'][2]:.2f})")
            print(f"  Centroid-Voxels: (z={m['voxel_centroid'][0]:.2f}, y={m['voxel_centroid'][1]:.2f}, x={m['voxel_centroid'][2]:.2f})")
            print(f"  Bounding box extents (x, y, z): {m['extents']}")
            print(f"  Equivalent spherical radius: {m['equivalent_radius']:.2f} µm")
            print(f"  Radii along axes (rz, ry, rx): {m['radii_axis']}")


        # Keep only the largest connected component
        #connected = itk.connected_component_image_filter(binary_uint8)
        #label_shape = itk.label_shape_keep_n_objects_image_filter(connected)
        #label_shape.SetNumberOfObjects(1)
        #label_shape.Update()
        #mask_itk = label_shape.GetOutput()



        viewer = napari.Viewer()
        viewer.add_image(image, name="stack")
        viewer.add_labels(binary_uint8, name="bead_mask")
        #viewer.add_labels(labeled_array, name="labeled array")#, scale=(z_spacing, y_spacing, x_spacing), name="stretched")


        napari.run()

    def set_stack_dimensions(self, itk_image):
        new_spacing = (self.pixel_size, self.pixel_size, self.pixel_size)
        resample = itk.ResampleImageFilter.New(itk_image)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize([int(sz * spc / new_spacing[0]) for sz, spc in
                          zip(itk_image.GetLargestPossibleRegion().GetSize(), itk_image.GetSpacing())])
        resample.Update()
        resampled_image = resample.GetOutput()
        return resampled_image

