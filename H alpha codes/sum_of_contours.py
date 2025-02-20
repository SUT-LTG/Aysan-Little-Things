import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI errors

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval
from matplotlib.colors import Normalize, ListedColormap
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import os

folder_path = r"C:\Users\AYSAN\Desktop\project\Trash"

# Import tkinter modules for file dialog
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

def enhance_contrast(image, stretch=AsinhStretch(), interval=PercentileInterval(99.5)):
    """Enhance the contrast of an image using stretching and interval scaling."""
    norm = ImageNormalize(image, interval=interval, stretch=stretch)
    return norm

def plot_contours_V_and_Halpha(
    galaxy_name, V_image, H_image, folder_path, alpha=1, sigma=5, contour_levels=[4, 6, 8, 10]
):
    """Plot V-band image with H-alpha contours and calculate pixel sums within contours."""
    # Enhance contrast of the V-band image
    norm_V = enhance_contrast(V_image)

    # Define colors for contours
    contour_colors = ['yellow', 'blue', 'magenta', 'red']

    # Plot the V-band image with enhanced contrast
    fig, ax = plt.subplots(figsize=(10, 10))
    im1 = ax.imshow(V_image, cmap='gray', alpha=alpha, norm=norm_V, origin='lower')

    # Smooth the H-alpha image for contour plotting
    smoothed = gaussian_filter(H_image, sigma=sigma)

    # Plot contours on the V-band image
    contour_set = ax.contour(
        smoothed, levels=contour_levels, colors=contour_colors, linestyles='solid', linewidths=2
    )

    # Prepare to store sums and masks for each contour level
    sums_per_contour = []
    masks_per_contour = [None] * len(contour_levels)  # Preallocate list

    # Get dimensions of the image
    ny, nx = H_image.shape

    # Prepare a grid of coordinates corresponding to the image pixels
    Y_indices, X_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    points = np.vstack((X_indices.flatten(), Y_indices.flatten())).T  # Shape (num_points, 2)

    # Loop over each contour level
    for i, (level, collection) in enumerate(zip(contour_levels, contour_set.collections)):
        # Initialize mask as zeros
        mask = np.zeros((ny, nx), dtype=bool)

        # Retrieve all paths for this contour level
        polygons = []
        for path in collection.get_paths():
            # Get the vertices of the path
            vertices = path.vertices
            # Create a polygon from the vertices
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
            if poly.is_empty:
                continue
            # If poly is a MultiPolygon, extend the polygons list
            if isinstance(poly, MultiPolygon):
                polygons.extend([p for p in poly.geoms if not p.is_empty])
            else:
                polygons.append(poly)

        if not polygons:
            continue

        # Combine all polygons into a single geometry
        combined_poly = unary_union(polygons)

        # Use Shapely's prepared geometry for efficient point-in-polygon testing
        prepared_poly = prep(combined_poly)

        # Create a mask by checking which points are inside the polygons
        contained = np.array([prepared_poly.contains(Point(xy)) for xy in points], dtype=bool)

        # Reshape mask to the image shape
        mask = contained.reshape((ny, nx))

        # Exclude areas covered by higher contour levels
        if i < len(contour_levels) - 1:
            for j in range(i + 1, len(contour_levels)):
                higher_mask = masks_per_contour[j]
                if higher_mask is not None:
                    mask = np.logical_and(mask, np.logical_not(higher_mask))

        # Store the mask for current contour level
        masks_per_contour[i] = mask.copy()

        # Sum the pixel values inside the mask
        pixel_values = H_image[mask]
        pixel_sum = pixel_values.sum()
        num_pixels = mask.sum()
        mean_pixel_value = pixel_values.mean() if num_pixels > 0 else 0
        max_pixel = pixel_values.max() if num_pixels > 0 else 0
        min_pixel = pixel_values.min() if num_pixels > 0 else 0

        sums_per_contour.append(
            {
                'level': level,
                'sum': pixel_sum,
                'num_pixels': num_pixels,
                'mean_pixel_value': mean_pixel_value,
                'max_pixel': max_pixel,
                'min_pixel': min_pixel,
            }
        )

        # Visualize the mask and save it
        plt.figure()
        plt.imshow(mask, origin='lower', cmap='gray')
        plt.title(f'Mask for Contour Level {level}')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        # Save the mask figure
        mask_fig_path = os.path.join(folder_path, f"{galaxy_name}_contour_level_{level}.png")
        plt.savefig(mask_fig_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Create a custom color bar for contours
    cmap = ListedColormap(contour_colors)
    norm = Normalize(vmin=min(contour_levels), vmax=max(contour_levels))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the color bar with labels for contours
    cbar_contour = fig.colorbar(sm, ax=ax, ticks=contour_levels, fraction=0.046, pad=0.04)
    cbar_contour.set_label('Contour Levels', labelpad=-30, fontsize=12)

    # Add colorbar for the V-band image
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.05)
    cbar1.set_label("V filter, Asinh", labelpad=-40, fontsize=10)

    # Set titles and labels with larger fonts
    ax.set_title(f"{galaxy_name}", fontsize=14)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)

    # Save the figure to the specified path
    output_path = os.path.join(folder_path, f"{galaxy_name}_H-alpha_regions.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure

    # Return the sums and statistics for each contour level
    return sums_per_contour

def open_fits(file_path):
    """Utility function to open FITS files and return the image data."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
    return data

# Main script
if __name__ == "__main__":
    # Create a Tkinter root window and hide it
    root = Tk()
    root.withdraw()

    # Prompt the user to select the V-band FITS file
    print("Please select the V-band FITS file.")
    V_path = askopenfilename(title="Select V-band FITS file", filetypes=[("FITS files", "*.fits")])
    if not V_path:
        print("No V-band file selected. Exiting.")
        exit()

    # Prompt the user to select the H-alpha FITS file
    print("Please select the H-alpha FITS file.")
    H_alpha_path = askopenfilename(title="Select H-alpha FITS file", filetypes=[("FITS files", "*.fits")])
    if not H_alpha_path:
        print("No H-alpha file selected. Exiting.")
        exit()

    # Prompt the user to input the galaxy name
    galaxy_name = input("Enter the galaxy name: ")


    # Load images
    V_image = open_fits(V_path)
    H_alpha_image = open_fits(H_alpha_path)

    # Optionally crop images to focus on the region of interest
    V_image = V_image[80:300, 50:200]
    H_alpha_image = H_alpha_image[80:300, 50:200]

    # Ensure the output folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Call the function and retrieve sums per contour
    sums = plot_contours_V_and_Halpha(
        galaxy_name,
        V_image,
        H_alpha_image,
        folder_path,
        alpha=1,
        sigma=5,
        contour_levels=[4, 6, 8, 10],
    )

    # Print the results with additional statistics
    for contour_info in sums:
        print(f"Contour Level: {contour_info['level']}")
        print(f"  Sum of Pixels: {contour_info['sum']}")
        print(f"  Number of Pixels: {contour_info['num_pixels']}")
        print(f"  Mean Pixel Value: {contour_info['mean_pixel_value']}")
        print(f"  Max Pixel Value: {contour_info['max_pixel']}")
        print(f"  Min Pixel Value: {contour_info['min_pixel']}")

    # Additionally, print the sum of values inside each mask
    print("\nSum of values inside each mask:")
    for contour_info in sums:
        print(f"Contour Level {contour_info['level']}: Sum = {contour_info['sum']}")
