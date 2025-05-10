import little_things_functions as ltf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import RectangleSelector, Button
from astropy.visualization import ZScaleInterval
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import median_filter
import cv2
import pywt  # Ensure PyWavelets is imported for estimate_sigma

# Function to apply de-speckling methods
def apply_despeckle_methods(image_section):
    methods = {}
    methods['original'] = image_section

    # Lee Filter
    def lee_filter(img, size):
        img = img.astype(np.float64)
        mean_kernel = np.ones((size, size)) / (size * size)
        img_mean = cv2.filter2D(img, -1, mean_kernel)
        img_sqr_mean = cv2.filter2D(img**2, -1, mean_kernel)
        img_variance = img_sqr_mean - img_mean**2
        img_variance = np.maximum(img_variance, 0)
        overall_variance = np.mean(img_variance)
        img_weights = img_variance / (img_variance + overall_variance + 1e-8)
        img_output = img_mean + img_weights * (img - img_mean)
        return img_output

    lee_filtered = lee_filter(image_section, size=5)
    methods['lee'] = lee_filtered

    # Kuan Filter
    def kuan_filter(img, size):
        img = img.astype(np.float64)
        mean_kernel = np.ones((size, size)) / (size * size)
        img_mean = cv2.filter2D(img, -1, mean_kernel)
        img_sqr_mean = cv2.filter2D(img**2, -1, mean_kernel)
        img_variance = img_sqr_mean - img_mean**2
        img_variance = np.maximum(img_variance, 0)
        overall_variance = np.mean(img_variance)
        img_weights = img_variance / (img_variance + overall_variance + 1e-8)
        img_output = img_mean + img_weights * (img - img_mean)
        return img_output

    kuan_filtered = kuan_filter(image_section, size=5)
    methods['kuan'] = kuan_filtered

    # Non-Local Means Filter
    sigma_est = np.mean(estimate_sigma(image_section, channel_axis=None))
    nl_means_filtered = denoise_nl_means(
        image_section,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=3,
        channel_axis=None
    )
    methods['nl_means'] = nl_means_filtered

    return methods

# Function to process and save images with thresholds
def process_image(galaxy_name, image_section, thresholds, output_folder):
    # Apply de-speckling methods
    methods = apply_despeckle_methods(image_section)

    # Create folders for outputs
    galaxy_folder = os.path.join(output_folder, galaxy_name)
    os.makedirs(galaxy_folder, exist_ok=True)
    despeckle_folder = os.path.join(galaxy_folder, 'de-speckled_images')
    os.makedirs(despeckle_folder, exist_ok=True)
    threshold_folder = os.path.join(galaxy_folder, 'thresholded_images')
    os.makedirs(threshold_folder, exist_ok=True)

    results = {"galaxy_name": galaxy_name}

    # Process each method
    for method_name, img in methods.items():
        # Save the de-speckled image
        plt.figure()
        plt.imshow(img, origin='lower', cmap='gray')
        plt.title(f'{method_name.capitalize()} De-speckled Image')
        plt.colorbar()
        despeckle_path = os.path.join(despeckle_folder, f'{galaxy_name}_{method_name}.png')
        plt.savefig(despeckle_path)
        plt.close()

        # Process thresholds for each method
        for threshold in thresholds:
            modified_image = np.copy(img)
            modified_image[modified_image <= threshold] = np.nan

            # Save the thresholded image
            plt.figure()
            plt.imshow(modified_image, origin='lower', cmap='gray')
            plt.title(f'{method_name.capitalize()} - Threshold {threshold}')
            plt.colorbar()
            threshold_image_path = os.path.join(threshold_folder, f'{galaxy_name}_{method_name}_threshold_{threshold}.png')
            plt.savefig(threshold_image_path)
            plt.close()

            # Calculate the sum of non-NaN values
            sum_non_nan = np.nansum(modified_image)
            results[f'{method_name}_{threshold}'] = sum_non_nan

    # Save results to CSV
    results_file = os.path.join(output_folder, "different methods results.csv")
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        results_df = pd.read_csv(results_file)
    else:
        columns = ["galaxy_name"] + [f'{method}_{threshold}' for method in methods.keys() for threshold in thresholds]
        results_df = pd.DataFrame(columns=columns)

    new_results_df = pd.DataFrame([results])
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    results_df.drop_duplicates(subset="galaxy_name", keep='last', inplace=True)
    results_df.to_csv(results_file, index=False)

    return results_df

# Function to display the image and let the user select a region
def select_image_region(image):
    fig, ax = plt.subplots()
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image)
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Select a region using the mouse')

    # Variables to store the coordinates
    coords = []

    # Function to be called when the rectangle is drawn
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        coords.append((xmin, xmax, ymin, ymax))

    rectangle_selector = RectangleSelector(
        ax, onselect,
        useblit=False, button=[1],
        minspanx=5, minspany=5, spancoords='pixels',
        interactive=True
    )

    # Function to be called when the "OK" button is clicked
    def on_ok(event):
        plt.close(fig)

    # Add "OK" button
    ok_ax = plt.axes([0.8, 0.01, 0.1, 0.075])
    ok_button = Button(ok_ax, 'OK')
    ok_button.on_clicked(on_ok)

    plt.show()

    if coords:
        xmin, xmax, ymin, ymax = coords[0]
        selected_region = image[ymin:ymax, xmin:xmax]
        return selected_region
    else:
        print("No region was selected.")
        return None

# Main script
def main():
    # Initialize tkinter and hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Browse to select the image file (FITS file)
    image_path = filedialog.askopenfilename(
        title="Select FITS Image File", 
        filetypes=[("FITS files", "*.fits")]
    )
    if not image_path:
        print("No file was selected.")
        return

    print(f"Selected file: {image_path}")

    # Ask for the galaxy name
    galaxy_name = input("Enter the galaxy name: ")

    # Open and process the image using a custom function (from little_things_functions)
    image = ltf.open_fits(image_path)

    # Perform byte order conversion if required for the main image
    if image.dtype.byteorder == '>':
        print("Converting image data from big-endian to native byte order.")
        image = image.byteswap().newbyteorder()

    # Handle NaN and infinite values in the main image
    if np.isnan(image).any():
        print("Warning: Image contains NaN values. Replacing NaNs with zeros.")
        image = np.nan_to_num(image)
    if np.isinf(image).any():
        print("Warning: Image contains infinite values.")
        image = np.where(np.isinf(image), np.finfo(np.float64).max, image)

    # Let the user select a region from the main image
    image_section = select_image_region(image)
    if image_section is None:
        print("Processing aborted.")
        return

    # Define threshold levels
    threshold_levels = [4, 6, 8, 10]

    # Define the output folder
    output_folder = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions"

    # Process the image with de-speckling and thresholding
    results_df = process_image(galaxy_name, image_section, threshold_levels, output_folder)
    print(f"Results saved to {os.path.join(output_folder, 'different methods results.csv')}")
    print(f"De-speckled images and thresholded images saved in: {os.path.join(output_folder, galaxy_name)}")

if __name__ == "__main__":
    main()
