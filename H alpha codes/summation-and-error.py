import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import os
import little_things_functions as ltf
from skimage.restoration import denoise_nl_means, estimate_sigma

galaxy_name = "DDO 101"
# -----------------------------
# Load the FITS image and plot raw data
# -----------------------------
def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    # Handle byte order if needed
    if image_data.dtype.byteorder == '>':
        image_data = image_data.byteswap().newbyteorder()

    # Replace NaNs and infinite values with zero.
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    return image_data

# Specify the path to your FITS file
fits_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\Code\d101\cropped_DDO 101_H.fits"

# Load and plot the full image.
image = load_fits_image(fits_file_path)
ltf.log_scale_plot(image, f"{galaxy_name} H-alpha plot", "log scale")

# Select a region of interest.
image_section = image[80:250, 80:220]
ltf.log_scale_plot(image_section, f"{galaxy_name} H-alpha plot (selected portion)", "log scale")

# -----------------------------
# Non-Local Means (NLM) Filter function
# -----------------------------
def apply_nl_means_filter(image):
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    nlm_filtered = denoise_nl_means(image, 
                                    h=1.15 * sigma_est, 
                                    fast_mode=True, 
                                    patch_size=5, 
                                    patch_distance=3, 
                                    channel_axis=None)
    return nlm_filtered

# Apply the NLM filter to the raw image_section (this is done once now)
filtered_image = apply_nl_means_filter(image_section)
ltf.log_scale_plot_2_images(image_section, filtered_image, "Original Image", 
                            "NLM Filtered", "Image after NL Filtering", "log scale", 0.1)

# -----------------------------
# Apply Threshold & Noise Injection Methods AFTER NLM Filtering
# -----------------------------
def apply_threshold_with_error_methods_after_nlm(filtered_image, thresholds, filter_type, 
                                                 assumed_pixel_error=1.0, n_bootstrap=1000):
    """
    For each threshold value, this function uses the already NL-filtered image and:
      - For n_bootstrap iterations, applies noise injection (three methods) to the filtered image.
      - Each noisy image is thresholded (pixels <= threshold become NaN) and integrated (sum over valid pixels).
      
    The three methods are:
      1. Bootstrap with Replacement (resampling pixel values from the filtered image)
      2. Gaussian Noise Injection (adding Gaussian noise to the filtered image)
      3. Poisson Noise Injection (scaling up the filtered image, applying Poisson noise, and scaling back)
      
    For each method, the final integrated sum is computed as the mean over iterations,
    and the error is the standard deviation.
    
    A comprehensive table is printed showing, for each threshold, the nominal sum (of the filtered image)
    along with each method's mean integrated sum ± error.
    """
    results_table = []  # To store results for each threshold
    # Loop over each threshold
    for threshold in thresholds:
        bootstrap_sums = []
        gaussian_sums = []
        poisson_sums = []
        
        # Define a scaling factor for Poisson noise to avoid low-lambda problems.
        scale_factor = 1000.0
        
        # Run iterations:
        for _ in range(n_bootstrap):
            # --- Method 1: Bootstrap with Replacement ---
            # Resample pixels from the already filtered image (flatten, then reshape)
            boot_img = np.random.choice(filtered_image.ravel(), 
                                        size=filtered_image.size, 
                                        replace=True).reshape(filtered_image.shape)
            boot_thresh = np.copy(boot_img)
            boot_thresh[boot_thresh <= threshold] = np.nan
            bootstrap_sums.append(np.nansum(boot_thresh))
            
            # --- Method 2: Gaussian Noise Injection ---
            gauss_img = filtered_image + np.random.normal(0, assumed_pixel_error, size=filtered_image.shape)
            gauss_thresh = np.copy(gauss_img)
            gauss_thresh[gauss_thresh <= threshold] = np.nan
            gaussian_sums.append(np.nansum(gauss_thresh))
            
            # --- Method 3: Poisson Noise Injection ---
            lam_scaled = np.clip(filtered_image * scale_factor, 0.1, None)
            poisson_img = np.random.poisson(lam=lam_scaled) / scale_factor
            poisson_thresh = np.copy(poisson_img)
            poisson_thresh[poisson_thresh <= threshold] = np.nan
            poisson_sums.append(np.nansum(poisson_thresh))
        
        # Compute the mean and standard deviation for each method
        boot_mean = np.mean(bootstrap_sums)
        boot_err = np.std(bootstrap_sums)
        gauss_mean = np.mean(gaussian_sums)
        gauss_err = np.std(gaussian_sums)
        poisson_mean = np.mean(poisson_sums)
        poisson_err = np.std(poisson_sums)
        
        # Compute the nominal integrated sum from the filtered image
        nominal_img = np.copy(filtered_image)
        nominal_img[nominal_img <= threshold] = np.nan
        nominal_sum = np.nansum(nominal_img)
        
        # Save results for this threshold
        results_table.append((threshold, nominal_sum,
                              boot_mean, boot_err,
                              gauss_mean, gauss_err,
                              poisson_mean, poisson_err))
        
        # (Optional) Plot histograms for this threshold:
        plt.figure(figsize=(15, 5))
        n_bins = 30
        
        plt.subplot(1, 3, 1)
        plt.hist(bootstrap_sums, bins=n_bins, color='skyblue', edgecolor='black')
        plt.title(f'Bootstrap with Replacement\nThreshold {threshold}')
        plt.xlabel('Integrated sum')
        plt.ylabel('Frequency')
        plt.text(0.95, 0.95, f'Mean: {boot_mean:.2f} ± {boot_err:.2f}', transform=plt.gca().transAxes, 
                 ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.subplot(1, 3, 2)
        plt.hist(gaussian_sums, bins=n_bins, color='lightgreen', edgecolor='black')
        plt.title(f'Gaussian Noise Injection\nThreshold {threshold}')
        plt.xlabel('Integrated sum')
        plt.text(0.95, 0.95, f'Mean: {gauss_mean:.2f} ± {gauss_err:.2f}', transform=plt.gca().transAxes, 
                 ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.subplot(1, 3, 3)
        plt.hist(poisson_sums, bins=n_bins, color='salmon', edgecolor='black')
        plt.title(f'Poisson Noise Injection\nThreshold {threshold}')
        plt.xlabel('Integrated sum')
        plt.text(0.95, 0.95, f'Mean: {poisson_mean:.2f} ± {poisson_err:.2f}', transform=plt.gca().transAxes, 
                 ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    # Print a comprehensive table
    header = ("Threshold", "Nominal Sum", "Bootstrap (Mean ± Error)", 
              "Gaussian (Mean ± Error)", "Poisson (Mean ± Error)")
    line_format = "{:<10} {:>15} {:>30} {:>30} {:>30}"
    print(line_format.format(*header))
    print("-" * 115)
    for row in results_table:
        threshold, nominal_sum, boot_mean, boot_err, gauss_mean, gauss_err, poisson_mean, poisson_err = row
        print(line_format.format(threshold,
                                 f"{nominal_sum:.2f}",
                                 f"{boot_mean:.2f} ± {boot_err:.2f}",
                                 f"{gauss_mean:.2f} ± {gauss_err:.2f}",
                                 f"{poisson_mean:.2f} ± {poisson_err:.2f}"))

# -----------------------------
# Run the Analysis AFTER NLM Filtering
# -----------------------------
# For the Gaussian noise injection, we use the standard deviation of the filtered image.
assumed_error = np.std(filtered_image)
threshold_levels = [4]
# Here n_bootstrap is set to 100 for testing speed; adjust as needed.
apply_threshold_with_error_methods_after_nlm(filtered_image, threshold_levels, 
                                               "NLM Filter (Noise Injection after NL)", 
                                               assumed_pixel_error=assumed_error, n_bootstrap=1000)

import os
import csv
import numpy as np
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt

# -----------------------------
# Basic Function to Load FITS Images
# -----------------------------
def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
    if image_data.dtype.byteorder == '>':
        image_data = image_data.byteswap().newbyteorder()
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    return image_data

# -----------------------------
# Define Paths
# -----------------------------
raw_folder = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha images raw starless"
crop_csv_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\H-alpha crop.csv"
analysis_csv_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\H-alpha analysis.csv"
save_folder = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\Bootstrap Hist"

# -----------------------------
# Read Crop Information from CSV
# -----------------------------
crop_info = []
with open(crop_csv_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        crop_info.append(row)

# -----------------------------
# Prepare the output analysis CSV file
# -----------------------------
with open(analysis_csv_path, "w", newline="") as csvfile_out:
    writer = csv.writer(csvfile_out)
    writer.writerow(["Number", "Y Coordinates", "X Coordinates", 
                     "Actual Value", "Bootstrap with Replacement Mean", "Bootstrap Error"])

    # -----------------------------
    # Process Each Entry in the Crop CSV
    # -----------------------------
    for info in crop_info:
        key = info["Number"].strip()
        galaxy_name = key
        y_range_str = info["Y Coordinates"].strip()
        x_range_str = info["X Coordinates"].strip()
        
        try:
            y_min, y_max = [int(val.strip()) for val in y_range_str.split("-")]
            x_min, x_max = [int(val.strip()) for val in x_range_str.split("-")]
        except Exception as e:
            print(f"Error parsing coordinates for key {key}: {e}")
            continue

        fits_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.fits', '.fit'))]
        selected_file = None
        if key.lower() == "wlm":
            for f in fits_files:
                if "wlmhmrms.fits" in f.lower():
                    selected_file = f
                    break
        else:
            for f in fits_files:
                if key.lower() in f.lower():
                    selected_file = f
                    break
        if selected_file is None:
            print(f"No FITS file found for key {key} in {raw_folder}")
            continue
        
        fits_file_path = os.path.join(raw_folder, selected_file)
        print(f"Processing file: {fits_file_path} (Key: {key}, Galaxy Name: {galaxy_name})")
        full_image = load_fits_image(fits_file_path)
        image_section = full_image[y_min:y_max, x_min:x_max]

        def apply_nl_means_filter(image):
            sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
            nlm_filtered = denoise_nl_means(image, 
                                            h=1.15 * sigma_est, 
                                            fast_mode=True, 
                                            patch_size=5, 
                                            patch_distance=3, 
                                            channel_axis=None)
            return nlm_filtered

        filtered_image = apply_nl_means_filter(image_section)

        def apply_threshold_with_error_methods_after_nlm(filtered_image, thresholds, 
                                                         assumed_pixel_error=1.0, n_bootstrap=1000):
            results_table = []
            for threshold in thresholds:
                bootstrap_sums = []
                for _ in range(n_bootstrap):
                    boot_img = np.random.choice(filtered_image.ravel(), 
                                                size=filtered_image.size, 
                                                replace=True).reshape(filtered_image.shape)
                    boot_thresh = np.copy(boot_img)
                    boot_thresh[boot_thresh <= threshold] = np.nan
                    bootstrap_sums.append(np.nansum(boot_thresh))
                
                boot_mean = np.mean(bootstrap_sums)
                boot_err = np.std(bootstrap_sums)
                nominal_img = np.copy(filtered_image)
                nominal_img[nominal_img <= threshold] = np.nan
                nominal_sum = np.nansum(nominal_img)
                
                results_table.append((threshold, nominal_sum, boot_mean, boot_err))
            return results_table[0], bootstrap_sums

        def plot_and_save_histogram(bootstrap_sums, key, boot_mean, boot_err, save_folder):
            plt.figure(figsize=(8, 6))
            plt.hist(bootstrap_sums, bins=30, alpha=0.75, color='blue', edgecolor='black')
            plt.axvline(boot_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {boot_mean:.2f}')
            plt.axvline(boot_mean - boot_err, color='green', linestyle='dashed', linewidth=1.2, label=f'- Error: {boot_err:.2f}')
            plt.axvline(boot_mean + boot_err, color='green', linestyle='dashed', linewidth=1.2, label=f'+ Error: {boot_err:.2f}')
            plt.title(f'Bootstrap Histogram for Key {key}')
            plt.xlabel('Bootstrap Sum Values')
            plt.ylabel('Frequency')
            plt.legend()
            save_path = os.path.join(save_folder, f"{key}_bootstrap_histogram.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Histogram for key {key} saved at {save_path}")

        threshold_levels = [4]
        assumed_error = np.std(filtered_image)
        analysis_result, bootstrap_sums = apply_threshold_with_error_methods_after_nlm(filtered_image, threshold_levels, 
                                                                        assumed_pixel_error=assumed_error, 
                                                                        n_bootstrap=1000)
        threshold_val, nominal_sum, boot_mean, boot_err = analysis_result
        writer.writerow([key, y_range_str, x_range_str, 
                         f"{nominal_sum:.2f}", f"{boot_mean:.2f}", f"{boot_err:.2f}"])
        print(f"Analysis for key {key} saved.")

        plot_and_save_histogram(bootstrap_sums, key, boot_mean, boot_err, save_folder)

print(f"Analysis complete. Results saved in {analysis_csv_path}")
