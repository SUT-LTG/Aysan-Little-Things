
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
import little_things_functions as ltf
import time 
t1 = time.time()

threshold = 4
# -----------------------------
# Load the FITS image
# -----------------------------
def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    # Handle byte order if needed.
    if image_data.dtype.byteorder == '>':
        image_data = image_data.byteswap().newbyteorder()

    # Replace NaNs and infinite values with zero.
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    return image_data

# Specify the path to your FITS file.
fits_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\Data\DDO 133\d133hmrms.fits"

# Load the full image and then select a region of interest.
image = load_fits_image(fits_file_path)
image_section = image[0:778, 150: 750]

# Optionally, check the image using your log scale plot (if desired)
# ltf.log_scale_plot(image_section, "DDO 87 H-alpha plot (selected portion)", "log scale")

# -----------------------------
# Define NLM Filter with variable parameters.
# -----------------------------
def apply_nl_means_filter(image, h_factor, patch_size, patch_distance):
    # Estimate the noise standard deviation from the image.
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    
    # Apply the NLM denoising algorithm.
    filtered = denoise_nl_means(
        image,
        h=h_factor * sigma_est,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None
    )
    return filtered

# -----------------------------
# Set up parameter ranges for brute-force exploration.
# -----------------------------
# 10 values for h_factor between 0.8 and 2.5.
h_values = np.linspace(0.8, 2.5, 10)

# Patch sizes from 2 to 10 (inclusive) -> 9 different values.
patch_sizes = np.arange(2, 11)

# Patch distances from 2 to 10 (inclusive) -> 9 different values.
patch_distances = np.arange(2, 11)

sum_values = []  # This will store the sum of pixel intensities for each parameter combination.
params_list = []  # (optional) save parameters used for each iteration.

total_iterations = len(h_values) * len(patch_sizes) * len(patch_distances)
counter = 0

# -----------------------------
# Brute-force loop over all parameter combinations.
# -----------------------------
for h in h_values:
    for patch in patch_sizes:
        for distance in patch_distances:
            # Apply the NLM filter with current parameters.
            filtered = apply_nl_means_filter(image_section, h, patch, distance)
            # Create a mask: set any pixel value less than 4 to NaN.
            masked = np.where(filtered < threshold, np.nan, filtered)
            # Compute the sum ignoring the NaN values.
            sum_val = np.nansum(masked)
            sum_values.append(sum_val)
            params_list.append((h, patch, distance))
            
            counter += 1
            if counter % 100 == 0:
                print(f"Completed {counter} / {total_iterations} iterations.")

# -----------------------------
# Plot a histogram of the calculated sums.
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(sum_values, bins=100, edgecolor='black')
plt.xlabel("Sum of pixel intensities (pixels >= 4)")
plt.ylabel("Frequency")
plt.title("Histogram of Summed Pixel Intensities Across NLM Filter Parameters")
plt.show()

print(f"The Median Sum Value for {threshold} is {np.median(sum_values)} +/- {np.std(sum_values)}")

t2 = time.time()

print(t2 - t1)


import random

# -----------------------------
# Show 10 Randomly Filtered Images in a Grid
# -----------------------------
def show_random_filtered_images(image_section, params_list, threshold, num_images=4):
    # Select 10 random parameter combinations from the params_list
    random_params = random.sample(params_list, num_images)
    
    plt.figure(figsize=(15, 10))
    
    for i, params in enumerate(random_params, 1):
        h, patch, distance = params
        filtered = apply_nl_means_filter(image_section, h, patch, distance)
        
        # Create a mask: set any pixel value less than the threshold to NaN
        masked = np.where(filtered < threshold, np.nan, filtered)
        
        plt.subplot(2, 2, i)  # Create a grid with 2 rows and 5 columns
        plt.imshow(masked, cmap='gray', interpolation='none')
        plt.title(f"h={h:.2f}, p={patch}, d={distance}")
        
        # Add x and y labels and keep the axis ticks
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.show()


# Call the function with your image section and parameters
show_random_filtered_images(image_section, params_list, threshold)

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Create a histogram
# -----------------------------
hist, bin_edges = np.histogram(sum_values, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Find the maximum value and calculate half maximum
max_value = max(hist)
half_max = max_value / 2

# -----------------------------
# Locate Intersection Points
# -----------------------------
# Find indices where histogram crosses half maximum
above_half_max = hist >= half_max  # Boolean array for values >= half maximum
edges_indices = np.where(np.diff(above_half_max.astype(int)) != 0)[0]  # Transition points

# Interpolate to find the precise values of bin centers at intersections
intersection_points = []
for idx in edges_indices:
    low_bin, high_bin = bin_centers[idx], bin_centers[idx + 1]
    low_val, high_val = hist[idx], hist[idx + 1]
    slope = (high_val - low_val) / (high_bin - low_bin)  # Linear interpolation
    intercept = low_val - slope * low_bin
    x_intersect = (half_max - intercept) / slope
    intersection_points.append(x_intersect)

# -----------------------------
# Plot the histogram and mark intersection points
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.7, color='blue', edgecolor='black', label='Histogram')
plt.axhline(half_max, color='red', linestyle='dashed', label=f'Half Maximum ({half_max:.2f})')
for point in intersection_points:
    plt.axvline(point, color='green', linestyle='dashed', label=f'Intersection at {point:.2f}')

plt.xlabel("Sum of pixel intensities (pixels >= 4)")
plt.ylabel("Frequency")
plt.title("Histogram with Half Maximum and Intersection Points")
plt.legend()
plt.show()

# Output intersection points
print(f"Intersection points at half maximum: {intersection_points}")
