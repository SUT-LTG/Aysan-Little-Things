import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import label, center_of_mass
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import numpy.ma as ma
import math

import pandas as pd
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV file containing galaxy parameters
csv_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\centers.csv"
num_farthest_patches = 10
# Function to fetch galaxy data from the CSV based on galaxy name
def get_galaxy_data(galaxy_name, csv_file_path):
    df = pd.read_csv(csv_file_path)
    if galaxy_name not in df["galaxy_name"].values:
        raise ValueError(f"Galaxy '{galaxy_name}' not found in the dataset.")
    return df[df["galaxy_name"] == galaxy_name].iloc[0]

# Request user input for the galaxy name
galaxy_name = input("Enter the galaxy name: ").strip()

try:
    # Retrieve the row for the specified galaxy
    galaxy_data = get_galaxy_data(galaxy_name, csv_file_path)
    
    # Extract parameters from the CSV data
    fits_file_path = galaxy_data["file path"]
    D = galaxy_data["D"]                  # Distance in Mpc
    pixel_scale = galaxy_data["pixel_scale"]
    galaxy_center_x = galaxy_data["galaxy_center_x"]
    galaxy_center_y = galaxy_data["galaxy_center_y"]

    # Convert distance from Mpc to parsecs (1 Mpc = 1,000,000 pc)
    D_pc = D * 1_000_000

    # Display the retrieved parameters
    print(f"Galaxy: {galaxy_name}")
    print(f"FITS File Path: {fits_file_path}")
    print(f"Distance: {D} Mpc ({D_pc} parsecs)")
    print(f"Pixel Scale: {pixel_scale} arcsec/pixel")
    print(f"Galaxy Center: ({galaxy_center_x}, {galaxy_center_y})")


except ValueError as e:
    print(e)



def load_fits_image(file_path):
    """
    Load a FITS image, adjust byte order if necessary, 
    and replace NaNs/infs with zeros.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        if image_data.dtype.byteorder == '>':
            image_data = image_data.byteswap().newbyteorder()
    return np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

# ----------------------------- #
# 1. LOAD IMAGE & SELECT ROI
# ----------------------------- #
image = load_fits_image(fits_file_path)
# Select a region of interest from the image.
image_section = image

# ----------------------------- #
# 2. FILTER & MASK IMAGE
# ----------------------------- #
sigma_est = np.mean(estimate_sigma(image_section, channel_axis=None))
filtered_image = denoise_nl_means(image_section, 
                                  h=1.15 * sigma_est, 
                                  fast_mode=True, 
                                  patch_size=5, 
                                  patch_distance=3, 
                                  channel_axis=None)
# Create a masked version: pixels below 4 become NaN.
masked_image = np.where(filtered_image < 4, np.nan, filtered_image)

# ----------------------------- #
# 3. IDENTIFY CONNECTED PATCHES (Once)
# ----------------------------- #
binary_mask = ~np.isnan(masked_image)
labeled_array, num_features = label(binary_mask)
unique_labels, counts = np.unique(labeled_array, return_counts=True)
# (Note: label 0 is background.)

# ----------------------------- #
# 4. LOOP OVER RADIUS VALUES UNTIL ONE OF THE TOP 10 PATCHES HAS SNR > 3
# ----------------------------- #
# We'll use snr_margin (in pixels) to define the ROI around each patch for SNR calculation.
snr_margin = 10
found = False
selected_Radius = None
selected_SNR = None

# Loop from Radius = 2.3 pc to 5.0 pc in steps of 0.1 pc.
for Radius in np.arange(2.3, 5.1, 0.1):
    # Compute the angular size (arcsec) for the given Radius (using small-angle approximation)
    theta_arcsec = (Radius / D_pc) * 206265
    num_pixels = theta_arcsec / pixel_scale  # effective radius in pixels
    # Define the patch threshold from the expected patch area
    patch_count_threshold = round(np.pi * (num_pixels)**2)
    
    # Select valid patches (ignore label 0)
    valid_labels = unique_labels[(counts >= patch_count_threshold) & (unique_labels != 0)]
    if valid_labels.size == 0:
        print(f"Radius {Radius:.1f} pc: no valid patches (threshold = {patch_count_threshold} pixels)")
        continue
    
    # Compute centers for these valid patches.
    patch_centers_current = {lbl: center_of_mass(binary_mask, labeled_array, lbl)
                             for lbl in valid_labels}
    
    # Compute distances from galaxy center.
    distances_current = {lbl: np.linalg.norm(np.array(coord) - np.array([galaxy_center_y, galaxy_center_x]))
                         for lbl, coord in patch_centers_current.items()}
    
    # Sort patches by distance (farthest first).
    sorted_patches = sorted(distances_current.items(), key=lambda x: x[1], reverse=True)
    
    # Consider the top num_farthest_patches (or all if fewer exist)
    top_patches = sorted_patches[:num_farthest_patches]
    
    # Compute SNR for each top patch.
    top_SNRs = {}
    for lbl, dist_px in top_patches:
        patch_mask = (labeled_array == lbl)
        indices = np.argwhere(patch_mask)
        if indices.size == 0:
            snr_val = np.nan
        else:
            y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
            x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
            # Expand bounding box using snr_margin.
            y_min_roi = max(y_min - snr_margin, 0)
            y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
            x_min_roi = max(x_min - snr_margin, 0)
            x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
            roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            patch_pixels = roi[roi_patch_mask]
            background_pixels = roi[~roi_patch_mask]
            if background_pixels.size > 0:
                background_median = np.median(background_pixels)
                background_std = np.std(background_pixels)
            else:
                background_median = 0
                background_std = sigma_est
            patch_mean = np.mean(patch_pixels)
            snr_val = (patch_mean - background_median) / background_std if background_std > 0 else np.nan
        top_SNRs[lbl] = snr_val
    
    # Check if any of the top patches has SNR > 3.
    if any(snr > 3 for snr in top_SNRs.values() if np.isfinite(snr)):
        selected_Radius = Radius
        # Optionally, pick the first top patch that meets it.
        for lbl, snr_val in top_SNRs.items():
            if np.isfinite(snr_val) and snr_val > 3:
                selected_SNR = snr_val
                break
        print(f"Selected Radius: {Radius:.1f} pc, because top patch (one among the 10) reached SNR = {selected_SNR:.3f}")
        found = True
        break
    else:
        print(f"Radius {Radius:.1f} pc: none of the top {num_farthest_patches} patches reached SNR > 3.")
        
if not found:
    print("No radius in the range produced a top patch with SNR > 3. Using maximum tested Radius = 5.0 pc.")
    selected_Radius = 5.0

# ----------------------------- #
# 5. FINAL ANALYSIS WITH THE SELECTED RADIUS
# ----------------------------- #
theta_arcsec = (selected_Radius / D_pc) * 206265
num_pixels = theta_arcsec / pixel_scale
patch_count_threshold = round(np.pi * (num_pixels)**2)

# Select valid patches based on the new threshold.
valid_labels = unique_labels[(counts >= patch_count_threshold) & (unique_labels != 0)]
patch_centers = {lbl: center_of_mass(binary_mask, labeled_array, lbl) for lbl in valid_labels}
distances = {lbl: np.linalg.norm(np.array(coord) - np.array([galaxy_center_y, galaxy_center_x]))
             for lbl, coord in patch_centers.items()}
sorted_patches = sorted(distances.items(), key=lambda x: x[1], reverse=True)

# Build table data and compute SNR for each top patch.
table_data = []
snr_values_all = []
for lbl, dist_px in sorted_patches[:num_farthest_patches]:
    patch_mask = (labeled_array == lbl)
    indices = np.argwhere(patch_mask)
    if indices.size == 0:
        snr_val = np.nan
    else:
        y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
        x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
        y_min_roi = max(y_min - snr_margin, 0)
        y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
        x_min_roi = max(x_min - snr_margin, 0)
        x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
        roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        patch_pixels = roi[roi_patch_mask]
        background_pixels = roi[~roi_patch_mask]
        if background_pixels.size > 0:
            background_median = np.median(background_pixels)
            background_std = np.std(background_pixels)
        else:
            background_median = 0
            background_std = sigma_est
        patch_mean = np.mean(patch_pixels)
        snr_val = (patch_mean - background_median) / background_std if background_std > 0 else np.nan
    snr_values_all.append(snr_val)
    dist_arcsec = dist_px * pixel_scale
    dist_arcmin = dist_arcsec / 60
    dist_pc = (dist_arcsec / 206265) * D_pc
    table_data.append([lbl, snr_val, dist_px, dist_arcmin, dist_pc])

columns = ["Patch Number", "SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]
patch_table = pd.DataFrame(table_data, columns=columns)
for col in ["SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]:
    patch_table[col] = patch_table[col].apply(lambda x: f"{x:.3f}" if np.isfinite(x) else "NaN")

print(f"\nFor selected Radius {selected_Radius:.1f} pc, using a patch threshold of {patch_count_threshold} pixels:")
print("Top Patches Table (with SNR):")
print(patch_table.to_string(index=False))

# ----------------------------- #
# 6. PLOT PATCH DISTRIBUTION WITH TABLE
# ----------------------------- #
colors = cm.rainbow(np.linspace(0, 1, len(sorted_patches[:num_farthest_patches])))
hex_colors = [to_hex(c) for c in colors]

fig, (ax_image, ax_table_plot) = plt.subplots(1, 2, figsize=(16, 6))
im = ax_image.imshow(masked_image, cmap="gray", origin="lower", interpolation="nearest")
ax_image.scatter(galaxy_center_x, galaxy_center_y, color="blue", marker="x", s=100)
for idx, (lbl, _) in enumerate(sorted_patches[:num_farthest_patches]):
    center_y, center_x = patch_centers[lbl]
    ax_image.scatter(center_x, center_y, color=hex_colors[idx], s=50)
ax_image.set_title(f"Patches Overlay (Top {num_farthest_patches} Farthest)")
fig.colorbar(im, ax=ax_image, fraction=0.046, pad=0.04)
ax_table_plot.axis('tight')
ax_table_plot.axis('off')
table_plot = ax_table_plot.table(cellText=patch_table.values, colLabels=columns, loc='center', cellLoc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)
table_plot.scale(1, 2)
for i in range(len(patch_table)):
    cell = table_plot[(i+1, 0)]
    cell.get_text().set_color(hex_colors[i])
ax_table_plot.set_title(f"Patches Details (with SNR) for patches above {patch_count_threshold} pixels")
plt.tight_layout()
plt.show()

# ----------------------------- #
# 7. VISUALIZE THE SNR ROI AND PATCH VIEWS
#     For each top patch, display three panels:
#     Left: SNR ROI (filtered image with red overlay using snr_margin),
#     Middle: Patch Only (red on white),
#     Right: Background Only (ROI with patch masked out).
# ----------------------------- #
num_top_visualize = len(sorted_patches[:num_farthest_patches])
fig, axes = plt.subplots(num_top_visualize, 3, figsize=(15, 4 * num_top_visualize))
if num_top_visualize == 1:
    axes = np.array([axes])

for i, (lbl, _) in enumerate(sorted_patches[:num_farthest_patches]):
    patch_mask = (labeled_array == lbl)
    indices = np.argwhere(patch_mask)
    if indices.size == 0:
        continue
    y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
    x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
    
    # Expand bounding box using snr_margin.
    y_min_roi = max(y_min - snr_margin, 0)
    y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
    x_min_roi = max(x_min - snr_margin, 0)
    x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
    
    roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
    roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
    
    # Left Panel: SNR ROI with red overlay.
    ax_roi = axes[i, 0]
    ax_roi.imshow(roi, cmap="gray", origin="lower", interpolation="nearest")
    overlay = np.zeros((*roi_patch_mask.shape, 4))
    overlay[roi_patch_mask] = [1, 0, 0, 0.5]
    ax_roi.imshow(overlay, origin="lower", interpolation="nearest")
    ax_roi.set_title(f"SNR ROI with Overlay (Patch {lbl})", fontsize=10)
    ax_roi.axis("off")
    
    # Middle Panel: Patch-Only View (red on white).
    ax_patch_view = axes[i, 1]
    patch_only = np.ones((*roi_patch_mask.shape, 3))
    patch_only[roi_patch_mask] = [1, 0, 0]
    ax_patch_view.imshow(patch_only, origin="lower", interpolation="nearest")
    ax_patch_view.set_title(f"Patch Only (Patch {lbl})", fontsize=10)
    ax_patch_view.axis("off")
    
    # Right Panel: Background-Only View (patch masked out).
    ax_bg_view = axes[i, 2]
    background = ma.masked_where(roi_patch_mask, roi)
    cmap_bg = plt.cm.gray.copy()
    cmap_bg.set_bad(color="white")
    ax_bg_view.imshow(background, cmap=cmap_bg, origin="lower", interpolation="nearest")
    ax_bg_view.set_title(f"Background for SNR (Patch {lbl})", fontsize=10)
    ax_bg_view.axis("off")

plt.tight_layout()
plt.show()

#===========================================================================================================================================#


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import label, center_of_mass
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import numpy.ma as ma
import math

# ----------------------------- #
# PARAMETERS & HELPER FUNCTIONS #
# ----------------------------- #
galaxy_name = "DDO 69"
galaxy_center_x = 182
galaxy_center_y = 154

# Path to the FITS file
fits_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\Cropped H-alpha\cropped_DDO 69_H.fits"

# Known values
D = 0.8              # Distance in Mega Parsecs (Mpc)
pixel_scale = 0.49   # Arcseconds per pixel
num_farthest_patches = 10  # Maximum number of patches to show in final output

# Convert distance to parsecs (1 Mpc = 1,000,000 pc)
D_pc = D * 1_000_000

def load_fits_image(file_path):
    """
    Load a FITS image, adjust the byte order if necessary, 
    and replace NaNs/infs with zeros.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        if image_data.dtype.byteorder == '>':
            image_data = image_data.byteswap().newbyteorder()
    return np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

# ----------------------------- #
# 1. LOAD IMAGE & SELECT ROI
# ----------------------------- #
image = load_fits_image(fits_file_path)
# Select a region of interest (ROI) from the image.
image_section = image[0:282, 1:460]

# ----------------------------- #
# 2. FILTER & MASK IMAGE
# ----------------------------- #
sigma_est = np.mean(estimate_sigma(image_section, channel_axis=None))
filtered_image = denoise_nl_means(image_section, 
                                  h=1.15 * sigma_est, 
                                  fast_mode=True, 
                                  patch_size=5, 
                                  patch_distance=3, 
                                  channel_axis=None)
# Create a masked version of the filtered image where low-intensity pixels become NaN.
masked_image = np.where(filtered_image < 4, np.nan, filtered_image)

# ----------------------------- #
# 3. IDENTIFY CONNECTED PATCHES (Once)
# ----------------------------- #
# We perform connected-component detection before looping over Radius.
binary_mask = ~np.isnan(masked_image)
labeled_array, num_features = label(binary_mask)
unique_labels, counts = np.unique(labeled_array, return_counts=True)

# ----------------------------- #
# 4. LOOP OVER RADIUS VALUES AND CHECK FARTHEST PATCH SNR
# ----------------------------- #
# We will vary Radius from 2.3 to 5.0 (step 0.1) until the farthest patch has SNR > 3.
snr_margin = 10   # Margin (in pixels) for SNR ROI extraction
found = False

for Radius in np.arange(2.3, 10, 0.1):
    # Calculate the expected angular size (arcsec) and convert to pixels.
    theta_arcsec = (Radius / D_pc) * 206265
    num_pixels = theta_arcsec / pixel_scale

    # Define the patch threshold (expected area in pixel count).
    patch_count_threshold = round(np.pi * (num_pixels)**2)
    
    # Select patches (ignore background label 0).
    valid_labels = unique_labels[(counts >= patch_count_threshold) & (unique_labels != 0)]
    if valid_labels.size == 0:
        print(f"Radius {Radius:.1f} produced no valid patches.")
        continue
    
    # Compute centers for valid patches.
    patch_centers = {lbl: center_of_mass(binary_mask, labeled_array, lbl) for lbl in valid_labels}
    
    # Calculate distances (in pixels) from the galaxy center.
    distances = {
        lbl: np.linalg.norm(np.array(coord) - np.array([galaxy_center_y, galaxy_center_x]))
        for lbl, coord in patch_centers.items()
    }
    # Sort patches by distance (farthest first).
    sorted_patches = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_patches) == 0:
        print(f"Radius {Radius:.1f} produced no patches after sorting.")
        continue

    # Get the farthest patch (first in sorted list)
    farthest_lbl, farthest_dist = sorted_patches[0]
    patch_mask = (labeled_array == farthest_lbl)
    indices = np.argwhere(patch_mask)
    if indices.size == 0:
        current_snr = np.nan
    else:
        y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
        x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
        # Expand the bounding box using snr_margin.
        y_min_roi = max(y_min - snr_margin, 0)
        y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
        x_min_roi = max(x_min - snr_margin, 0)
        x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
        roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]

        # Separate patch and background pixel intensities.
        patch_pixels = roi[roi_patch_mask]
        background_pixels = roi[~roi_patch_mask]
        if background_pixels.size > 0:
            background_median = np.median(background_pixels)
            background_std = np.std(background_pixels)
        else:
            background_median = 0
            background_std = sigma_est
        patch_mean = np.mean(patch_pixels)
        current_snr = (patch_mean - background_median) / background_std if background_std > 0 else np.nan

    print(f"Radius {Radius:.1f}: Farthest patch (label {farthest_lbl}) SNR = {current_snr:.3f}")
    if current_snr > 3:
        found = True
        break

if not found:
    print("No radius found with farthest patch SNR > 3 in the given range.")
    # Optionally, you might decide to proceed with the highest Radius tested.
else:
    print(f"Selected Radius: {Radius:.1f} with farthest patch SNR = {current_snr:.3f}")

# ----------------------------- #
# 5. FINAL ANALYSIS WITH SELECTED RADIUS
# ----------------------------- #
if found:  # Proceed only if condition was met.
    # Recompute parameters using the selected Radius.
    theta_arcsec = (Radius / D_pc) * 206265
    num_pixels = theta_arcsec / pixel_scale
    patch_count_threshold = round(np.pi * (num_pixels)**2)
    
    valid_labels = unique_labels[(counts >= patch_count_threshold) & (unique_labels != 0)]
    patch_centers = {lbl: center_of_mass(binary_mask, labeled_array, lbl) for lbl in valid_labels}
    distances = {
        lbl: np.linalg.norm(np.array(coord) - np.array([galaxy_center_y, galaxy_center_x]))
        for lbl, coord in patch_centers.items()
    }
    sorted_patches = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    
    # Build table data and recompute SNR for each top patch.
    table_data = []
    snr_values_all = []
    for lbl, dist_px in sorted_patches[:num_farthest_patches]:
        patch_mask = (labeled_array == lbl)
        indices = np.argwhere(patch_mask)
        if indices.size == 0:
            snr_val = np.nan
        else:
            y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
            x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
            y_min_roi = max(y_min - snr_margin, 0)
            y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
            x_min_roi = max(x_min - snr_margin, 0)
            x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
            roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            patch_pixels = roi[roi_patch_mask]
            background_pixels = roi[~roi_patch_mask]
            if background_pixels.size > 0:
                background_median = np.median(background_pixels)
                background_std = np.std(background_pixels)
            else:
                background_median = 0
                background_std = sigma_est
            patch_mean = np.mean(patch_pixels)
            snr_val = (patch_mean - background_median) / background_std if background_std > 0 else np.nan
        snr_values_all.append(snr_val)
        dist_arcsec = dist_px * pixel_scale
        dist_arcmin = dist_arcsec / 60
        dist_pc = (dist_arcsec / 206265) * D_pc
        table_data.append([lbl, snr_val, dist_px, dist_arcmin, dist_pc])
    
    columns = ["Patch Number", "SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]
    patch_table = pd.DataFrame(table_data, columns=columns)
    for col in ["SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]:
        patch_table[col] = patch_table[col].apply(lambda x: f"{x:.3f}" if np.isfinite(x) else "NaN")
    
    print("\nTop Patches Table (with SNR):")
    print(patch_table.to_string(index=False))
    
    # ----------------------------- #
    # 6. VISUALIZE PATCH DISTRIBUTION WITH TABLE
    # ----------------------------- #
    colors = cm.rainbow(np.linspace(0, 1, len(sorted_patches[:num_farthest_patches])))
    hex_colors = [to_hex(c) for c in colors]
    
    fig, (ax_image, ax_table_plot) = plt.subplots(1, 2, figsize=(16, 6))
    im = ax_image.imshow(masked_image, cmap="gray", origin="lower", interpolation="nearest")
    ax_image.scatter(galaxy_center_x, galaxy_center_y, color="blue", marker="x", s=100)
    for idx, (lbl, _) in enumerate(sorted_patches[:num_farthest_patches]):
        center_y, center_x = patch_centers[lbl]
        ax_image.scatter(center_x, center_y, color=hex_colors[idx], s=50)
    ax_image.set_title(f"Patches Overlay (Top {num_farthest_patches} Farthest)")
    fig.colorbar(im, ax=ax_image, fraction=0.046, pad=0.04)
    
    ax_table_plot.axis('tight')
    ax_table_plot.axis('off')
    table_plot = ax_table_plot.table(cellText=patch_table.values, colLabels=columns, loc='center', cellLoc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1, 2)
    for i in range(len(patch_table)):
        cell = table_plot[(i+1, 0)]
        cell.get_text().set_color(hex_colors[i])
    ax_table_plot.set_title(f"Patches Details (With SNR) for patches above {patch_count_threshold} pixels")
    
    plt.tight_layout()
    plt.show()
    
    # ----------------------------- #
    # 7. VISUALIZE THE SNR ROI AND PATCH VIEWS
    #     For each top patch, display three panels showing:
    #         Left: SNR ROI (filtered image with red overlay) using snr_margin,
    #         Middle: Patch Only (red on white),
    #         Right: Background Only (ROI with patch masked out).
    # ----------------------------- #
    num_top_visualize = len(sorted_patches[:num_farthest_patches])
    fig, axes = plt.subplots(num_top_visualize, 3, figsize=(15, 4 * num_top_visualize))
    if num_top_visualize == 1:
        axes = np.array([axes])
    
    for i, (lbl, _) in enumerate(sorted_patches[:num_farthest_patches]):
        patch_mask = (labeled_array == lbl)
        indices = np.argwhere(patch_mask)
        if indices.size == 0:
            continue
        y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
        x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
    
        # Expand bounding box using snr_margin.
        y_min_roi = max(y_min - snr_margin, 0)
        y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
        x_min_roi = max(x_min - snr_margin, 0)
        x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
    
        roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
    
        # Left Panel: SNR ROI with red overlay.
        ax_roi = axes[i, 0]
        ax_roi.imshow(roi, cmap="gray", origin="lower", interpolation="nearest")
        overlay = np.zeros((*roi_patch_mask.shape, 4))
        overlay[roi_patch_mask] = [1, 0, 0, 0.5]
        ax_roi.imshow(overlay, origin="lower", interpolation="nearest")
        ax_roi.set_title(f"SNR ROI with Overlay (Patch {lbl})", fontsize=10)
        ax_roi.axis("off")
    
        # Middle Panel: Patch-Only View (red on white).
        ax_patch_view = axes[i, 1]
        patch_only = np.ones((*roi_patch_mask.shape, 3))
        patch_only[roi_patch_mask] = [1, 0, 0]
        ax_patch_view.imshow(patch_only, origin="lower", interpolation="nearest")
        ax_patch_view.set_title(f"Patch Only (Patch {lbl})", fontsize=10)
        ax_patch_view.axis("off")
    
        # Right Panel: Background-Only View (patch masked out).
        ax_bg_view = axes[i, 2]
        background = ma.masked_where(roi_patch_mask, roi)
        cmap_bg = plt.cm.gray.copy()
        cmap_bg.set_bad(color="white")
        ax_bg_view.imshow(background, cmap=cmap_bg, origin="lower", interpolation="nearest")
        ax_bg_view.set_title(f"Background for SNR (Patch {lbl})", fontsize=10)
        ax_bg_view.axis("off")
    
    plt.tight_layout()
    plt.show()
#===========================================================================================================================================#

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import label, center_of_mass
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import numpy.ma as ma
import math

# ----------------------------- #
# PARAMETERS & HELPER FUNCTIONS #
# ----------------------------- #
galaxy_name = "DDO 69"
galaxy_center_x = 182
galaxy_center_y = 154

# Path to the FITS file
fits_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\Cropped H-alpha\cropped_DDO 69_H.fits"

# Known values
D = 0.8              # Distance in Mega Parsecs
pixel_scale = 0.49   # Arcseconds per pixel
Radius = 4         # pc (characteristic patch radius)
num_farthest_patches = 10  # Maximum number of patches to show

# Convert distance to parsecs (1 Mpc = 1,000,000 pc)
D_pc = D * 1_000_000

# Compute the angular size (arcsec) using the small-angle approximation and convert to pixels.
theta_arcsec = (Radius / D_pc) * 206265
num_pixels = theta_arcsec / pixel_scale

# Define patch threshold (in pixel count) from the expected patch area.
patch_count_threshold = round(np.pi * (num_pixels)**2)
print(f"Estimated patch diameter in pixels: {num_pixels:.2f}")

def load_fits_image(file_path):
    """
    Load a FITS image, adjust the byte order if necessary, 
    and replace NaNs/infs with zeros.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        if image_data.dtype.byteorder == '>':
            image_data = image_data.byteswap().newbyteorder()
    return np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

# ----------------------------- #
# 1. LOAD IMAGE & SELECT ROI
# ----------------------------- #
image = load_fits_image(fits_file_path)
# Select a region of interest (ROI) from the image.
image_section = image[0:282, 1:460]

# ----------------------------- #
# 2. FILTER & MASK IMAGE
# ----------------------------- #
sigma_est = np.mean(estimate_sigma(image_section, channel_axis=None))
filtered_image = denoise_nl_means(image_section, 
                                  h=1.15 * sigma_est, 
                                  fast_mode=True, 
                                  patch_size=5, 
                                  patch_distance=3, 
                                  channel_axis=None)
# Create a masked version of the filtered image where low-intensity pixels become NaN.
masked_image = np.where(filtered_image < 4, np.nan, filtered_image)

# ----------------------------- #
# 3. IDENTIFY CONNECTED PATCHES
# ----------------------------- #
binary_mask = ~np.isnan(masked_image)
labeled_array, num_features = label(binary_mask)
unique_labels, counts = np.unique(labeled_array, return_counts=True)

# Only consider patches with a sufficient number of pixels (ignore the background label 0).
valid_labels = unique_labels[(counts >= patch_count_threshold) & (unique_labels != 0)]

# Compute centers of the patches.
patch_centers = {lbl: center_of_mass(binary_mask, labeled_array, lbl) for lbl in valid_labels}

# Calculate distances (in pixels) from the galaxy center.
distances = {
    lbl: np.linalg.norm(np.array(coord) - np.array([galaxy_center_y, galaxy_center_x]))
    for lbl, coord in patch_centers.items()
}

# ----------------------------- #
# 4. FIND TOP PATCHES, COMPUTE SNR & CREATE TABLE
# ----------------------------- #
# Sort patches by distance (farthest first)
sorted_patches = sorted(distances.items(), key=lambda x: x[1], reverse=True)
top_patches = sorted_patches[:num_farthest_patches] if len(sorted_patches) >= num_farthest_patches else sorted_patches

# Define the margin (in pixels) for the SNR ROI.
snr_margin = 10

table_data = []
snr_values = []

for lbl, dist_px in top_patches:
    patch_mask = (labeled_array == lbl)
    indices = np.argwhere(patch_mask)
    if indices.size == 0:
        snr = np.nan
    else:
        # Determine the tight bounding box for the patch.
        y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
        x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
        # Expand the bounding box using the snr_margin.
        y_min_roi = max(y_min - snr_margin, 0)
        y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
        x_min_roi = max(x_min - snr_margin, 0)
        x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)
        roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
        
        # Separate the patch and background pixel intensities.
        patch_pixels = roi[roi_patch_mask]
        background_pixels = roi[~roi_patch_mask]
        
        if background_pixels.size > 0:
            background_median = np.median(background_pixels)
            background_std = np.std(background_pixels)
        else:
            background_median = 0
            background_std = sigma_est
        
        patch_mean = np.mean(patch_pixels)
        snr = (patch_mean - background_median) / background_std if background_std > 0 else np.nan

    snr_values.append(snr)
    
    # Convert distance metrics.
    dist_arcsec = dist_px * pixel_scale
    dist_arcmin = dist_arcsec / 60
    dist_pc = (dist_arcsec / 206265) * D_pc
    
    table_data.append([lbl, snr, dist_px, dist_arcmin, dist_pc])

columns = ["Patch Number", "SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]
patch_table = pd.DataFrame(table_data, columns=columns)

# Format numeric columns to three decimals.
for col in ["SNR", "Distance (px)", "Distance (arcminutes)", "Distance (pc)"]:
    patch_table[col] = patch_table[col].apply(lambda x: f"{x:.3f}" if np.isfinite(x) else "NaN")

print(f"\nA patch with a radius {Radius} pc corresponds to approximately a patch with {patch_count_threshold:.2f} pixels.\n")
print("Top Patches Table (with SNR):\n")
print(patch_table.to_string(index=False))

# ----------------------------- #
# 5. PLOT PATCH DISTRIBUTION WITH TABLE
# ----------------------------- #
colors = cm.rainbow(np.linspace(0, 1, len(top_patches)))
hex_colors = [to_hex(c) for c in colors]

fig, (ax_image, ax_table_plot) = plt.subplots(1, 2, figsize=(16, 6))
im = ax_image.imshow(masked_image, cmap="gray", origin="lower", interpolation="nearest")
ax_image.scatter(galaxy_center_x, galaxy_center_y, color="blue", marker="x", s=100)

for idx, (lbl, _) in enumerate(top_patches):
    center_y, center_x = patch_centers[lbl]
    ax_image.scatter(center_x, center_y, color=hex_colors[idx], s=50)
ax_image.set_title(f"Patches Overlay (Top {len(top_patches)} Farthest)")
fig.colorbar(im, ax=ax_image, fraction=0.046, pad=0.04)

ax_table_plot.axis('tight')
ax_table_plot.axis('off')
table_plot = ax_table_plot.table(cellText=patch_table.values, colLabels=columns, loc='center', cellLoc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)
table_plot.scale(1, 2)
# Color-code the "Patch Number" column to match the overlay.
for i in range(len(patch_table)):
    cell = table_plot[(i+1, 0)]
    cell.get_text().set_color(hex_colors[i])
ax_table_plot.set_title(f"Patches Details (With SNR) for patches above {patch_count_threshold} pixels")

plt.tight_layout()
plt.show()

# ----------------------------- #
# 6. VISUALIZE THE SNR ROI AND PATCH VIEWS
#     For each top patch, display three panels showing:
#         Left: SNR ROI (filtered image with red overlay) using the snr_margin,
#         Middle: Patch Only (red on white),
#         Right: Background Only (ROI with patch masked out).
# ----------------------------- #
num_top_visualize = len(top_patches)
fig, axes = plt.subplots(num_top_visualize, 3, figsize=(15, 4 * num_top_visualize))
if num_top_visualize == 1:
    axes = np.array([axes])

for i, (lbl, _) in enumerate(top_patches):
    patch_mask = (labeled_array == lbl)
    indices = np.argwhere(patch_mask)
    if indices.size == 0:
        continue
    y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
    x_min, x_max = indices[:, 1].min(), indices[:, 1].max()

    # Expand bounding box using snr_margin.
    y_min_roi = max(y_min - snr_margin, 0)
    y_max_roi = min(y_max + snr_margin, filtered_image.shape[0] - 1)
    x_min_roi = max(x_min - snr_margin, 0)
    x_max_roi = min(x_max + snr_margin, filtered_image.shape[1] - 1)

    roi = filtered_image[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
    roi_patch_mask = patch_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]

    # Left Panel: SNR ROI with red overlay.
    ax_roi = axes[i, 0]
    ax_roi.imshow(roi, cmap="gray", origin="lower", interpolation="nearest")
    overlay = np.zeros((*roi_patch_mask.shape, 4))
    overlay[roi_patch_mask] = [1, 0, 0, 0.5]
    ax_roi.imshow(overlay, origin="lower", interpolation="nearest")
    ax_roi.set_title(f"SNR ROI with Overlay (Patch {lbl})", fontsize=10)
    ax_roi.axis("off")

    # Middle Panel: Patch-Only View (red over white).
    ax_patch_view = axes[i, 1]
    patch_only = np.ones((*roi_patch_mask.shape, 3))
    patch_only[roi_patch_mask] = [1, 0, 0]
    ax_patch_view.imshow(patch_only, origin="lower", interpolation="nearest")
    ax_patch_view.set_title(f"Patch Only (Patch {lbl})", fontsize=10)
    ax_patch_view.axis("off")

    # Right Panel: Background-Only View (patch masked out).
    ax_bg_view = axes[i, 2]
    background = ma.masked_where(roi_patch_mask, roi)
    cmap_bg = plt.cm.gray.copy()
    cmap_bg.set_bad(color="white")
    ax_bg_view.imshow(background, cmap=cmap_bg, origin="lower", interpolation="nearest")
    ax_bg_view.set_title(f"Background for SNR (Patch {lbl})", fontsize=10)
    ax_bg_view.axis("off")

plt.tight_layout()
plt.show()


# -----------------------------
# 5. EXTRACT CONTOURS AT LEVELS 2, 3, 4, 5
# -----------------------------
contour_levels = [2, 3, 4, 5]
contours = {}
for level in contour_levels:
    # Use the filtered image for contour extraction.
    contours[level] = ltf.contour_lines_coordinates(filtered_image, 3, [level])

plt.figure(figsize=(8, 6))
plt.imshow(filtered_image, cmap="gray", origin="lower", interpolation="nearest")
colors = ['red', 'green', 'blue', 'yellow']
for idx, level in enumerate(contour_levels):
    x_points, y_points = contours[level]
    # Check that the arrays are not empty:
    if x_points.size > 0 and y_points.size > 0:
        plt.plot(x_points, y_points, color=colors[idx], label=f"Level {level}")
plt.title("Contour Levels on Filtered Image")
plt.legend()
plt.colorbar()
plt.show()

# -----------------------------
# 6. ELLIPSE FITTING TO LEVEL 5 CONTOUR & AXES VISUALIZATION
# -----------------------------
# Use level 5 contour points.
x_points, y_points = contours[5]
if x_points.size > 0 and y_points.size > 0:
    # Initial guess: center from mean values, approximate semi-axes from spans, theta=0
    init_xc = np.mean(x_points)
    init_yc = np.mean(y_points)
    init_a = (np.max(x_points) - np.min(x_points)) / 2
    init_b = (np.max(y_points) - np.min(y_points)) / 2
    init_theta = 0
    initial_guess = [init_xc, init_yc, init_a, init_b, init_theta]
    
    # Fit the ellipse so the model residual is near zero.
    popt, _ = curve_fit(ellipse_model, (x_points, y_points), np.zeros_like(x_points),
                        p0=initial_guess)
    xc, yc, a, b, theta = popt
    print("Fitted ellipse parameters:")
    print("Center: ({:.2f}, {:.2f})".format(xc, yc))
    print("Semi-axes: a = {:.2f}, b = {:.2f}".format(a, b))
    print("Rotation (degrees): {:.2f}".format(np.degrees(theta)))
    
    # Determine which axis is the semi‑major: the larger value.
    if a >= b:
        semi_major = a
        semi_minor = b
        angle_for_axes = theta
    else:
        semi_major = b
        semi_minor = a
        angle_for_axes = theta + np.pi/2  # Adjust angle if swapped.

    # Plot the level 5 contour, fitted ellipse, and display the axes.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(filtered_image, cmap="gray", origin="lower", interpolation="nearest")
    ax.plot(x_points, y_points, 'r-', label="Level 5 Contour")
    
    # Overlay the fitted ellipse (using the fitted a and b as semi-axes):
    ellipse_patch = Ellipse(xy=(xc, yc), width=2*a, height=2*b,
                            angle=np.degrees(theta),
                            edgecolor='cyan', facecolor='none', linewidth=2,
                            label="Fitted Ellipse")
    ax.add_patch(ellipse_patch)
    
    # Draw the semi-major axis (red):
    x_major_end = xc + semi_major * np.cos(angle_for_axes)
    y_major_end = yc + semi_major * np.sin(angle_for_axes)
    ax.plot([xc, x_major_end], [yc, y_major_end], 'r-', lw=2, label="Semi-major Axis")
    ax.plot([xc, xc - semi_major * np.cos(angle_for_axes)],
            [yc, yc - semi_major * np.sin(angle_for_axes)], 'r-', lw=2)
    
    # Draw the semi-minor axis (green):
    perp_angle = angle_for_axes + np.pi/2
    x_minor_end = xc + semi_minor * np.cos(perp_angle)
    y_minor_end = yc + semi_minor * np.sin(perp_angle)
    ax.plot([xc, x_minor_end], [yc, y_minor_end], 'g-', lw=2, label="Semi-minor Axis")
    ax.plot([xc, xc - semi_minor * np.cos(perp_angle)],
            [yc, yc - semi_minor * np.sin(perp_angle)], 'g-', lw=2)
    
    ax.set_title("Ellipse Fit to Level 5 Contour with Semi-Axes")
    ax.legend()
    plt.show()
else:
    print("No contour points found for level 5! Unable to fit ellipse.")
