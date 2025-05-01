import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.transform import AffineTransform, warp
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch

# Import your custom functions (for example, open_fits).
import little_things_functions as ltf

# ==============================================================================
#       INTERACTIVE ALIGNMENT FUNCTIONS
# ==============================================================================

def V_and_Halpha_alignment_interactive(
    galaxy_name, image1_path, image2_path, output_dir, alpha1=0.5, alpha2=0.5
):
    """
    Opens two FITS images (V and H-alpha) and allows the user to interactively
    click on at least 4 corresponding points in each image. A magnifier follows
    the mouse to help with precise clicking. Once the 'Proceed' button is clicked,
    the selected coordinates are returned.
    """
    norm = ImageNormalize(vmin=0.0, stretch=LogStretch())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open FITS files using your custom module.
    V_image = ltf.open_fits(image1_path)
    H_image = ltf.open_fits(image2_path)

    # Create figure with two subplots.
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    coords_V = []
    coords_H = []

    im_V = axes[0].imshow(V_image, cmap="gray", norm=norm, origin="lower")
    axes[0].set_title(f"{galaxy_name} V Image")
    im_H = axes[1].imshow(H_image, cmap="gray", norm=norm, origin="lower")
    axes[1].set_title(f"{galaxy_name} H-alpha Image")

    plt.subplots_adjust(bottom=0.2)
    plt.suptitle("Click on corresponding points in each image.\n"
                 "Select at least 4 points per image in the same order.")

    selected_points_text = fig.text(0.5, 0.05, "", ha="center")
    num_points_V = 0
    num_points_H = 0

    # Magnifier settings.
    rect_size = 20

    # Initialize magnifier for V image.
    axins_V = inset_axes(axes[0], width="20%", height="20%", loc="upper right", borderpad=1)
    axins_V.imshow(V_image, norm=norm, origin="lower")
    axins_V.axis("off")
    rect_V = Rectangle((0, 0), rect_size, rect_size, edgecolor="red", facecolor="none", linewidth=1)
    axes[0].add_patch(rect_V)
    h_line_V = axins_V.axhline(color='yellow', linewidth=0.5)
    v_line_V = axins_V.axvline(color='yellow', linewidth=0.5)

    # Initialize magnifier for H-alpha image.
    axins_H = inset_axes(axes[1], width="20%", height="20%", loc="upper right", borderpad=1)
    axins_H.imshow(H_image, norm=norm, origin="lower")
    axins_H.axis("off")
    rect_H = Rectangle((0, 0), rect_size, rect_size, edgecolor="red", facecolor="none", linewidth=1)
    axes[1].add_patch(rect_H)
    h_line_H = axins_H.axhline(color='yellow', linewidth=0.5)
    v_line_H = axins_H.axvline(color='yellow', linewidth=0.5)

    def update_text():
        selected_points_text.set_text(
            f"Points selected - V Image: {num_points_V}, H-alpha Image: {num_points_H}"
        )
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal num_points_V, num_points_H
        if event.inaxes == axes[0]:
            coords_V.append([event.xdata, event.ydata])
            axes[0].plot(event.xdata, event.ydata, "rx")
            num_points_V += 1
        elif event.inaxes == axes[1]:
            coords_H.append([event.xdata, event.ydata])
            axes[1].plot(event.xdata, event.ydata, "rx")
            num_points_H += 1
        else:
            return
        update_text()
        fig.canvas.draw_idle()

    def on_mouse_move(event):
        if event.inaxes in [axes[0], axes[1]]:
            xdata, ydata = event.xdata, event.ydata
            if xdata is not None and ydata is not None:
                size = rect_size / 2
                x1 = xdata - size
                x2 = xdata + size
                y1 = ydata - size
                y2 = ydata + size

                if event.inaxes == axes[0]:
                    axins = axins_V
                    rect = rect_V
                    img_shape = V_image.shape
                    h_line = h_line_V
                    v_line = v_line_V
                else:
                    axins = axins_H
                    rect = rect_H
                    img_shape = H_image.shape
                    h_line = h_line_H
                    v_line = v_line_H

                x1 = max(x1, 0)
                x2 = min(x2, img_shape[1])
                y1 = max(y1, 0)
                y2 = min(y2, img_shape[0])

                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.invert_yaxis()

                rect.set_xy((x1, y1))
                rect.set_width(x2 - x1)
                rect.set_height(y2 - y1)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                h_line.set_ydata([center_y])
                v_line.set_xdata([center_x])

                fig.canvas.draw_idle()

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    ax_proceed = plt.axes([0.45, 0.01, 0.1, 0.05])
    btn_proceed = Button(ax_proceed, "Proceed")

    def proceed(event):
        if len(coords_V) >= 4 and len(coords_V) == len(coords_H):
            plt.close()
        else:
            print("Please select at least 4 corresponding points on both images.")
            print("Ensure that the number of points matches between images.")

    btn_proceed.on_clicked(proceed)
    plt.show()

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_move)

    star_coords_V = np.array(coords_V)
    star_coords_H = np.array(coords_H)

    return star_coords_V, star_coords_H

def V_and_Halpha_alignment(
    galaxy_name,
    image1_path,
    image2_path,
    star_coords_V,
    star_coords_H,
    output_dir,
    x_start,
    y_start,
    cropped_center_x,
    cropped_center_y,
    alpha1=0.5,
    alpha2=0.5,
):
    """
    Aligns the H-alpha starless image to the V starless image using
    an affine transformation estimated from the user-selected star coordinates.
    Marks the galaxy center in the V image, transfers it to the aligned H-alpha image,
    crops the overlapping region, and plots the overlaid images.
    """
    norm = ImageNormalize(vmin=0.0, stretch=LogStretch())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open FITS images.
    V_image = ltf.open_fits(image1_path)
    H_image = ltf.open_fits(image2_path)

    # Convert to a consistent byte order.
    V_image = V_image.byteswap().newbyteorder()
    H_image = H_image.byteswap().newbyteorder()

    # --- Mark the Galaxy Center ---
    x_center = x_start + cropped_center_x  # column index
    y_center = y_start + cropped_center_y  # row index
    unique_value = np.max(V_image) + 100  
    V_image[y_center, x_center] = unique_value

    # --- Align the H-alpha Image ---
    tform = AffineTransform()
    tform.estimate(star_coords_H, star_coords_V)
    aligned_H_image = warp(H_image, inverse_map=tform.inverse, output_shape=V_image.shape)

    # Save the aligned H-alpha image.
    aligned_H_image_path = os.path.join(output_dir, "aligned_Halpha.fits")
    fits.writeto(aligned_H_image_path, aligned_H_image.astype(np.float32), overwrite=True)

    # --- Define the Overlap/Cropping Region ---
    overlap_mask = (aligned_H_image > 0) & (V_image > 0)
    coords_overlap = np.argwhere(overlap_mask)
    if coords_overlap.size == 0:
        raise ValueError("No overlapping region found!")
    row_min, col_min = coords_overlap.min(axis=0)
    row_max, col_max = coords_overlap.max(axis=0)
    V_image_cropped = V_image[row_min:row_max+1, col_min:col_max+1]
    aligned_H_image_cropped = aligned_H_image[row_min:row_max+1, col_min:col_max+1]

    # --- Track the Marked Center ---
    cropped_y = y_center - row_min  # new row coordinate
    cropped_x = x_center - col_min   # new column coordinate
    print("Cropped marked pixel position:", (cropped_y, cropped_x))
    
    if not (0 <= cropped_y < V_image_cropped.shape[0] and 0 <= cropped_x < V_image_cropped.shape[1]):
        print("Warning: The marked center is not within the cropped region!")
    
    aligned_H_image_cropped[cropped_y, cropped_x] = np.max(aligned_H_image_cropped) + 100

    # Save the cropped images.
    V_image_cropped_path = os.path.join(output_dir, f"cropped_{galaxy_name}_V_test.fits")
    aligned_H_image_cropped_path = os.path.join(output_dir, f"cropped_{galaxy_name}_H_test.fits")
    fits.writeto(V_image_cropped_path, V_image_cropped.astype(np.float32), overwrite=True)
    fits.writeto(aligned_H_image_cropped_path, aligned_H_image_cropped.astype(np.float32), overwrite=True)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(V_image_cropped, cmap="Blues", alpha=alpha1, norm=norm, origin="lower")
    ax.imshow(aligned_H_image_cropped, cmap="Reds", alpha=alpha2, norm=norm, origin="lower")
    ax.scatter(cropped_x, cropped_y, color='red', marker='x', s=100, label="Marked Center")
    blue_patch = Rectangle((0, 0), 1, 1, color="blue", label=f"{galaxy_name} V filter")
    red_patch = Rectangle((0, 0), 1, 1, color="red", label=f"{galaxy_name} H-alpha filter")
    plt.legend(handles=[blue_patch, red_patch])
    plt.title(f"Overlay for {galaxy_name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

    return V_image_cropped, aligned_H_image_cropped, cropped_y, cropped_x

# ==============================================================================
#       HELPER FUNCTIONS FOR BATCH PROCESSING
# ==============================================================================

def extract_key(filename):
    """
    Extracts a unique key from the filename.
    If "wlm" (case-insensitive) is in the filename, returns "wlm".
    Otherwise, returns the first found number (with no extra spaces).
    """
    if "wlm" in filename.lower():
        return "wlm"
    else:
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1).strip()
    return None

def find_matching_file(key, folder):
    """
    Searches for a file in the given folder whose name contains the key.
    Returns the full path if found, otherwise None.
    """
    for f in os.listdir(folder):
        if key in f:
            return os.path.join(folder, f)
    return None

# ==============================================================================
#                      MAIN BATCH PROCESSING BLOCK
# ==============================================================================

if __name__ == "__main__":
    # Directories for your four image types.
    folder_V_star = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\V with Stars"
    folder_V_starless = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\V Starless"
    folder_H_star = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\H with Stars"
    folder_H_starless = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\H Starless"

    # Excel file with alignment parameters.
    excel_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\DATA (1).xlsx"
    
    # Output CSV file path.
    csv_output_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\centers.csv"
    
    # Base output directory for processed images.
    base_output_dir = r"C:\Users\AYSAN\Desktop\project\Galaxy\Output"
    
    # Visualization alpha values.
    alpha1 = 0.5
    alpha2 = 0.5
    
    # Read the Excel file with pandas.
    params_df = pd.read_excel(excel_path)
    
    # Assume the first column is the galaxy name (if not already labeled "Galaxy").
    if 'Galaxy' not in params_df.columns:
        params_df.rename(columns={params_df.columns[0]: 'Galaxy'}, inplace=True)
    # Convert galaxy names to lowercase and strip whitespace.
    params_df['Galaxy'] = params_df['Galaxy'].astype(str).str.strip().str.lower()
    
    # Create a matching key from the galaxy name.
    def get_match_key(galaxy):
        # If the galaxy name is "wlm" (any case), return "wlm";
        # else, extract digits from the galaxy name.
        if "wlm" in galaxy:
            return "wlm"
        match = re.search(r'(\d+)', galaxy)
        if match:
            return match.group(1)
        return galaxy
    params_df["match_key"] = params_df["Galaxy"].apply(get_match_key)
    
    # List the files in your V-with-Stars folder.
    files_in_V_star = os.listdir(folder_V_star)
    results = []  # Will store tuples (key, center_x, center_y).
    
    for fname in files_in_V_star:
        key = extract_key(fname)
        if key is None:
            print(f"Skipping {fname}: couldn't extract a key.")
            continue
        
        # Get file paths for each image type using the extracted key.
        V_star_path = os.path.join(folder_V_star, fname)
        V_starless_path = find_matching_file(key, folder_V_starless)
        H_star_path = find_matching_file(key, folder_H_star)
        H_starless_path = find_matching_file(key, folder_H_starless)
        
        if not all([V_star_path, V_starless_path, H_star_path, H_starless_path]):
            print(f"Missing matching file for key {key}. Skipping.")
            continue
        
        # Lookup alignment parameters using the "match_key" from Excel.
        param_row = params_df[params_df['match_key'] == key]
        if param_row.empty:
            print(f"No parameter entry found for key {key} in Excel. Skipping.")
            continue
        
        try:
            row = param_row.iloc[0]
            x_start = int(row['X_start'])
            y_start = int(row['Y_start'])
            cropped_center_x = int(row['X0'])
            cropped_center_y = int(row['Y0'])
        except Exception as e:
            print(f"Error reading parameters for key {key}: {e}")
            continue
        
        print(f"Processing galaxy key: {key} with parameters: x_start={x_start}, y_start={y_start}, X0={cropped_center_x}, Y0={cropped_center_y}")
        
        # Create an output directory for the current galaxy.
        galaxy_output_dir = os.path.join(base_output_dir, key)
        if not os.path.exists(galaxy_output_dir):
            os.makedirs(galaxy_output_dir)
        
        # Interactive star selection.
        print(f"Select corresponding points for galaxy key {key} (see interactive window).")
        star_coords_V, star_coords_H = V_and_Halpha_alignment_interactive(
            key, V_star_path, H_star_path, galaxy_output_dir, alpha1, alpha2
        )
        
        try:
            V_cropped, H_cropped, cropped_y, cropped_x = V_and_Halpha_alignment(
                key,
                V_starless_path,
                H_starless_path,
                star_coords_V,
                star_coords_H,
                galaxy_output_dir,
                x_start,
                y_start,
                cropped_center_x,
                cropped_center_y,
                alpha1,
                alpha2
            )
        except Exception as e:
            print(f"Alignment failed for key {key}: {e}")
            continue
        
        # Save the center coordinates.
        results.append((key, cropped_x, cropped_y))
    
    # Write out CSV with the results.
    with open(csv_output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["key", "center_x", "center_y"])
        for row in results:
            writer.writerow(row)
    
    print(f"CSV with centers has been written to {csv_output_path}")
