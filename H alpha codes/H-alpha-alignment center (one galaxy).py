import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage.transform import AffineTransform, warp
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
import os
import little_things_functions as ltf

def V_and_Halpha_alignment_interactive(
    galaxy_name, image1_path, image2_path, output_dir, alpha1=0.5, alpha2=0.5
):
    norm = ImageNormalize(vmin=0.0, stretch=LogStretch())

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open FITS files and get data
    V_image = ltf.open_fits(image1_path)
    H_image = ltf.open_fits(image2_path)

    # Interactive point selection with magnifier
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    coords_V = []
    coords_H = []

    # Display images
    im_V = axes[0].imshow(V_image, cmap="gray", norm=norm, origin="lower")
    axes[0].set_title(f"{galaxy_name} V Image")
    im_H = axes[1].imshow(H_image, cmap="gray", norm=norm, origin="lower")
    axes[1].set_title(f"{galaxy_name} H-alpha Image")

    plt.subplots_adjust(bottom=0.2)

    # Instructions
    plt.suptitle(
        "Click on corresponding points in each image.\n"
        "Select at least 4 points per image in the same order."
    )

    selected_points_text = fig.text(0.5, 0.05, "", ha="center")

    # Variables to track state
    num_points_V = 0
    num_points_H = 0

    # Magnifier settings
    zoom_factor = 7  # Zoom factor for magnification
    rect_size = 20  # Size of the rectangle indicating the magnified area

    # Initialize magnifier for V image
    axins_V = inset_axes(
        axes[0], width="20%", height="20%", loc="upper right", borderpad=1
    )
    img_V_magnified = axins_V.imshow(
        V_image, norm=norm, origin="lower"
    )
    axins_V.axis("off")

    rect_V = Rectangle(
        (0, 0),
        rect_size,
        rect_size,
        edgecolor="red",
        facecolor="none",
        linewidth=1,
    )
    axes[0].add_patch(rect_V)

    # Crosshairs for V image magnifier
    h_line_V = axins_V.axhline(color='yellow', linewidth=0.5)
    v_line_V = axins_V.axvline(color='yellow', linewidth=0.5)

    # Initialize magnifier for H-alpha image
    axins_H = inset_axes(
        axes[1], width="20%", height="20%", loc="upper right", borderpad=1
    )
    img_H_magnified = axins_H.imshow(
        H_image, norm=norm, origin="lower"
    )
    axins_H.axis("off")

    rect_H = Rectangle(
        (0, 0),
        rect_size,
        rect_size,
        edgecolor="red",
        facecolor="none",
        linewidth=1,
    )
    axes[1].add_patch(rect_H)

    # Crosshairs for H-alpha image magnifier
    h_line_H = axins_H.axhline(color='yellow', linewidth=0.5)
    v_line_H = axins_H.axvline(color='yellow', linewidth=0.5)

    # Event handler functions
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
            return  # Ignore clicks outside the axes
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

                # Update the inset axes limits
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                # Invert y-axis for correct orientation
                axins.invert_yaxis()

                # Update the rectangle position
                rect.set_xy((x1, y1))
                rect.set_width(x2 - x1)
                rect.set_height(y2 - y1)

                # Update crosshairs in the magnifier
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                h_line.set_ydata([center_y])  # Pass as sequence
                v_line.set_xdata([center_x])  # Pass as sequence

                fig.canvas.draw_idle()

    def update_text():
        selected_points_text.set_text(
            f"Points selected - V Image: {num_points_V}, "
            f"H-alpha Image: {num_points_H}"
        )
        fig.canvas.draw_idle()

    # Connecting the click events
    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # Adding a 'Proceed' button
    ax_proceed = plt.axes([0.45, 0.01, 0.1, 0.05])
    btn_proceed = Button(ax_proceed, "Proceed")

    def proceed(event):
        if len(coords_V) >= 4 and len(coords_V) == len(coords_H):
            plt.close()
        else:
            print(
                "Please select at least 4 corresponding points on both images."
            )
            print("Ensure that the number of points matches between images.")

    btn_proceed.on_clicked(proceed)

    plt.show()

    # Disconnect the event handlers
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_move)

    # Convert to numpy arrays
    star_coords_V = np.array(coords_V)
    star_coords_H = np.array(coords_H)

    # Proceed with the rest of the code using the selected coordinates
    return star_coords_V, star_coords_H


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import AffineTransform, warp
from matplotlib.patches import Rectangle
from astropy.visualization import LogStretch, ImageNormalize

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import AffineTransform, warp
from matplotlib.patches import Rectangle
from astropy.visualization import LogStretch, ImageNormalize

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
    norm = ImageNormalize(vmin=0.0, stretch=LogStretch())

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open FITS files
    V_image = ltf.open_fits(image1_path)
    H_image = ltf.open_fits(image2_path)

    # Convert to little-endian format
    V_image = V_image.byteswap().newbyteorder()
    H_image = H_image.byteswap().newbyteorder()

    # --- Marking the Galaxy Center ---
    # In our convention, the row (first index) corresponds to y and the column (second index) to x.
    # So we calculate the galaxy's center as given by the provided offsets.
    x_center = x_start + cropped_center_x  # column index
    y_center = y_start + cropped_center_y  # row index

    # Use a unique value that is guaranteed to be higher than any other pixel:
    unique_value = np.max(V_image) + 100  
    V_image[y_center, x_center] = unique_value

    # --- Aligning the H-alpha Image ---
    # Estimate the affine transformation from H_image to V_image based on the star coordinates.
    tform = AffineTransform()
    tform.estimate(star_coords_H, star_coords_V)

    # Warp H_image so that it is aligned to V_image (output shape matching V_image)
    aligned_H_image = warp(H_image, inverse_map=tform.inverse, output_shape=V_image.shape)

    # Save the full aligned image
    aligned_H_image_path = os.path.join(output_dir, "aligned_Halpha.fits")
    fits.writeto(aligned_H_image_path, aligned_H_image.astype(np.float32), overwrite=True)

    # --- Defining the Overlap/Cropping Region ---
    # Create an overlap mask based on pixels present (values > 0) in both images.
    overlap_mask = (aligned_H_image > 0) & (V_image > 0)
    coords_overlap = np.argwhere(overlap_mask)
    if coords_overlap.size == 0:
        raise ValueError("No overlapping region found!")
    # Note: np.argwhere returns [row, column] pairs.
    row_min, col_min = coords_overlap.min(axis=0)  # crop boundaries (y_min, x_min)
    row_max, col_max = coords_overlap.max(axis=0)  # crop boundaries (y_max, x_max)

    # Crop the full images to the region where both images overlap.
    V_image_cropped = V_image[row_min:row_max+1, col_min:col_max+1]
    aligned_H_image_cropped = aligned_H_image[row_min:row_max+1, col_min:col_max+1]

    # --- Tracking the Marked Center ---
    # The original marked pixel in V_image is at (y_center, x_center). In the cropped image,
    # its new coordinates become (y_center - row_min, x_center - col_min).
    cropped_y = y_center - row_min  # new row coordinate (y)
    cropped_x = x_center - col_min  # new column coordinate (x)
    print("Cropped marked pixel position:", (cropped_y, cropped_x))
    
    # Check that the computed center is inside the cropped frame.
    if not (0 <= cropped_y < V_image_cropped.shape[0] and 0 <= cropped_x < V_image_cropped.shape[1]):
        print("Warning: The marked center is not within the cropped region!")
    
    # Mark the corresponding pixel in the aligned H-alpha image.
    aligned_H_image_cropped[cropped_y, cropped_x] = np.max(aligned_H_image_cropped) + 100

    # --- Save Results ---
    V_image_cropped_path = os.path.join(output_dir, f"cropped_{galaxy_name}_V_test.fits")
    aligned_H_image_cropped_path = os.path.join(output_dir, f"cropped_{galaxy_name}_H_test.fits")
    fits.writeto(V_image_cropped_path, V_image_cropped.astype(np.float32), overwrite=True)
    fits.writeto(aligned_H_image_cropped_path, aligned_H_image_cropped.astype(np.float32), overwrite=True)

    # --- Visualization ---
    # When plotting, remember that imshow() takes (column, row) coordinates,
    # so we use (cropped_x, cropped_y) in scatter.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(V_image_cropped, cmap="Blues", alpha=alpha1, norm=norm, origin="lower")
    ax.imshow(aligned_H_image_cropped, cmap="Reds", alpha=alpha2, norm=norm, origin="lower")
    ax.scatter(cropped_x, cropped_y, color='red', marker='x', s=100, label="Marked Center")
    
    # Create legend patches for clarity.
    blue_patch = Rectangle((0, 0), 1, 1, color="blue", label=f"{galaxy_name} V filter")
    red_patch = Rectangle((0, 0), 1, 1, color="red", label=f"{galaxy_name} H-alpha filter")
    plt.legend(handles=[blue_patch, red_patch])
    
    plt.title(f"Overlay for {galaxy_name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

    return V_image_cropped, aligned_H_image_cropped , cropped_y, cropped_x

# Main code
if __name__ == "__main__":
    # Load the data
    galaxy_name = "DDO 87"

    # Paths to the images
    V_image_path_star = r"C:\Users\AYSAN\Desktop\project\Galaxy\Data\WLM\wlmv.fits" #location to the V-filter image (with stars)
    H_image_path_star = r"C:\Users\AYSAN\Desktop\project\Galaxy\Data\WLM\wlmha.fits"#location to the H-alpha-filter image (with stars)

    output_dir_no_stars = r"C:\Users\AYSAN\Desktop\project\Galaxy\Code\WLM" #location to where your starless cropped and resized images are saved.
    output_dir_stars = r"C:\Users\AYSAN\Desktop\project\Galaxy\Code\WLM"#location to where your cropped and resized images are saved (withstars).
    folder_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\Junk"#location to where the final plots are saved.
    # Paths to the starless images
    V_image_path_starless = r"C:\Users\AYSAN\Desktop\project\Galaxy\Starless images\WLM\final_wLMv.fits"#location to the starless V-filter image
    H_image_path_starless = r"C:\Users\AYSAN\Desktop\project\Galaxy\Data\WLM\wlmhmrms.fits"#location to the starless H-alpha-filter image
    # Load H-alpha image and plot in log scale
    DDO168_H = ltf.open_fits(H_image_path_star)
    ltf.log_scale_plot(DDO168_H, f"H-alpha image for {galaxy_name}", "log scale")

    # Select corresponding points interactively
    star_coords_V, star_coords_H = V_and_Halpha_alignment_interactive(
        galaxy_name,
        V_image_path_star,
        H_image_path_star,
        output_dir_stars,
        alpha1=0.5,
        alpha2=0.5
    )


    # Perform alignment
    V_image_cropped, H_image_cropped , center_x, center_y = V_and_Halpha_alignment(
        galaxy_name,
        V_image_path_starless,
        H_image_path_starless,
        star_coords_V,
        star_coords_H,
        output_dir_no_stars,
        x_start = 100,
        y_start = 100,
        cropped_center_x = 407,
        cropped_center_y = 390,
        alpha1=0.5,
        alpha2=0.5
    )


    # Plot contours (assuming this function exists in your ltf module)
    #ltf.plot_contours_V_and_Halpha(
        #galaxy_name,
        #V_image_cropped,
        #H_image_cropped,
        #r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions", contour_levels=[4,6,8,10])

    #H_pixelscale = ltf.calculate_pixelscale(star_coords_V, star_coords_H, 1.134) #enter V-filter pixel scale. 
    #print("H-alpha pixelscale:", H_pixelscale)

