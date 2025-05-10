import os
import re
import csv
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch, ImageNormalize, ZScaleInterval
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector, Button

# ---------------------------
# Build the list of image sets
# ---------------------------
folder1 = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\Cropped H-alpha"  # FITS starless images
folder2 = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha length\H with Stars"    # FITS images with star
folder3 = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\Regions"  # PNG files

# Get lists of files in each folder
files1 = [f for f in os.listdir(folder1) if re.search(r'\.(fits|fit)$', f, re.IGNORECASE)]
files2 = os.listdir(folder2)
files3 = [f for f in os.listdir(folder3) if f.lower().endswith('.png')]

# Prepare a list for image sets (only if matches in both folder2 and folder3 exist)
image_sets = []
for file1 in files1:
    key = None
    # First check if the filename has the keyword "wlm"
    if "wlm" in file1.lower():
        key = "wlm"
    else:
        match = re.search(r'\d+', file1)
        if match:
            key = match.group(0)
        else:
            print(f"No number or 'wlm' found in filename: {file1}")
            continue

    # Search for matching file in folder2 using the key
    matching_file_2 = None
    for file2 in files2:
        if re.search(key, file2, re.IGNORECASE):
            matching_file_2 = file2
            break

    # Search for matching PNG in folder3 using the key
    matching_file_3 = None
    for file3 in files3:
        if re.search(key, file3, re.IGNORECASE):
            matching_file_3 = file3
            break

    if matching_file_2 and matching_file_3:
        image_sets.append({
            'number': key,
            'file1': os.path.join(folder1, file1),
            'file1_title': file1,
            'file2': os.path.join(folder2, matching_file_2),
            'file2_title': matching_file_2,
            'file3': os.path.join(folder3, matching_file_3),
            'file3_title': matching_file_3
        })
    else:
        if not matching_file_2:
            print(f"No matching file in folder2 for {file1} (key: {key})")
        if not matching_file_3:
            print(f"No matching PNG file in folder3 for {file1} (key: {key})")

if not image_sets:
    raise RuntimeError("No image sets found with matches in all three folders.")

# ---------------------------
# Global variables for interactive state
# ---------------------------
current_index = 0          # Index of the current image set
records = []               # List to hold [number, "y_min - y_max", "x_min - x_max"]
selected_coords = None     # To hold the current rectangle selection (x_min, x_max, y_min, y_max)
rect_selector = None       # Global reference to the RectangleSelector

# CSV file output path
csv_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\H-alpha crop for cropped images.csv"

# ---------------------------
# Callback for rectangle selection on the first axis
# ---------------------------
def onselect(eclick, erelease):
    global selected_coords
    # Get coordinates from the click and release events
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    if None in (x1, x2, y1, y2):
        return
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    selected_coords = (x_min, x_max, y_min, y_max)
    print(f"Selected region: x: {int(x_min)}-{int(x_max)}, y: {int(y_min)}-{int(y_max)}")

# ---------------------------
# Update the figure with the current image set
# ---------------------------
def update_figure():
    global selected_coords, rect_selector
    # Clear all axes
    for ax in axs:
        ax.cla()

    # Get the current image set data
    curr_set = image_sets[current_index]
    try:
        # Load FITS data for folder1 and folder2 images
        data1 = fits.getdata(curr_set['file1'])
        data2 = fits.getdata(curr_set['file2'])
        # Load PNG image data for folder3
        data3 = mpimg.imread(curr_set['file3'])
    except Exception as e:
        print(f"Error loading image data for set {curr_set['number']}: {e}")
        return

    # Create normalization objects
    norm1 = ImageNormalize(vmin=0, stretch=LogStretch())
    zscale = ZScaleInterval()
    vmin2, vmax2 = zscale.get_limits(data2)
    norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2)

    # Plot the three images
    axs[0].imshow(data1, origin="lower", norm=norm1, aspect='auto', cmap = "Greys_r")
    axs[0].set_title(curr_set['file1_title'])
    axs[0].set_box_aspect(1)

    axs[1].imshow(data2, origin="lower", norm=norm2, aspect='auto')
    axs[1].set_title(curr_set['file2_title'])
    axs[1].set_box_aspect(1)

    axs[2].imshow(data3, aspect='auto')
    axs[2].set_title(curr_set['file3_title'])
    axs[2].set_box_aspect(1)

    fig.suptitle(f"Comparison for {curr_set['number']}")
    selected_coords = None  # Reset selection for new image

    # Remove old RectangleSelector if present and create a new one on the first axis.
    # Removed the 'drawtype' parameter since it's unsupported in your version.
    if rect_selector is not None:
        rect_selector.disconnect_events()
        rect_selector.set_active(False)
    rect_selector = RectangleSelector(axs[0], onselect, useblit=True,
                                      button=[1],    # Left-click for selection
                                      minspanx=5, minspany=5, spancoords='pixels',
                                      interactive=True)
    plt.draw()

# ---------------------------
# Next button callback
# ---------------------------
def next_button_callback(event):
    global current_index, selected_coords, records, image_sets
    if selected_coords is None:
        print("No region selected on the first image. Please select a region first.")
        return

    x_min, x_max, y_min, y_max = selected_coords
    # Format the coordinate ranges as "min - max"
    x_range = f"{int(x_min)} - {int(x_max)}"
    y_range = f"{int(y_min)} - {int(y_max)}"
    curr_key = image_sets[current_index]['number']
    records.append([curr_key, y_range, x_range])
    print(f"Saved selection for image set with key '{curr_key}'")

    # Move to the next image set if available
    current_index += 1
    if current_index < len(image_sets):
        update_figure()
    else:
        # Save CSV records and close the figure
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Number", "Y Coordinates", "X Coordinates"])
            writer.writerows(records)
        print("All images processed. CSV file saved at:")
        print(csv_file_path)
        plt.close(fig)

# ---------------------------
# Create the initial figure and Next button
# ---------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.3, top=0.85)  # Increased space between images

# Create Next button axis (positioned at the bottom-right of the figure)
button_ax = plt.axes([0.81, 0.02, 0.15, 0.06])
next_button = Button(button_ax, "Next")
next_button.on_clicked(next_button_callback)

# Initialize the first image set
update_figure()

plt.show()

# import os
# import csv
# from astropy.io import fits
# import matplotlib.pyplot as plt
# from astropy.visualization import LogStretch, ImageNormalize, ZScaleInterval
# from matplotlib.widgets import RectangleSelector, Button

# # --- File Paths for the WLM exception pair ---
# file1_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha images raw starless\wlmhmrms.fits"
# file2_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha images with star\wlmha.fits"
# key = "wlm"

# # CSV file output path (will be appended to or created if missing)
# csv_file_path = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\H-alpha crop.csv"

# # --- Load FITS Data ---
# try:
#     image_data1 = fits.getdata(file1_path)
#     image_data2 = fits.getdata(file2_path)
# except Exception as e:
#     print(f"Error loading FITS files: {e}")
#     exit()

# # --- Create normalization objects ---
# norm1 = ImageNormalize(vmin=0, stretch=LogStretch())
# zscale = ZScaleInterval()
# vmin2, vmax2 = zscale.get_limits(image_data2)
# norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2)

# # --- Create a figure with 2 subplots ---
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.3, top=0.85)

# # Left: image from folder1 with log normalization
# axs[0].imshow(image_data1, origin="lower", norm=norm1, aspect='auto')
# axs[0].set_title(f"{key} - " + os.path.basename(file1_path))

# # Right: image from folder2 with z-scale normalization
# axs[1].imshow(image_data2, origin="lower", norm=norm2, aspect='auto')
# axs[1].set_title(f"{key} - " + os.path.basename(file2_path))

# selected_coords = None

# # --- Rectangle selector callback ---
# def onselect(eclick, erelease):
#     global selected_coords
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata  # Fixed the typo here
#     if None in (x1, x2, y1, y2):
#         return
#     x_min, x_max = sorted([x1, x2])
#     y_min, y_max = sorted([y1, y2])
#     selected_coords = (x_min, x_max, y_min, y_max)
#     print(f"Selected region: x: {int(x_min)}-{int(x_max)}, y: {int(y_min)}-{int(y_max)}")

# # Create the RectangleSelector on the left axis for interactive selection
# rect_selector = RectangleSelector(axs[0], onselect,
#                                   useblit=True,
#                                   button=[1],         # left-click only
#                                   minspanx=5, minspany=5,
#                                   spancoords='pixels',
#                                   interactive=True)

# # --- Next button callback ---
# def next_button_callback(event):
#     global selected_coords
#     if selected_coords is None:
#         print("No region selected. Please drag-select a region on the left image before clicking Next.")
#         return
#     x_min, x_max, y_min, y_max = selected_coords
#     # Format coordinate ranges as "min - max"
#     x_range = f"{int(x_min)} - {int(x_max)}"
#     y_range = f"{int(y_min)} - {int(y_max)}"
#     print(f"Saving selection for key '{key}': Y: {y_range}, X: {x_range}")
    
#     # Create the CSV file with header if it doesn't exist; otherwise, append a new row.
#     file_exists = os.path.exists(csv_file_path)
#     with open(csv_file_path, "a", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             writer.writerow(["Number", "Y Coordinates", "X Coordinates"])
#         writer.writerow([key, y_range, x_range])
        
#     print(f"Selection saved in CSV at {csv_file_path}")
#     plt.close(fig)

# # Create a Next button at the bottom-right of the figure.
# button_ax = plt.axes([0.85, 0.02, 0.1, 0.06])
# next_button = Button(button_ax, "Next")
# next_button.on_clicked(next_button_callback)

# plt.show()
