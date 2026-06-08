import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end # matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402



def set_up_folders(db_path, datatype="OCT_VIB_NEUR", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, datatype, "2_processed", "oct", "1_minimal_processing")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file(
        db_path_input, "morph.pkl", automatic=automatic, select_multiple=False
    )

    print("------------------")
    print("Input acquisitions of interest (absolute):")
    print(input_foldernames_abs)
    print("Acquisitions of interest:")
    print(input_foldernames)
    print("Total number of acquisitions:")
    print(len(input_foldernames))
    print("done.")
    print("------------------")
    
    return db_path_input, input_foldernames, input_foldernames_abs


if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    datatype = "OCT_HAIR-DEFLECTION" # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    force_processing = True
    show = False
    save_results = True
    save_figure = False
    
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, datatype=datatype, automatic=set_path_automatic)
    
    # 2. Extracting scans
    print(datetime.now())
    n_success = 0
    
    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_folder_abs = input_foldernames_abs[acq_id - 1]
        #output_folder_abs = input_folder_abs.replace("1_minimal_processing", "2_skin_surface")
        output_folder_abs = input_folder_abs
        output_filename = "skin_displacement_estimation.csv"
        output_filename_abs = os.path.join(output_folder_abs, output_filename)
        if not(force_processing) and os.path.exists(output_filename_abs):
            continue

        octr = OCTRecordingManager(input_folder_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata(force_processing=False, save_hdd=False, destdir=input_folder_abs)
        if octr.metadata.isStructural:
            continue
        octr.compute_morph(force_processing=False, save_hdd=False, destdir=input_folder_abs, verbose=True)
        octr.morph.get_morph_video()  # build dB_video
        depth_offset = 15
        [nalines, ndepths, nsamples] = octr.morph.morph_dB_video.shape
        
        # Initialize an empty DataFrame to hold all the data
        df = pd.DataFrame()

        for a in range(nalines):
            d = octr.morph.morph_dB_video[a, depth_offset:, :]
            # Calculate the mean and standard deviation along depth
            mean = np.mean(d, axis=0, keepdims=True)
            std = np.std(d, axis=0, keepdims=True)
            # Define a threshold for noise, for example, mean ± 2*std
            threshold_low = mean + 1 * std
            # Set pixels that are considered noise to 0
            d[(d < threshold_low)] = 0
            # Normalize the image to the range 0-255
            d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
            # Convert the normalized image to uint8 type
            d = d.astype(np.uint8)
            # Apply a median blur to remove small particles
            d = cv2.medianBlur(d, 5)
            # Create a binary image and convert boolean to integer (0 or 1)
            d = (d > np.mean(d)).astype(np.uint8)


            # --- Added from Code 2: Morphological closing ---
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            d_morph = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- Added from Code 2: Connected component labeling ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                d_morph, connectivity=8
            )

            # Keep only large components (remove tiny noise)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area = np.max(areas) if areas.size > 0 else 0
                area_threshold = max_area * 0.01  # remove very small components

                d_filtered = np.zeros_like(d_morph)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                        d_filtered[labels == i] = 1
            else:
                d_filtered = d_morph
                            
            # --- Replace code1 surface detection with code2 version ---
            expected_skin_locations = np.zeros(d_filtered.shape[1], dtype=int)
            for col in range(d_filtered.shape[1]):
                nonzero_indices = np.nonzero(d_filtered[:, col])[0]
                if len(nonzero_indices) > 0:
                    expected_skin_locations[col] = nonzero_indices[0] + depth_offset
                else:
                    if col > 0:
                        expected_skin_locations[col] = expected_skin_locations[col - 1]
                    else:
                        expected_skin_locations[col] = depth_offset
                        
            # # Extract the location of the first value equal to 1 of each column
            # expected_skin_locations = np.argmax(d == 1, axis=0) + depth_offset

            
            # Use the directory name as the column name
            column_name = f"aline_id{a}"
            # Add this column to the large DataFrame
            df[column_name] = expected_skin_locations

            if show or save_figure:
                fig, axs = plt.subplots(2, 1, figsize=(16, 9))
                im = axs[0].imshow(octr.morph.morph_dB_video[a, :, :], cmap='gray', aspect='auto')
                axs[0].set_title('Initial')
                axs[0].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[0])

                im = axs[1].imshow(d, cmap='gray', aspect='auto')
                axs[1].set_title('Processed')
                axs[1].set_ylabel('Depth (pxl)')
                fig.colorbar(im, ax=axs[1])
                # Plot the locations of the first value equal to 1 in each column as a red line
                axs[1].plot(expected_skin_locations-depth_offset, color='red')
                fig.suptitle(f"{input_folder_abs}: a-line {a}/{nalines}")
                if save_figure:
                    output_img = f"_skin-displacement-estimation_figure_a-line-{a}.png"
                    output_img_abs = output_folder_abs + output_img
                    # Create the directory if it doesn't exist
                    if not os.path.exists(os.path.dirname(output_img_abs)):
                        os.makedirs(os.path.dirname(output_img_abs))
                    # fig.savefig(output_img_abs, dpi=300, bbox_inches='tight')  # Use dpi=300 for high-quality images
                if show:
                    plt.show(block=True)

        if save_results:
            # Create the directory if it doesn't exist
            if not os.path.exists(output_folder_abs):
                os.makedirs(output_folder_abs)
            
            # Write to CSV file
            df.to_csv(output_filename_abs, index=False)

        n_success += 1

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")

