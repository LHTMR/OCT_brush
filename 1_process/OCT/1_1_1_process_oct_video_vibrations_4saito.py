import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import cv2

# Add the path to the library_python module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import library_python.data_management.path_tools as path_tools  # noqa: E402
from library_python.sensors.OCT.OCTRecordingManager1 import OCTRecordingManager  # noqa: E402
from library_python.sensors.OCT.OCTMorph import OCTMorph  # noqa: E402
from library_python.signal_processing.InteractiveVectorEditor import InteractiveVectorEditor  # noqa: E402

def set_up_folders(db_path, dataset="OCT_BRUSH", automatic=True):
    """Sets up the input and output folders and retrieves folder names with required files."""
    db_path_input = os.path.join(db_path, dataset, "1_primary", "oct")
    input_foldernames, input_foldernames_abs, input_folder_session_abs = path_tools.get_folders_with_file(
        db_path_input, "Measurement.srm", automatic=automatic, select_multiple=False
    )
    return db_path_input, input_foldernames, input_foldernames_abs

if __name__ == "__main__":
    # 0. Initialization of parameters
    data_external_hdd = False
    set_path_automatic = False
    dataset = "OCT_BRU_NEUR"  # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION, OCT_BRU_NEUR
    datatype = "OCT_HAIR-DEFLECTION"  # OCT_VIB_NEUR, OCT_HAIR-DEFLECTION

    force_processing = True
    save_results = True
    show = False
    save_figure = True
    # Initialize paths and setup folders
    db_path = path_tools.define_OCT_database_path(data_external_hdd)  # Assumes this is a custom function
    print(f"Path initialized:\ndb_path = '{db_path}'")
    db_path_input, input_foldernames, input_foldernames_abs = set_up_folders(db_path, dataset=dataset, automatic=set_path_automatic)

    # 2. Extracting scans
    print(datetime.now())
    n_success = 0

    for acq_id, input_fn in enumerate(input_foldernames, start=1):
        t = f"Acquisition nº {acq_id}/{len(input_foldernames)}: {input_fn}"
        print(f"{datetime.now()}\t{t}")
        input_fn_abs = input_foldernames_abs[acq_id - 1]

        output_folder_abs = input_fn_abs.replace("1_primary", "2_processed")

        octr = OCTRecordingManager(input_fn_abs, output_folder_abs, autosave=save_results)
        octr.load_metadata()
        if octr.metadata.isStructural:
            continue
        if octr.exist("morph.pkl")[0] and not(force_processing):
            print("Morph has been previously processed and processing is not forced.")
            continue

        octr.compute_morph(force_processing=force_processing, save_hdd=False)  # save_results

        # Process the morph data without saving
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

            expected_skin_locations = np.argmax(d == 1, axis=0) + depth_offset

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
                axs[1].plot(expected_skin_locations - depth_offset, color='red')
                fig.suptitle(f"{input_fn_abs}: a-line {a}/{nalines}")
                if save_figure:
                    output_img = f"_skin-displacement-estimation_figure_a-line-{a}.png"
                    output_img_abs = output_folder_abs + output_img
                    # Create the directory if it doesn't exist
                    if not os.path.exists(os.path.dirname(output_img_abs)):
                        os.makedirs(os.path.dirname(output_img_abs))
                    fig.savefig(output_img_abs, dpi=30, bbox_inches='tight')  # Use dpi=300 for high-quality images
                if show:
                    plt.show(block=True)

        if save_results:
            output_filename = "skin_displacement_estimation.csv"
            output_filename_abs = os.path.join(output_folder_abs, output_filename)
            if not os.path.exists(output_folder_abs):
                os.makedirs(output_folder_abs)
            # Write to CSV file
            df.to_csv(output_filename_abs, index=False)

        n_success += 1

    print(datetime.now())
    print(f"{n_success}/{len(input_foldernames_abs)} acquisitions have been processed.")



