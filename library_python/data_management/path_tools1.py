import os
import ctypes
import socket
import subprocess
import platform
import os
import psutil
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import warnings
import pandas as pd

def define_OCT_database_path(data_external_hdd=False):
    # Determine the computer ID based on environment variables
    if platform.system() == "Windows":
        computer_id = os.getenv("COMPUTERNAME", "")
    elif platform.system() == "Linux":
        # Assume being on Linux (Anders Eklund server)
        computer_id = socket.gethostname()
    else:
        # Assume ThinLinc Client on Linux (Anders Fridberger server)
        computer_id = os.getenv("ARCH", "")

    # Define paths based on computer ID and external HDD flag
    if computer_id == 'BASIL':
        # Personal computer
        if data_external_hdd:
            letter = get_drive_letter("OCT DATA")
            if letter:
                database_abs_path = os.path.join(letter, "OCT_VIB")
            else:
                raise ValueError("Drive with label 'OCT DATA' not found.")
        else:
            database_abs_path = r"F:\OneDrive - Linköpings universitet\_Teams\OCT Skin Documents\data"
    elif computer_id in {'WIN03072', 'WIN03143'}:
        # OCT computer or professional laptop path
        if data_external_hdd:
            letter = get_drive_letter("OCT DATA")
            if letter:
                database_abs_path = os.path.join(letter, "OCT_VIB")
            else:
                raise ValueError("Drive with label 'OCT DATA' not found.")
        else:
            database_abs_path = os.path.join(r"C:\Users\basdu83", 
                r"OneDrive - Linköpings universitet\_Teams\OCT Skin Documents\data")
    elif computer_id in {'WIN05441'}:
        if data_external_hdd:
            letter = get_drive_letter("OCT DATA")
            if letter:
                database_abs_path = os.path.join(letter, "OCT_VIB")
            else:
                raise ValueError("Drive with label 'OCT DATA' not found.")
        else:
            database_abs_path = os.path.join(r"C:\Users\saisa68", 
                r"OneDrive - Linköpings universitet\_Teams\OCT Skin Documents\data")     
    elif computer_id == 'glnxa64':
        # ThinLinc Client
        if data_external_hdd:
            database_abs_path = "/data/home/basil/thindrives/OCT_VIB/"
        else:
            database_abs_path = "/data/home/basil/thindrives/data/"
    elif computer_id in {'lnx00400', 'lnx00400.ad.liu.se'}:
        if data_external_hdd:
            raise ValueError(f"Workstation: {computer_id}. Asking for a external HDD (=true) but not sure how it can be possible.")
        else:
            database_abs_path = "/coop/l/lhtmr_oct/OCT_database"
    else:
        raise ValueError(f"Unknown workstation: {computer_id}")

    return database_abs_path


# much faster than subprocess
def get_drive_label(drive):
    """Get the volume name (label) of a drive using ctypes on Windows."""
    volume_name_buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive), volume_name_buffer, ctypes.sizeof(volume_name_buffer),
        None, None, None, None, 0)
    return volume_name_buffer.value

def get_drive_letter(target_label):
    """
    Check for a specific hard drive name (volume label) and return its mount point.
    
    Parameters:
        target_label (str): The volume label of the hard drive to search for.
    
    Returns:
        str: The mount point of the hard drive with the matching label, or None if not found.
    """
    for partition in psutil.disk_partitions():
        if partition.device.endswith("\\"):  # For Windows drives like C:\\
            label = get_drive_label(partition.device)
            if label.lower() == target_label.lower():
                return partition.mountpoint
    return None


def get_folders_with_file(folder, target_file, search_by_substring=False, automatic=False, select_multiple=True, verbose=False):
    if automatic:
        foldernames, foldernames_abs, session_abs = select_input_folders_auto(folder)
    else:
        foldernames, foldernames_abs, session_abs = select_input_folders_manual(folder, select_multiple)
    
    if isinstance(foldernames, str):
        foldernames = [foldernames]
        foldernames_abs = [foldernames_abs]
        # warnings.warn(f"Warning: foldernames and foldernames_abs variables aren't of type List. Converts them...")

    # Filter folders to only include those containing the target file or substring
    indices_to_remove = []
    for i, folder_abs in enumerate(foldernames_abs):
        if search_by_substring:
            # Check if any file in the folder contains the substring
            if not any(target_file in f.name for f in Path(folder_abs).iterdir() if f.is_file()):
                indices_to_remove.append(i)
        else:
            # Check if the specific file exists
            if not (Path(folder_abs) / target_file).is_file():
                indices_to_remove.append(i)
    
    # Remove folders that don't meet the criteria
    foldernames = [name for i, name in enumerate(foldernames) if i not in indices_to_remove]
    foldernames_abs = [name_abs for i, name_abs in enumerate(foldernames_abs) if i not in indices_to_remove]
    
    # no folder has been found fullfilling the target_file requirement
    if foldernames_abs:
        # Make sure that they use the same standard path/directory system
        if "/" in session_abs:
            session_abs = session_abs.replace("/", "\\")
        if "/" in foldernames_abs[0]:
            foldernames_abs = [name_abs.replace("/", "\\") for name_abs in foldernames_abs]
        if verbose:
            print("------------------")
            print("Input acquisitions of interest (absolute):")
            print(foldernames_abs)
            print("Acquisitions of interest:")
            print(foldernames)
            print(f"Total number of acquisitions: {len(foldernames)}")
            print(f"Path initialized:\nsession_abs = '{session_abs}'")
            print("------------------")
    else:
        warnings.warn(f"No subfolders of interest found for the target file {target_file} in:\n{folder}.")
        foldernames = None
        foldernames_abs = None
        session_abs = None

    return foldernames, foldernames_abs, session_abs


def select_input_folders_auto(folder_base_abs):
    foldernames_abs = get_leaf_folders(Path(folder_base_abs))
    foldernames = [Path(folder).name for folder in foldernames_abs]
    return foldernames, foldernames_abs, folder_base_abs



def select_input_folders_manual(input_folder_base_abs, select_multiple=False):
    input_foldernames = []
    input_foldernames_abs = []
    work = True

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)
    while work:
        input_folder_session_abs = filedialog.askdirectory(initialdir=input_folder_base_abs, 
                                                           parent=root)

        if input_folder_session_abs:
            choice = show_custom_dialog()
            if choice == 'Yes':
                leaf_folders_abs = get_leaf_folders(Path(input_folder_session_abs))
                leaf_folders = [Path(folder).name for folder in leaf_folders_abs]
                input_foldernames.extend(leaf_folders)
                input_foldernames_abs.extend(leaf_folders_abs)
                
                for folder in leaf_folders_abs:
                    data_folder = Path(folder) / "data"
                    if data_folder.exists():
                        csv_files = list(data_folder.glob("*_results.csv"))
                        target_intervals = get_target_intervals(csv_files)
                        interval_folders = find_interval_folders(folder, target_intervals)
                        for interval_folder in interval_folders:
                                input_foldernames.append(interval_folder.name)
                                input_foldernames_abs.append(str(interval_folder))
            elif choice == 'Manual':
                subfolders = select_subfolders(Path(input_folder_session_abs))
                if subfolders:
                    input_foldernames.extend(subfolders)
                    input_foldernames_abs.extend([str(Path(input_folder_session_abs) / subfolder) for subfolder in subfolders])
            elif choice == "Is Data Folder":
                # Skip fetching subfolders
                input_foldernames = os.path.basename(input_folder_session_abs)
                input_foldernames_abs = input_folder_session_abs
        
        if select_multiple:
            choice = messagebox.askquestion("Select input folders manually", "Fetch more acquisitions?")
            work = (choice == 'yes')
        else:
            work = False
            
    root.quit()
    root.destroy()  # Close the Tkinter window
    
    return input_foldernames, input_foldernames_abs, input_folder_session_abs

def extract_block_info(filename):
    # extract block info. from filename: "2025-05-05_11-50-15_P01_OVP_block-01_freq-5.0Hz_results.csv"
    parts = filename.split('_')
    block_info = {
        'block_number': parts[5].split('-')[1],
        'frequency_hz': parts[6].split('-')[1].replace('Hz', '')
    }
    return block_info

def read_csv_file(csv_file):
    # read CSV, extract trial_number & target_interval
    df = pd.read_csv(csv_file)
    trials_info = df[['trial_number', 'target_interval']].to_dict('records')
    return trials_info


def get_target_intervals(csv_files):
    """Extracts target intervals from CSV files."""
    target_intervals = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            trial_number = row['trial_number']
            block_number = row['block_number']
            target_interval = row['target_interval']
            if block_number not in target_intervals:
                target_intervals[block_number] = {}
            target_intervals[block_number][trial_number] = target_interval
    return target_intervals

def find_interval_folders(base_path, target_intervals):
    """Finds interval folders based on target intervals."""
    interval_folders = []
    for block_number, trials in target_intervals.items():
        for trial_number, target_interval in trials.items():
            # Generate the folder name dynamically based on the block and trial numbers
            folder_name = Path(base_path) / f"block-{block_number:02d}_trial-{trial_number:02d}" / f"interval{target_interval}"
            interval_folders.append(folder_name)
    return interval_folders



def get_leaf_folders(folder, verbose=False):
    leaf_folders = []
    folder_content = [f for f in Path(folder).iterdir() if f.is_dir()]

    if verbose:
        print("-----------")
        print("folder_content.name:", [f.name for f in folder_content])

    if not folder_content:
        leaf_folders = [folder]
        if verbose:
            print("+++++++++++")
            print("Leaf found.")
            print("+++++++++++")
    else:
        for subfolder in folder_content:
            # if f"intercal{target_interval}" in subfolder.name:
            leaf_folders.extend(get_leaf_folders(subfolder, verbose))
    
    # Convert each Path object to string
    leaf_folders = [str(p) for p in leaf_folders]

    return leaf_folders


def select_subfolders(folder):
    selected_folders = []

    def list_subfolders(folder_path):
        subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
        for subfolder in subfolders:
            listbox.insert(tk.END, subfolder)

    def get_selected_folders():
        nonlocal selected_folders
        selected_folders = [listbox.get(i) for i in listbox.curselection()]
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Select Subfolders")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=50, height=15)
    listbox.pack(pady=10)

    ok_button = tk.Button(frame, text="OK", command=get_selected_folders)
    ok_button.pack()

    list_subfolders(folder)

    root.mainloop()

    return selected_folders


def show_custom_dialog():
    """
    Display a custom dialog box with three options: Yes, Manual, and Is Data Folder.
    Block execution until the user selects one of the options. The "Yes" button is preselected.
    """
    # Initialize a value to store the user's choice
    user_choice = None

    # Function to set the user choice, quit the dialog loop, and destroy the window
    def on_select(choice):
        nonlocal user_choice
        user_choice = choice
        dialog.quit()
        dialog.destroy()

    # Create the dialog window
    dialog = tk.Tk()
    dialog.title("Folder Selection")
    dialog.geometry("400x200")  # Set the default size

    # Center the dialog on the screen
    dialog.update_idletasks()
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    window_width = 400
    window_height = 200
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
    dialog.resizable(False, False)

    # Add the main message
    label = tk.Label(dialog, text="Automatically get all leaf folders?", font=("Arial", 12))
    label.pack(pady=10)

    # Add the details of the options
    detail_label = tk.Label(
        dialog, text="Yes: Automatically select all leaf folders\n"
                     "Manual: Manually select subfolders\n"
                     "Is Data Folder: The selected folder contains the data",
        justify="left", font=("Arial", 10)
    )
    detail_label.pack(pady=5)

    # Add the buttons
    yes_button = tk.Button(dialog, text="Yes", width=15, command=lambda: on_select("Yes"))
    manual_button = tk.Button(dialog, text="Manual", width=15, command=lambda: on_select("Manual"))
    is_data_folder_button = tk.Button(dialog, text="Is Data Folder", width=15, command=lambda: on_select("Is Data Folder"))

    yes_button.pack(side="left", expand=True, padx=10, pady=20)
    manual_button.pack(side="left", expand=True, padx=10, pady=20)
    is_data_folder_button.pack(side="left", expand=True, padx=10, pady=20)

    # Preselect the "Yes" button and bind Enter key to its action
    yes_button.focus_set()
    dialog.bind('<Return>', lambda event: on_select("Yes"))

    dialog.attributes('-topmost', 0)
    time.sleep(0.001)
    dialog.attributes('-topmost', 1)
    dialog.focus_force()

    # Block execution until the dialog is closed
    dialog.mainloop()

    return user_choice


