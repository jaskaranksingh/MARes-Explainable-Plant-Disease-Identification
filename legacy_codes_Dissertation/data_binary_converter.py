import os
import shutil

def move_files(src_dirs, dest_dir):
    """
    Moves all files from the source directories to the destination directory.

    :param src_dirs: List of source directories.
    :param dest_dir: Destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for src_dir in src_dirs:
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.move(src_file, dest_file)
            print(f"Moved {src_file} to {dest_file}")

def main():
    # Define your directories
    folder1 = '/cs/home/psxjs24/data/PlantVillage/val/Apple___Apple_scab'
    folder2 = '/cs/home/psxjs24/data/PlantVillage/val/Apple___Black_rot'
    folder3 = '/cs/home/psxjs24/data/PlantVillage/val/Apple___Cedar_apple_rust'
    folder4 = '/cs/home/psxjs24/data/PlantVillage/val/Apple___healthy'

    # Define destination directories
    combined_folder = '/cs/home/psxjs24/data/PlantVillage/plantapple/val/disease'
    separate_folder = '/cs/home/psxjs24/data/PlantVillage/plantapple/val/healthy'

    # Move files from folder1, folder2, and folder3 to the combined folder
    move_files([folder1, folder2, folder3], combined_folder)

    # Move files from folder4 to the separate folder (if needed)
    # If you want to keep folder4 as it is, you can skip this step
    if not os.path.exists(separate_folder):
        os.makedirs(separate_folder)

    for filename in os.listdir(folder4):
        src_file = os.path.join(folder4, filename)
        dest_file = os.path.join(separate_folder, filename)
        shutil.move(src_file, dest_file)
        print(f"Moved {src_file} to {dest_file}")

if __name__ == "__main__":
    main()
