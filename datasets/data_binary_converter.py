import os
import shutil

def convert_to_binary(input_dir, output_dir, healthy_keyword='healthy'):
    """
    Convert a multi-class dataset to binary by labeling:
    - folders containing 'healthy' as 'healthy'
    - all others as 'diseased'

    Args:
        input_dir (str): Input folder with class-wise subfolders
        output_dir (str): Where binary-labeled data will be written
        healthy_keyword (str): Keyword to identify healthy classes
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        label = 'healthy' if healthy_keyword in class_name.lower() else 'diseased'
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        for img_file in os.listdir(class_path):
            src = os.path.join(class_path, img_file)
            dst = os.path.join(label_dir, img_file)
            shutil.copy2(src, dst)
