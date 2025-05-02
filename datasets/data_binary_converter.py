import os
import shutil

def convert_to_binary_classes(input_dir, output_dir, healthy_classes, diseased_classes):
    """
    Convert a multi-class dataset into binary (healthy vs diseased).

    Args:
        input_dir (str): Path to the original dataset with multiple class folders.
        output_dir (str): Path where binary-labeled folders ('healthy', 'diseased') will be saved.
        healthy_classes (list): List of folder names that are healthy.
        diseased_classes (list): List of folder names that are diseased.
    """
    healthy_path = os.path.join(output_dir, "healthy")
    diseased_path = os.path.join(output_dir, "diseased")

    os.makedirs(healthy_path, exist_ok=True)
    os.makedirs(diseased_path, exist_ok=True)

    for class_name in healthy_classes:
        class_folder = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"⚠️  Warning: {class_folder} does not exist")
            continue

        for img_name in os.listdir(class_folder):
            src = os.path.join(class_folder, img_name)
            dst = os.path.join(healthy_path, img_name)
            shutil.copyfile(src, dst)

    for class_name in diseased_classes:
        class_folder = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"⚠️  Warning: {class_folder} does not exist")
            continue

        for img_name in os.listdir(class_folder):
            src = os.path.join(class_folder, img_name)
            dst = os.path.join(diseased_path, img_name)
            shutil.copyfile(src, dst)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert multiclass PlantVillage dataset to binary (healthy/diseased)")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to original dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output binary dataset')
    args = parser.parse_args()

    healthy_classes = [
        "Apple___healthy", "Tomato___healthy", "Potato___healthy", "Strawberry___healthy", 
        "Corn_(maize)___healthy", "Pepper,_bell___healthy", "Grape___healthy", 
        "Blueberry___healthy", "Soybean___healthy", "Cherry_(including_sour)___healthy",
        "Raspberry___healthy", "Peach___healthy"
    ]

    diseased_classes = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Potato___Early_blight", "Potato___Late_blight", "Strawberry___Leaf_scorch",
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
        "Pepper,_bell___Bacterial_spot", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", 
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)", 
        "Squash___Powdery_mildew", "Peach___Bacterial_spot"
    ]

    convert_to_binary_classes(args.input_dir, args.output_dir, healthy_classes, diseased_classes)