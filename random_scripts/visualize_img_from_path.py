import os
import random
import numpy as np
import matplotlib.pyplot as plt

def collect_patient_npy_files(main_dir):
    npy_files = []
    for entry in os.listdir(main_dir):
        patient_path = os.path.join(main_dir, entry)
        if os.path.isdir(patient_path):
            files = os.listdir(patient_path)
            npy_candidates = [f for f in files if f.endswith('.npy')]
            if npy_candidates:
                # Just take the first .npy file in case there's more than one
                npy_path = os.path.join(patient_path, npy_candidates[0])
                npy_files.append(npy_path)
    return npy_files

def show_images(image_paths, max_images=30):
    selected = random.sample(image_paths, min(len(image_paths), max_images))
    images = [np.load(path) for path in selected]

    cols = 6
    rows = (len(images) + cols - 1) // cols

    plt.figure(figsize=(15, 2.5 * rows))
    for idx, img in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        elif img.ndim == 3:
            plt.imshow(np.squeeze(img))
        else:
            plt.imshow(img.reshape(int(np.sqrt(img.size)), -1), cmap='gray')
        plt.axis('off')
        plt.title(f"Image {idx+1}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_dir = input("Enter the main directory path: ").strip()
    npy_paths = collect_patient_npy_files(main_dir)
    if not npy_paths:
        print("No .npy files found in patient directories.")
    else:
        show_images(npy_paths)
