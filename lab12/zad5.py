import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
# from skimage.metrics import mutual_information  # ← usuń lub zakomentuj tę linię
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import pooch
def mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_information_images(img1, img2, bins=256):
    hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    return mutual_information(hgram)
# --- Helper Functions ---
def show_images(images, titles, main_title=""):
    """Displays multiple images in a single figure."""
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        if images[i].ndim == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_exercise_5():
    """Main function to run all parts of Exercise 5."""
    print("--- Running Exercise 5: Image Registration (Optional) ---")

    # --- Part 1: Automatic License Plate Detection ---
    print("\n--- Part 1: Automatic License Plate Detection ---")
    car_image = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\car2.jpg")
    if car_image is None:
        raise FileNotFoundError("Nie znaleziono pliku car2.jpg w katalogu lab12.")
    car_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    
    # Simplified detection pipeline
    blurred = cv2.GaussianBlur(car_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    cleaned = remove_small_objects(closed.astype(bool), min_size=150)
    plate_candidate_mask = binary_fill_holes(cleaned).astype(np.uint8) * 255

    # Use regionprops to find the best candidate
    labeled_mask = label(plate_candidate_mask)
    props = regionprops(labeled_mask)
    
    best_candidate = None
    for region in props:
        minr, minc, maxr, maxc = region.bbox
        h, w = maxr - minr, maxc - minc
        area = region.area
        aspect_ratio = w / h if h > 0 else 0
        # Dodaj warunek na minimalny rozmiar (np. area > 1000)
        if 3 < aspect_ratio < 6 and region.solidity > 0.9 and area > 100:
            best_candidate = region
        break
    
    if best_candidate:
        minr, minc, maxr, maxc = best_candidate.bbox
        # 1. Maska tablicy
        mask = np.zeros_like(car_gray, dtype=np.uint8)
        mask[labeled_mask == best_candidate.label] = 1

        # 2. Wymnożenie maski z oryginałem i przypisanie tłu wartości 255
        tablica_only = car_gray.copy()
        tablica_only[mask == 0] = 255  # tło na biało

        # 3. Wycięcie ramki z tablicą (nie maską!)
        detected_plate = tablica_only[minr:maxr, minc:maxc]

        # 4. Wizualizacja
        show_images(
            [car_image, mask*255, tablica_only, detected_plate],
            ["Original Car", "Detected Plate Mask", "Plate on White BG", "Extracted Plate"]
        )
    else:
        print("No suitable license plate candidate found automatically. Using a manual crop for demonstration.")
        detected_plate = car_gray[120:150, 90:220] # Fallback

    # --- Part 2: Registration with Mutual Information ---
    print("\n--- Part 2: Registration with Mutual Information ---")
    # Create a synthetic template (tab_wz.jpg)
    template_plate = np.zeros((40, 160), dtype=np.uint8)
    cv2.putText(template_plate, "TEMPLATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Prepare the detected ("moving") image
    moving_plate = cv2.resize(detected_plate, (160, 40))

    # Calculate Mutual Information for the current alignment
    mi_initial = mutual_information_images(template_plate, moving_plate)
    print(f"Initial Mutual Information: {mi_initial:.4f}")

    # Demonstrate how MI changes with a transformation (e.g., a shift)
    M_shift = np.float32([[1, 0, 10], [0, 1, 5]]) # Shift 10px right, 5px down
    shifted_plate = cv2.warpAffine(moving_plate, M_shift, (160, 40))
    mi_shifted = mutual_information_images(template_plate, shifted_plate)
    
    show_images(
        [template_plate, moving_plate, shifted_plate],
        ["Template", f"Detected (MI={mi_initial:.2f})", f"Shifted (MI={mi_shifted:.2f})"]
    )
    print("A full registration algorithm would use an optimizer to find the transformation that MAXIMIZES mutual information.")

if __name__ == "__main__":
    run_exercise_5()