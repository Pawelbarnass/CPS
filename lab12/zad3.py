# exercise_3_contour_detection.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import pooch

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

def run_exercise_3():
    """Main function to run all parts of Exercise 3."""
    print("--- Running Exercise 3: Object Contour Detection ---")

    # Wczytaj lokalne obrazy car1 i car2
    car1 = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\car1.jpg")
    car2 = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\car2.jpg")
    if car1 is None or car2 is None:
        raise FileNotFoundError("Nie znaleziono car1.jpg lub car2.jpg w katalogu lab12.")

    # Przetwarzaj car1 (możesz analogicznie car2)
    car_image = car2
    car_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

    # --- 3.1: Gaussian Blur ---
    print("\n--- 3.1: Gaussian Blur ---")
    car_blurred = cv2.GaussianBlur(car_gray, (5, 5), 0)
    show_images([car_gray, car_blurred], ["Original Grayscale", "Blurred"])

    # --- 3.2: Binarization ---
    print("\n--- 3.2: Binarization ---")
    threshold_value = 130
    _, car_binary = cv2.threshold(car_blurred, threshold_value, 255, cv2.THRESH_BINARY)
    show_images([car_blurred, car_binary], ["Blurred Image", f"Binary (Thresh={threshold_value})"])

    # --- 3.3: Edge Detection ---
    print("\n--- 3.3: Edge Detection ---")
    sobel_x = cv2.Sobel(car_blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(car_blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.uint8(255 * np.sqrt(sobel_x**2 + sobel_y**2) / np.max(np.sqrt(sobel_x**2 + sobel_y**2)))
    canny_edges = cv2.Canny(car_blurred, threshold1=100, threshold2=200)
    show_images([car_gray, sobel_mag, canny_edges], ["Original", "Sobel Edges", "Canny Edges"])
    print("Canny is generally more robust for edge detection.")

    # --- 3.4 (Optional): Morphological Operations ---
    print("\n--- 3.4 (Optional): Morphological Operations to Isolate Plate ---")
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(car_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = remove_small_objects(closed.astype(bool), min_size=150)
    cleaned_uint8 = (cleaned * 255).astype(np.uint8)
    filled = binary_fill_holes(cleaned).astype(np.uint8) * 255
    show_images(
        [car_binary, closed, cleaned_uint8, filled],
        ["Initial Binary", "After Closing", "After Removing Small Objects", "After Filling Holes"]
    )
    print("The goal is to produce a solid mask for the license plate.")

if __name__ == "__main__":
    run_exercise_3()