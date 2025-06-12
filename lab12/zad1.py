# exercise_1_orthogonal_transform.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from skimage import data
# --- Helper Functions ---
def dct2(a):
    """Performs a 2D DCT."""
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    """Performs a 2D Inverse DCT."""
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

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

def run_exercise_1():
    """Main function to run all parts of Exercise 1."""
    print("--- Running Exercise 1: Orthogonal Transformation ---")

    # --- 1.1: DCT and Zeroing Coefficients ---
    print("\n--- 1.1: DCT of an image and zeroing coefficients ---")
    im1 = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\im1.png", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\im2.png", cv2.IMREAD_GRAYSCALE)
    if im1 is None or im2 is None:
        raise FileNotFoundError("Nie znaleziono im1.png lub im2.png w katalogu lab12.")
    im1_float = im1.astype(float)
    dct_im1 = dct2(im1_float)
    dct_log_viz = np.log(np.abs(dct_im1) + 1)
    show_images([im1, dct_log_viz], ["Original im1.png", "Log of DCT Coefficients"])

    # Zero out half of the significant AC coefficients
    dct_im1_modified = dct_im1.copy()
    flat_ac_coeffs = dct_im1_modified[1:].flatten()
    indices_to_zero = np.argsort(np.abs(flat_ac_coeffs))[:len(flat_ac_coeffs) // 2]
    np.put(flat_ac_coeffs, indices_to_zero, 0)
    dct_im1_modified[1:] = flat_ac_coeffs.reshape(dct_im1_modified[1:].shape)
    im1_reconstructed = idct2(dct_im1_modified)
    show_images([im1, im1_reconstructed], ["Original Image", "Reconstructed after zeroing 50% AC Coeffs"])

    # --- 1.2: DCT Filtering on cameraman.tif ---
    print("\n--- 1.2: DCT Filtering ---")
    cameraman = data.camera()
    cameraman_float = cameraman.astype(float)
    dct_cameraman = dct2(cameraman_float)

    # a) Low-pass filter (keep top-left quadrant)
    dct_low_pass = np.zeros_like(dct_cameraman)
    rows, cols = cameraman.shape
    dct_low_pass[:rows // 4, :cols // 4] = dct_cameraman[:rows // 4, :cols // 4]
    reconstructed_low_freq = idct2(dct_low_pass)

    # b) Thresholding filter
    threshold = 500
    dct_thresholded = dct_cameraman.copy()
    dct_thresholded[np.abs(dct_thresholded) < threshold] = 0
    reconstructed_thresholded = idct2(dct_thresholded)
    show_images(
        [cameraman, reconstructed_low_freq, reconstructed_thresholded],
        ["Original", "Reconstructed (Low Freq Only)", f"Reconstructed (Threshold > {threshold})"]
    )

    # --- 1.3: Generating DCT Basis Images ---
    print("\n--- 1.3: Generating and summing DCT Basis Images ---")
    size = 128
    im_sum = np.zeros((size, size))
    basis_coords = [(5, 10), (10, 5), (20, 20), (30, 15)]
    for r, c in basis_coords:
        dct_basis = np.zeros((size, size))
        dct_basis[r, c] = 1
        im_sum += idct2(dct_basis)
    
    dct_sum = dct2(im_sum)
    dct_sum_viz = np.zeros_like(dct_sum)
    dct_sum_viz[np.abs(dct_sum) > 0.1] = 1
    show_images([im_sum, dct_sum_viz], ["Sum of 4 Basis Images", "DCT of Sum (should have 4 lit pixels)"])

    # Reconstruct from one coefficient
    dct_final = np.zeros((size, size))
    dct_final[5, 10] = dct_sum[5, 10]
    im_final = idct2(dct_final)
    show_images([im_sum, im_final], ["Sum of Basis Images", "Reconstructed from one coefficient"])

    # --- 1.4: Riddle ---
    print("\n--- 1.4: Riddle ---")
    # The riddle's answer is that the image itself is a DCT basis function.
    riddle_image_dct = np.zeros((128, 128))
    riddle_image_dct[10, 20] = 1 # A single frequency component
    riddle_image = idct2(riddle_image_dct)
    dct_of_riddle = dct2(riddle_image)
    dct_of_riddle_viz = np.log(np.abs(dct_of_riddle) + 1)
    show_images([riddle_image, dct_of_riddle_viz], ["Riddle Image (a cosine wave)", "Its DCT (one bright spot)"])
    print("Riddle Answer: The DCT is sparse because the image is a single DCT basis function.")


if __name__ == "__main__":
    run_exercise_1()