import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from skimage import data

# --- Helper Functions ---
def dct2(a):
    """Performs a 2D DCT."""
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def show_images(images, titles, main_title=""):
    """Displays multiple images in a single figure."""
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_exercise_2():
    """Main function to run all parts of Exercise 2."""
    print("--- Running Exercise 2: Image Filtering ---")

    # --- 2.1: 2D FIR Filtering ---
    print("\n--- 2.1: 2D FIR Filtering ---")
    # 1. Generate filter kernels
    win_1d = np.hanning(32)
    lp_kernel = np.outer(win_1d, win_1d)
    lp_kernel /= np.sum(lp_kernel)

    hp_kernel = np.zeros((32, 32))
    hp_kernel[15, 15] = 1
    hp_kernel -= lp_kernel

    # 2. Plot kernels
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(range(32), range(32))
    ax1.plot_surface(X, Y, lp_kernel, cmap='viridis')
    ax1.set_title('LP Filter Kernel')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, hp_kernel, cmap='viridis')
    ax2.set_title('HP Filter Kernel')
    plt.show()

    # 3. Apply filters to an image
    # 3. Apply filters to an image
    image = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\lena512.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Nie znaleziono pliku lena512.png w katalogu lab12.")
    image_float = image.astype(float)
    image_lp = cv2.filter2D(image_float, -1, lp_kernel)
    image_hp = cv2.filter2D(image_float, -1, hp_kernel)

    # 4. Show image and spectrum before and after filtering
    dct_image = dct2(image_float)
    dct_image_lp = dct2(image_lp)
    dct_image_hp = dct2(image_hp)

    show_images(
        [image, image_lp, image_hp],
        ["Original Image", "After LP Filter (Blurred)", "After HP Filter (Edges)"],
        "Filtering in Spatial Domain"
    )
    show_images(
        [np.log(abs(dct_image)+1), np.log(abs(dct_image_lp)+1), np.log(abs(dct_image_hp)+1)],
        ["Original Spectrum", "LP Filtered Spectrum", "HP Filtered Spectrum"],
        "Filtering in Frequency Domain"
    )

    # --- 2.2: Gaussian Filter Analysis ---
    print("\n--- 2.2: Gaussian Filter Analysis ---")
    # Varying sigma
    ksize = (31, 31)
    blurred_s1 = cv2.GaussianBlur(image, ksize, sigmaX=1)
    blurred_s5 = cv2.GaussianBlur(image, ksize, sigmaX=5)
    blurred_s10 = cv2.GaussianBlur(image, ksize, sigmaX=10)
    show_images(
        [image, blurred_s1, blurred_s5, blurred_s10],
        ["Original", "Sigma=1", "Sigma=5", "Sigma=10"],
        f"Effect of Sigma (Kernel Size={ksize})"
    )

    # Varying kernel size
    sigma = 5
    blurred_k5 = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma)
    blurred_k15 = cv2.GaussianBlur(image, (15, 15), sigmaX=sigma)
    blurred_k31 = cv2.GaussianBlur(image, (31, 31), sigmaX=sigma)
    show_images(
        [image, blurred_k5, blurred_k15, blurred_k31],
        ["Original", "Kernel=5x5", "Kernel=15x15", "Kernel=31x31"],
        f"Effect of Kernel Size (Sigma={sigma})"
    )
    print("Observations: Larger sigma causes more blurring. For a given sigma, the kernel must be large enough to contain the Gaussian curve.")

if __name__ == "__main__":
    run_exercise_2()