# exercise_4_jpeg_compression.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from skimage import data
from skimage.util import view_as_blocks

# --- JPEG-related Helper Functions ---

Q_LUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
])

def dct2(a): return dct(dct(a.T, norm='ortho').T, norm='ortho')
def idct2(a): return idct(idct(a.T, norm='ortho').T, norm='ortho')

def zigzag(matrix):
    rows, cols = matrix.shape
    solution = [[] for _ in range(rows + cols - 1)]
    for i in range(rows):
        for j in range(cols):
            sum_idx = i + j
            if sum_idx % 2 == 0:
                solution[sum_idx].insert(0, matrix[i, j])
            else:
                solution[sum_idx].append(matrix[i, j])
    return np.concatenate(solution)

def jpeg_encode(img, q_factor=50):
    h, w = img.shape
    h_pad, w_pad = (8 - h % 8) % 8, (8 - w % 8) % 8
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad)), mode='edge')
    
    S = 50.0 / q_factor if q_factor < 50 else (100 - q_factor) / 50.0
    S = 0.001 if S == 0 else S
    Q_scaled = np.clip(Q_LUM * S, 1, 255)
    
    encoded_data = []
    blocks = view_as_blocks(img_padded, block_shape=(8, 8))
    for r in range(blocks.shape[0]):
        for c in range(blocks.shape[1]):
            block = blocks[r, c].astype(float) - 128
            dct_block = dct2(block)
            quantized_block = np.round(dct_block / Q_scaled).astype(int)
            encoded_data.append(zigzag(quantized_block))
            
    return np.array(encoded_data), Q_scaled, (h, w)

def jpeg_decode(encoded_data, Q_scaled, original_shape):
    h_orig, w_orig = original_shape
    h_pad, w_pad = (8 - h_orig % 8) % 8, (8 - w_orig % 8) % 8
    h_padded, w_padded = h_orig + h_pad, w_orig + w_pad
    
    reconstructed_img = np.zeros((h_padded, w_padded))
    block_idx = 0
    for r in range(h_padded // 8):
        for c in range(w_padded // 8):
            zigzag_vec = encoded_data[block_idx]
            quantized_block = np.zeros((8, 8))
            # Simplified inverse zigzag (easier to implement)
            it = np.nditer(quantized_block, flags=['multi_index'], op_flags=['writeonly'])
            for i, val in enumerate(it):
                quantized_block[it.multi_index] = zigzag_vec[i]
            
            dequantized_block = quantized_block * Q_scaled
            idct_block = idct2(dequantized_block)
            block = idct_block + 128
            reconstructed_img[r*8:(r+1)*8, c*8:(c+1)*8] = block
            block_idx += 1
            
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    return reconstructed_img[:h_orig, :w_orig].astype(np.uint8)

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def show_images(images, titles):
    plt.figure(figsize=(10, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def run_exercise_4():
    """Main function to run all parts of Exercise 4."""
    print("--- Running Exercise 4: JPEG-style Compression ---")

    image = cv2.imread(r"C:\Users\wiedzmok\CPS\lab12\barbara512.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Nie znaleziono pliku lena512.png w katalogu lab12.")

    q_factor = 80 # Low quality, high compression
    encoded, Qs, orig_shape = jpeg_encode(image, q_factor=q_factor)
    decoded = jpeg_decode(encoded, Qs, orig_shape)
    psnr = calculate_psnr(image, decoded)

    show_images([image, decoded], ["Original", f"Decoded (q={q_factor}, PSNR={psnr:.2f} dB)"])

    print("\nPlotting PSNR vs. Quantization Factor...")
    q_factors = np.arange(1, 100, 5)
    psnrs = [calculate_psnr(image, jpeg_decode(*jpeg_encode(image, q))[0]) for q in q_factors]

    plt.figure()
    plt.plot(q_factors, psnrs, marker='o')
    plt.title('Image Quality vs. JPEG Quantization Factor')
    plt.xlabel('Quantization Factor (1=Best, 99=Worst)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    run_exercise_4()