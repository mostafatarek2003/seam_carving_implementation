import numpy as np
from imageio.v2 import imread, imwrite  # Updated import
import matplotlib.pyplot as plt

def calc_energy(img):
    """Calculate energy using the required formula e1 = |∂/∂x I| + |∂/∂y I|"""
    # Convert to grayscale if colored
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    # Calculate x and y derivatives manually
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)

    # X derivative
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    dx[:, 0] = gray[:, 1] - gray[:, 0]
    dx[:, -1] = gray[:, -1] - gray[:, -2]

    # Y derivative
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    dy[0, :] = gray[1, :] - gray[0, :]
    dy[-1, :] = gray[-1, :] - gray[-2, :]

    # Calculate energy
    energy = np.abs(dx) + np.abs(dy)
    return energy

def find_seam(energy):
    h, w = energy.shape

    # Dynamic programming to find minimum energy path
    dp = energy.copy()
    backtrack = np.zeros_like(energy, dtype=np.int32)  # Changed from np.int to np.int32

    for i in range(1, h):
        for j in range(w):
            if j == 0:
                idx = np.argmin(dp[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = dp[i-1, idx + j]
            else:
                idx = np.argmin(dp[i-1, max(0, j-1):min(w, j+2)])
                backtrack[i, j] = idx + max(0, j-1)
                min_energy = dp[i-1, idx + max(0, j-1)]

            dp[i, j] += min_energy

    # Backtrack to find the seam
    seam = np.zeros(h, dtype=np.int32)  # Changed from np.int to np.int32
    seam[-1] = np.argmin(dp[-1])
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]

    return seam

def remove_seam(img, seam):
    h, w = img.shape[:2]
    output = np.zeros((h, w-1) + img.shape[2:], dtype=img.dtype)

    for i in range(h):
        col = seam[i]
        output[i, :col] = img[i, :col]
        output[i, col:] = img[i, col+1:]

    return output

def seam_carving(image, scale_c):
    img = image.copy()
    seams_visualization = image.copy()
    h, w = img.shape[:2]
    new_w = int(w * scale_c)
    seams_to_remove = w - new_w

    for i in range(seams_to_remove):
        energy = calc_energy(img)
        seam = find_seam(energy)

        # Visualize seam
        for row in range(h):
            seams_visualization[row, seam[row]] = [255, 0, 0]  # Red color for seam

        img = remove_seam(img, seam)

    return img, seams_visualization

def main():
    # Read image
    input_image = imread('input.jpg')

    # Reduce width to half
    output_image, seams_visual = seam_carving(input_image, 0.5)

    # Save results
    imwrite('output_resized.jpg', output_image)
    imwrite('output_seams.jpg', seams_visual)

if __name__ == '__main__':
    main()
