"""
MFAD Mini Project 2: Matrix operations and image manipulation
==============================================================
A grayscale image is just a matrix: every element is one pixel (0 = black,
255 = white). This script applies matrix operations (addition, subtraction,
scalar multiplication, transpose, slicing, concatenation, ...) to that matrix
to lighten/darken, crop, flip, rotate and change the contrast of the image.

Usage:
    python lab02.py                  # run everything, show windows, save PNGs
    python lab02.py --no-show        # run everything, only save PNGs to results/
    python lab02.py --image my.jpg   # use a different image
"""

import argparse
import os
import sys

import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
SHOW = True  # changed by --no-show
MAX_SIDE = 800  # the lion photo is 3264x4928; shrink it so the J-matrix bonus stays fast


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def show(images, titles, filename, figsize=None):
    """Display one or more images side by side and save the figure as a PNG."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=figsize or (5 * len(images), 5))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=120)
    if SHOW:
        plt.show()
    plt.close()


def clip_u8(a):
    """Clamp any numeric array into 0..255 and convert back to uint8.

    Needed because uint8 arithmetic WRAPS AROUND (e.g. 10 - 50 becomes 216),
    it does not go negative. So we compute in a bigger type, clip, then convert.
    """
    return np.clip(a, 0, 255).astype(np.uint8)


def load_image(path):
    """Load the image as a 2-D uint8 matrix (grayscale), downscaled to MAX_SIDE."""
    if not os.path.exists(path):
        print(f"[!] '{path}' not found. Put lion.png into this folder")
        print("    (see README). Using a built-in test image so the script still runs.\n")
        y, x = np.mgrid[0:300, 0:260]
        img = (x / 260 * 200 + 30 * np.sin(y / 15)).astype(float)
        img[100:200, 80:180] = 240  # a bright square so flips/rotations are visible
        return clip_u8(img)
    pil = Image.open(path)
    if max(pil.size) > MAX_SIDE:  # keep aspect ratio, longest side = MAX_SIDE
        scale = MAX_SIDE / max(pil.size)
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.LANCZOS)
    img = np.asarray(pil.convert("RGB") if pil.mode not in ("L", "RGB") else pil)
    if img.ndim == 3:  # colour image -> average the colour channels to get gray
        img = img[..., :3].mean(axis=2)
    return clip_u8(img)


# ----------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------
def main(image_path):
    # ---- Task 1: load the image into a matrix -------------------------------
    ImJPG = load_image(image_path)

    # ---- Task 2: dimensions --------------------------------------------------
    m, n = ImJPG.shape
    print(f"Q1: The dimensions of the image are {m} x {n} (rows x columns)")

    # ---- Task 3: integer type? -----------------------------------------------
    isInt = np.issubdtype(ImJPG.dtype, np.integer)
    print(f"Is ImJPG of integer type? {isInt}  (dtype = {ImJPG.dtype})")

    # ---- Task 4: range of colours ----------------------------------------------
    maxImJPG = np.max(ImJPG)
    minImJPG = np.min(ImJPG)
    print(f"Maximum pixel value: {maxImJPG}")
    print(f"Minimum pixel value: {minImJPG}")

    # ---- Task 5: display original ----------------------------------------------
    show([ImJPG], ["Lion Grayscale Image (original)"], "01_original.png", (5, 5))

    # ---- Task 6: crop central part (submatrix) ----------------------------------
    ImJPG_center = ImJPG[100:m - 100, 100:n - 70]
    show([ImJPG_center], ["Cropped Central Part"], "02_cropped.png", (5, 5))

    # ---- Task 7: paste the crop into a zero (black) matrix -----------------------
    ImJPG_border = np.zeros((m, n), dtype=np.uint8)
    ImJPG_border[100:m - 100, 100:n - 70] = ImJPG_center
    show([ImJPG_border], ["Image with Pasted Center"], "03_border.png", (5, 5))

    # ---- Task 8: vertical flip (reverse the order of rows) ------------------------
    ImJPG_vertflip = np.flipud(ImJPG)
    show([ImJPG, ImJPG_vertflip], ["Original Image", "Vertically Flipped Image"],
         "04_vertical_flip.png", (10, 5))

    # ---- Task 9: transpose (swap rows and columns) ---------------------------------
    ImJPG_transpose = ImJPG.T
    show([ImJPG, ImJPG_transpose], ["Original Image", "Transposed Image"],
         "05_transpose.png", (10, 5))

    # ---- Task 10: horizontal flip via transpose --------------------------------------
    # Flipping the COLUMNS of A is the same as flipping the ROWS of A.T, so:
    #   transpose -> flip rows (np.flipud) -> transpose back.
    # NOTE: the lab sheet uses np.fliplr on the transposed matrix here, but
    # (fliplr(A.T)).T is actually a VERTICAL flip, so we use flipud instead.
    t = ImJPG.T
    t = np.flipud(t)
    ImJPG_horflip = t.T
    assert np.array_equal(ImJPG_horflip, np.fliplr(ImJPG))  # same as direct fliplr
    show([ImJPG, ImJPG_horflip], ["Original Image", "Horizontally Flipped Image"],
         "06_horizontal_flip.png", (10, 5))

    # ---- Task 11: rot90 -----------------------------------------------------------------
    ImJPG90 = np.rot90(ImJPG)
    print("Q3: np.rot90 rotates the matrix by 90 degrees COUNTERCLOCKWISE.")
    show([ImJPG, ImJPG90], ["Original Image", "90 Degrees Rotated Image"],
         "07_rot90.png", (10, 5))

    # ---- Task 12: colour inversion ----------------------------------------------------------
    ImJPG_inv = 255 - ImJPG
    print("Q4: 255 - ImJPG subtracts every pixel from 255, so black (0) becomes "
          "white (255) and white becomes black: a photographic negative.")
    show([ImJPG, ImJPG_inv], ["Original Image", "Inverted Image"],
         "08_inverted.png", (10, 5))

    # ---- Task 13: darken / lighten ------------------------------------------------------------
    # Work in int16 so values can go below 0 / above 255, then clip.
    ImJPG_dark = clip_u8(ImJPG.astype(np.int16) - 50)
    ImJPG_light = clip_u8(ImJPG.astype(np.int16) + 50)
    print("Q5: To lighten the image, ADD a positive constant to every pixel "
          "(e.g. ImJPG + 50) and clip values above 255 to 255.")
    show([ImJPG, ImJPG_dark, ImJPG_light],
         ["Original Image", "Darkened Image", "Lightened Image"],
         "09_dark_light.png", (15, 5))

    # ---- Task 14: Andy Warhol style 2x2 block matrix -----------------------------------------------
    A = ImJPG.astype(np.int16)
    top_left = ImJPG
    top_right = clip_u8(A - 50)
    bottom_left = clip_u8(A + 100)
    bottom_right = clip_u8(A + 50)
    top_row = np.concatenate((top_left, top_right), axis=1)
    bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)
    ImJPG_Warhol = np.concatenate((top_row, bottom_row), axis=0)
    show([ImJPG_Warhol], ["Andy Warhol Style Image"], "10_warhol.png", (8, 8))

    # ---- Task 15: naive black and white ----------------------------------------------------------------
    # floor(x/128) is 0 for x < 128 and 1 for x >= 128 (and 1 for 255), then x255.
    ImJPG_bw = np.uint8(255 * np.floor(ImJPG / 128))
    show([ImJPG, ImJPG_bw], ["Original Image", "Black and White Image"],
         "11_black_white.png", (10, 5))

    # ---- Task 16: reduce 256 shades to 8 ------------------------------------------------------------------
    normalized = ImJPG / 255.0                     # range [0, 1]
    reduced = np.round(normalized * 7)             # integers 0..7
    ImJPG8 = np.uint8(reduced * (255 / 7))         # back to 0..255 (8 levels)
    print(f"Task 16: number of distinct gray levels now = {len(np.unique(ImJPG8))}")
    show([ImJPG, ImJPG8], ["Original (256 shades)", "Reduced to 8 shades"],
         "12_eight_shades.png", (10, 5))

    # ---- Task 17: contrast (scalar multiplication) ------------------------------------------------------------
    contrast_factor = 1.25
    ImJPG_HighContrast = clip_u8(ImJPG * contrast_factor)
    show([ImJPG, ImJPG_HighContrast],
         ["Original Image", f"Contrast x{contrast_factor}"],
         "13_contrast.png", (10, 5))

    # ---- Task 18: gamma correction (non-linear) ------------------------------------------------------------------
    # gamma < 1 brightens the mid-tones, gamma > 1 darkens them.
    ImJPG_Gamma05 = clip_u8(((ImJPG / 255.0) ** 0.5) * 255)
    ImJPG_Gamma15 = clip_u8(((ImJPG / 255.0) ** 1.5) * 255)
    show([ImJPG, ImJPG_Gamma05, ImJPG_Gamma15],
         ["Original Image", "Gamma = 0.5", "Gamma = 1.5"],
         "14_gamma.png", (15, 5))

    # ---- Bonus: flips as MATRIX MULTIPLICATION (linear algebra link) ------------------------------------------------
    # The "exchange matrix" J has 1s on the anti-diagonal. Then
    #     J_m @ A  reverses the ROWS    (vertical flip)
    #     A @ J_n  reverses the COLUMNS (horizontal flip)
    A = ImJPG.astype(float)
    J_m = np.fliplr(np.eye(m))
    J_n = np.fliplr(np.eye(n))
    vflip_matrix = clip_u8(J_m @ A)
    hflip_matrix = clip_u8(A @ J_n)
    print("Bonus: J @ A equals np.flipud(A)?", np.array_equal(vflip_matrix, np.flipud(ImJPG)))
    print("Bonus: A @ J equals np.fliplr(A)?", np.array_equal(hflip_matrix, np.fliplr(ImJPG)))
    show([ImJPG, vflip_matrix, hflip_matrix],
         ["Original", "J_m @ A (vertical flip)", "A @ J_n (horizontal flip)"],
         "15_flip_by_matrix_multiplication.png", (15, 5))

    print(f"\nDone. All figures are saved in the '{RESULTS_DIR}/' folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFAD Project 2: matrix operations on images")
    parser.add_argument("--image", default="lion.png", help="path to a grayscale/colour image (png/jpg)")
    parser.add_argument("--max-side", type=int, default=800, help="shrink the image so its longest side is at most this many pixels")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()
    MAX_SIDE = args.max_side
    if args.no_show:
        SHOW = False
    sys.exit(main(args.image))
