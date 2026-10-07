# MFAD Mini Project: Matrix Operations and Image Manipulation

**Course:** Mathematical Foundations (MFAD), PES University
**Team members:** <Name 1 (SRN)>, <Name 2 (SRN)>, <Name 3 (SRN)>, <Name 4 (SRN)>

## About the project
A grayscale image is stored as a matrix: each element is one pixel from
0 (black) to 255 (white). This project applies elementary linear algebra
operations to that matrix to manipulate the image:

| Operation | Matrix concept used |
|---|---|
| Crop / paste a region | Submatrix slicing, zero matrix |
| Lighten / darken | Matrix + constant matrix (with clipping to 0..255) |
| Invert colours | `255 - A` (constant matrix minus A) |
| Vertical / horizontal flip | Row / column reversal, exchange matrix `J` (`J @ A`, `A @ J`) |
| Transpose | `A.T` (swap rows and columns) |
| Rotate 90 degrees | `np.rot90` (transpose + flip) |
| Andy Warhol 2x2 art | Block matrices / concatenation |
| Contrast | Scalar multiplication `c * A` |
| Black and white, 8 shades | Element-wise floor / round |
| Gamma correction | Element-wise power (a non-linear operation) |

## Files
- `lab02.py`: the complete project code (all tasks, runs top to bottom)
- `requirements.txt`: Python libraries needed
- `lion.png`: the input image (a colour photo, converted to grayscale by the code)
- `results/`: output images, created automatically when you run the code

## How to set up and run
1. Install Python 3.9 or newer from https://www.python.org/downloads/
   (on Windows tick "Add Python to PATH" during installation).
2. Download this repository: green **Code** button, then **Download ZIP**, and unzip it
   (or run `git clone <repo-url>`). Open a terminal inside the project folder.
3. Make sure **lion.png** is in the project folder.
   (Any other image also works: `python lab02.py --image yourfile.jpg`)
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
5. Run the project:
   ```
   python lab02.py
   ```
   Each figure opens in a window; close a window to see the next one.
   Use `python lab02.py --no-show` to just save all pictures into `results/`.

(On Mac/Linux use `python3` and `pip3` if `python` does not work.)

## Output
The console prints the image dimensions, data type, pixel range and the answers
to the lab questions (Q1 to Q5). All result images are saved in `results/`.

## Note on image size
The original lion photo is 3264 x 4928 pixels. The code shrinks it so the longest side is
800 pixels (change with `--max-side`), because the exchange-matrix bonus builds
m x m matrices, which would need gigabytes of memory at full size.

## Note on the horizontal flip
The lab sheet flips the transposed matrix with `fliplr` and transposes back, but that
sequence is actually a *vertical* flip. We use `transpose -> flipud -> transpose`,
which is a true horizontal flip and is checked in code against `np.fliplr`.
Also, uint8 arithmetic wraps around (10 - 50 = 216), so lightening/darkening is done
in `int16` and then clipped to 0..255.
