import cv2
import numpy as np
import os
import sys
from pathlib import Path

def genDiag(nR, nC, valUpper, valDiag, valLower, line=3):
    slope = nC / nR
    tbl = np.full((nR, nC), valDiag, dtype=float)
    for r in range(nR):
        tbl[r, 0 : int(round(slope * (r - line), 0))] = valLower
        tbl[r, int(round(slope * (r + line), 0)) : nC] = valUpper
    return tbl

def genCheckerboard(nR, nC, valUpper, valLower, line=0):
    # create a nR * nC mask
    tile_size=50
    mask = np.full((nR, nC), valUpper, dtype=int)
    small_mask = np.full((int(nR/tile_size)+1, int(nC/tile_size)+1), valUpper, dtype=int)

    # fill with 1 the alternate cells in rows and columns
    small_mask[1::2, ::2] = valLower
    small_mask[::2, 1::2] = valLower
    temp_mask = np.kron(small_mask, np.ones((tile_size, tile_size)))
    mask = temp_mask[tuple(slice(0, n) for n in mask.shape)]
    # mask[temp_mask==1] = valUpper
    # mask[temp_mask==0] = valLower
    return mask

def merge_images_and_videos(input1, input2, output_path, checkerboard):
    # Function to check if the input is an image or video
    def is_image(file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']

    def is_video(file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in ['.mp4', '.avi', '.mkv', '.mov']

    # Read inputs
    if is_image(input1) and is_image(input2):
        img1 = cv2.imread(input1)
        img2 = cv2.imread(input2)

        if img1.shape != img2.shape:
            raise ValueError("Input images must be of the same size")

        # Create a mask
        w = int(img1.shape[0])
        h = int(img1.shape[1])

        if checkerboard:
            mask = genCheckerboard(w, h, 1, -1)[..., None]
        else:
            mask = genDiag(w, h, 1, 0, -1)[..., None]
        mask1 = mask == 1
        mask2 = mask == -1
        mask_diag = mask == 0

        # Merge images along the diagonal
        merged_image = np.zeros_like(img1)
        merged_image += img1 * mask1
        # merged_image += np.uint8([0, 0, 0]) * mask1
        merged_image += img2 * mask2
        # merged_image += np.uint8([255, 255, 255]) * mask2
        merged_image += np.uint8([255, 255, 255]) * mask_diag

        cv2.imwrite(output_path, merged_image)

    elif is_video(input1) and is_video(input2):
        cap1 = cv2.VideoCapture(input1)
        cap2 = cv2.VideoCapture(input2)

        if (cap1.get(3), cap1.get(4)) != (cap2.get(3), cap2.get(4)):
            raise ValueError("Input videos must have the same resolution")

        h = int(cap1.get(3))
        w = int(cap1.get(4))

        if checkerboard:
            mask = genCheckerboard(w, h, 1, -1)[..., None]
        else:
            mask = genDiag(w, h, 1, 0, -1)[..., None]
        mask1 = mask == 1
        mask2 = mask == -1
        mask_diag = mask == 0


        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPEG codec
        # fourcc = 0x00000021
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap1.get(3)), int(cap1.get(4))))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Merge frames along the diagonal
            merged_frame = np.zeros_like(frame1)
            merged_frame += frame1 * mask1
            # merged_frame += 0 * mask1
            merged_frame += frame2 * mask2
            # merged_frame += 255 * mask2
            merged_frame += np.uint8([255, 255, 255]) * mask_diag

            out.write(merged_frame)

        cap1.release()
        cap2.release()
        out.release()

    else:
        raise ValueError("Invalid input types. Please provide either two images or two videos.")

def merge_images_in_dir(image_dir1, image_dir2, output_dir, checkerboard):
    # Get list of image files in both directories
    images1 = os.listdir(image_dir1)
    images2 = os.listdir(image_dir2)

    if len(images1) == 1:
        images1 = images1 * len(images2)
    if len(images2) == 1:
        images2 = images2 * len(images1)
    # print(images1)
    # print(images2)

    # Ensure both directories contain the same number of images
    if len(images1) != len(images2):
        print("Error: Directories must contain the same number of images.")
        return

    # Iterate over images and merge them
    for image_name1, image_name2 in zip(images1, images2):
        output_path = os.path.join(output_dir, f"merged_{image_name1}")
        image_path1 = os.path.join(image_dir1, image_name1)
        image_path2 = os.path.join(image_dir2, image_name2)

        # Merge images (assuming merge_images function is defined elsewhere)
        merge_images_and_videos(image_path1, image_path2, output_path, checkerboard)

        print(f"Merged {image_name1} and {image_name2} successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py input1_path input2_path output_path checkerboard")
        sys.exit(1)

    input_path1 = sys.argv[1]
    input_path2 = sys.argv[2]
    output_path = sys.argv[3]
    checkerboard = sys.argv[4] == "True"

    if os.path.isdir(input_path1) and os.path.isdir(input_path2):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        merge_images_in_dir(input_path1, input_path2, output_path, checkerboard)
    else:
        merge_images_and_videos(input_path1, input_path2, output_path, checkerboard)
