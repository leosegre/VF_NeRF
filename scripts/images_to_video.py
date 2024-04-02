import os
import cv2
import argparse

def create_video_from_images(image_folder, video_name='output_video.mp4', fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # Sort images based on their numeric part (assuming filenames like "merged_1.png", "merged_2.png")
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Determine the width and height from the first image
    image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to a video.')
    parser.add_argument('image_folder', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Name of the output video file (default: output_video.mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video (default: 30)')
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output, args.fps)
