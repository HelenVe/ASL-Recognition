import cv2
import os
import numpy as np
import time
import argparse


# code to get the optical flow from the whole images
# use -i all/train and -s optical_flow/train


def frobenius_norm(img1, img2):
    """Calculates the average pixel squared distance between 2 gray scale images."""
    return np.power(img2 - img1, 2).sum() / np.prod(img1.shape)


def optical_flow(data_dir, save_dir):
    start = time.time()
    subdir = sorted(os.listdir(data_dir))
    for path in subdir:
        output_dir = os.path.join(save_dir + path)  # file structure we want to keep in features folder
        subfolder = os.path.join(data_dir, path)  # [every subfolder]
        #  create subfolders if they dont exist inside features folder

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        count = 0
        for img in sorted(os.listdir(subfolder)):  # list the images

            img_folders = os.path.join(subfolder, img)  # keep a path variable

            # for filename in sorted(os.listdir(img_folders)):  # list all  frames
            #     images = os.path.join(img_folders, filename)  # keep a path
            #     print(images)

            if not (img.endswith(".jpg") or img.endswith(".jpeg")):
                print("File not an image")
                exit(-1)

            if count == 0:
                first_frame = cv2.imread(img_folders)

                # Creates an image filled with zero intensities with the same dimensions as the frame
                mask = np.zeros_like(first_frame)
                mask[..., 1] = 255  # set image saturation to maximum
                prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                count += 1
                continue

            gray = cv2.cvtColor(cv2.imread(img_folders), cv2.COLOR_BGR2GRAY)
            if frobenius_norm(prev_gray, gray) > 1:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Computes the magnitude and angle of the 2D vectors
                mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                if (mag.max() - mag.min()) == 0:
                    mag = np.zeros_like(mag)
                elif mag.max() == np.inf:
                    mag = np.nan_to_num(mag, copy=True, posinf=mag.min())
                    mag = (mag - mag.min()) / float(mag.max() - mag.min())
                else:
                    mag = (mag - mag.min()) / float(mag.max() - mag.min())

                # Sets image hue according to the optical flow direction
                mask[..., 0] = angle * 180 / np.pi / 2
                # Sets image value according to the optical flow magnitude (normalized)
                mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # Converts HSV to RGB (BGR) color representation
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

                prev_gray = gray
                save_path = os.path.join(save_dir, path)
                save_path = os.path.join(save_path, img)
                print(save_path)

                cv2.imwrite(save_path, rgb)

        print("------------------------------------------------------------->")
    end = time.time()
    cv2.destroyAllWindows()
    print("Elapsed time for optical flow: ", end - start)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
    ap.add_argument('-s', '--save', default="save_dir", help='Output images save directory')
    args = ap.parse_args()

    data_dir = args.images + "/"
    save_dir = args.save + "/"

    optical_flow(data_dir, save_dir)
