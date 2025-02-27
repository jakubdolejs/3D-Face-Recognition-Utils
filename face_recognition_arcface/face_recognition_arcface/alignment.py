import numpy as np
import math
import cv2

class LinearRegression:
    def __init__(self):
        self.data = []
        self.results = []

    def add(self, result, x, y, a, b):
        self.data.append([x, y, a, b])
        self.results.append(result)

    def compute(self):
        A = np.array(self.data)
        B = np.array(self.results)
        # Solve Ax = B using least squares
        c, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return c


class RotatedBox:
    def __init__(self, center, angle, width, height):
        self.center = center
        self.angle = angle
        self.width = width
        self.height = height


def align_face(pts, scale=1.0):
    """
    Align face based on 5-point facial landmarks.
    :param pts: List of 5 facial landmarks [Point(x, y)]
    :param scale: Scaling factor for the face box
    :return: RotatedBox object with alignment information
    """
    reg = LinearRegression()

    # Parameters for alignment
    yofs = 0.35  # Adjust as necessary for specific datasets
    y0 = yofs - 0.5  # Eyes level
    y1 = yofs + 0.04  # Nose level
    y2 = yofs + 0.5  # Mouth level

    # Add left eye
    reg.add(pts[0][0],   -0.46,   -y0,  1.0,  0.0)
    reg.add(pts[0][1],      y0, -0.46,  0.0,  1.0)

    # Add right eye
    reg.add(pts[1][0],   0.46,    -y0,  1.0,  0.0)
    reg.add(pts[1][1],     y0,   0.46,  0.0,  1.0)

    # Add nose tip
    reg.add(pts[2][0],    0.0,    -y1,  1.0,  0.0)
    reg.add(pts[2][1],     y1,    0.0,  0.0,  1.0)

    if len(pts) == 4:
        # Mouth center
        reg.add(pts[3][0], 0.0, -y2, 1.0, 0.0)
        reg.add(pts[3][1], y2, 0.0, 0.0, 1.0)
    else:
        # Left mouth corner
        reg.add(pts[3][0], -0.39, -y2, 1.0, 0.0)
        reg.add(pts[3][1], y2, -0.39, 0.0, 1.0)

        # Right mouth corner
        reg.add(pts[4][0], 0.39, -y2, 1.0, 0.0)
        reg.add(pts[4][1], y2, 0.39, 0.0, 1.0)

    # Compute the alignment coefficients
    c = reg.compute()
    assert len(c) == 4, "Unexpected coefficient size"

    # Calculate rotated box parameters
    center_x, center_y = c[2], c[3]
    angle = math.atan2(c[1], c[0])
    width_height = scale * math.sqrt(c[0] ** 2 + c[1] ** 2)

    return RotatedBox(center=(center_x, center_y), angle=angle, width=width_height, height=width_height)


def crop_face(image, rotated_box, target_size=(112, 112)):
    """
    Crop the aligned face from the input image based on the RotatedBox alignment information.
    :param image: Input image as a numpy array.
    :param rotated_box: RotatedBox object with alignment information.
    :param target_size: Tuple (width, height) for the output cropped face.
    :return: Cropped and aligned face as a numpy array.
    """
    # Get alignment parameters
    center = rotated_box.center
    angle = rotated_box.angle * 180.0 / np.pi  # Convert angle to degrees
    scale = rotated_box.width / target_size[0]  # Scaling factor to fit the target size

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0 / scale)

    # Warp the image to align the face
    warped_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Calculate the cropping box
    crop_x = int(center[0] - target_size[0] // 2)
    crop_y = int(center[1] - target_size[1] // 2)
    cropped_face = warped_image[crop_y:crop_y + target_size[1], crop_x:crop_x + target_size[0]]

    return cropped_face



def warp_face(img, kps):

    aligned_box = align_face(kps, scale=2.85)
    aligned_face = crop_face(img, aligned_box)
    return aligned_face

