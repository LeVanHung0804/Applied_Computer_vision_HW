import numpy as np
from numpy.linalg import det, lstsq, norm
import cv2
from numpy import float32
from functools import cmp_to_key
import matplotlib.pyplot as plt

####################
# define parameter #
####################
TOTAL_OCTAVE            = 4
TOTAL_IMAGE_EACH_OCTAVE = 5
SIGMA                   = 1.6
K                       = np.sqrt(2.0)
TOTAL_DOG_EACH_OCTAVE   = 4
S                       = 2
BORDER_WIDTH            = 5
CONTRAST_THRESHOLD      = 0.04
EIGENVALUE_RATIO        = 10
TOTAL_BIN               = 36
GOOD_PEAK_RATIO         = 0.8
RADIUS_FACTOR           = 3
SCALE_FACTOR            = 1.5
BIN_WIDTH               = 10

#################
# gen all image #
#################
def gen_all_image(original_image):
# resize original image
    original_image = original_image.astype('float32')
    init_image = cv2.resize(original_image, (0,0), fx=2, fy=2, interpolation= cv2.INTER_LINEAR)

# init gaussian kernel of nth image 1,2,3,4
# 0 image have SIGMA kernel
    kernel = []
    for image_index in range(1, TOTAL_IMAGE_EACH_OCTAVE):
        last_sigma = (K ** (image_index - 1)) * SIGMA
        this_sigma = K * last_sigma
        this_kernel = np.sqrt(this_sigma ** 2 - last_sigma ** 2)
        kernel.append(this_kernel)
    kernel = np.array(kernel)

# gen all image in pyramid
    all_image = []
    for _ in range(TOTAL_OCTAVE):
        octave = [init_image]
        for nth_kernel in kernel:
            init_image = cv2.GaussianBlur(init_image, (0,0), sigmaX= nth_kernel, sigmaY= nth_kernel)
            octave.append(init_image)
        all_image.append(octave)
        next_octave_original_image = octave[2]
        init_image = cv2.resize(next_octave_original_image, (0,0), fx=1/2 , fy=1/2, interpolation= cv2.INTER_NEAREST)
    all_image = np.array(all_image)
    return all_image

###############
# gen all DoG #
###############
def gen_all_DoG(all_image):
    all_DoG = []
    for nth_octave in all_image:
        nth_DoG = []
        for image_index in range(1,TOTAL_IMAGE_EACH_OCTAVE):
            last_image = nth_octave[image_index]
            this_image = nth_octave[image_index-1]
            nth_DoG.append(this_image-last_image)
        all_DoG.append(nth_DoG)
    all_DoG = np.array(all_DoG)

    return all_DoG

##################
# get key points #
##################
def extrema_detection(slide0, slide1, slide2):
    rubik3x3 = [slide0,slide1,slide2]
    rubik3x3 = np.array(rubik3x3)
    if np.argmax(rubik3x3) == 13 or np.argmin(rubik3x3) == 13:
        return True
    return False

def localize_keypoint(rubik3x3):

    rubik3x3 = np.array(rubik3x3)

    dx = 0.5 * (rubik3x3[1, 1, 2] - rubik3x3[1, 1, 0])
    dy = 0.5 * (rubik3x3[1, 2, 1] - rubik3x3[1, 0, 1])
    ds = 0.5 * (rubik3x3[2, 1, 1] - rubik3x3[0, 1, 1])

    jacobian = np.array([dx,dy,ds])

    dxx = rubik3x3[1, 1, 2] - 2 * rubik3x3[1,1,1] + rubik3x3[1, 1, 0]
    dyy = rubik3x3[1, 2, 1] - 2 * rubik3x3[1,1,1] + rubik3x3[1, 0, 1]
    dss = rubik3x3[2, 1, 1] - 2 * rubik3x3[1,1,1] + rubik3x3[0, 1, 1]
    dxy = 0.25 * (rubik3x3[1, 2, 2] - rubik3x3[1, 2, 0] - rubik3x3[1, 0, 2] + rubik3x3[1, 0, 0])
    dxs = 0.25 * (rubik3x3[2, 1, 2] - rubik3x3[2, 1, 0] - rubik3x3[0, 1, 2] + rubik3x3[0, 1, 0])
    dys = 0.25 * (rubik3x3[2, 2, 1] - rubik3x3[2, 0, 1] - rubik3x3[0, 2, 1] + rubik3x3[0, 0, 1])

    tensor = np.array([[dxx, dxy, dxs],
                       [dxy, dyy, dys],
                       [dxs, dys, dss]])

    hessian = tensor[:2,:2]
    offset = -np.linalg.lstsq(tensor,jacobian,rcond=None)[0]
    contrast = rubik3x3[1,1,1] + 0.5*np.dot(jacobian,offset)
    trace_xy = np.trace(hessian) #trace
    det_xy = np.linalg.det(hessian) #det

    return jacobian, tensor, hessian, trace_xy, det_xy, offset, contrast

def find_one_keypoint(nth_DOG, i,j,image_index, octave_index, convergence =5):

    image_shape = nth_DOG[0].shape
    for attempt_index in range(convergence):
        slide0, slide1, slide2 = nth_DOG[image_index-1:image_index+2]
        rubik3x3 = np.stack([slide0[i-1:i+2, j-1:j+2],
                            slide1[i-1:i+2, j-1:j+2],
                            slide2[i-1:i+2, j-1:j+2]]).astype('float32') / 255.0
        jacobian, tensor, hessian, trace_xy, det_xy, offset, contrast = localize_keypoint(rubik3x3)

        if abs(offset[0]) < 0.5 and abs(offset[1]) < 0.5 and abs(offset[2]) < 0.5:
            break
        j += int(round(offset[0]))
        i += int(round(offset[1]))
        image_index += int(round(offset[2]))

        if i < BORDER_WIDTH or i >= image_shape[0] - BORDER_WIDTH or j < BORDER_WIDTH \
                or j >= image_shape[1] - BORDER_WIDTH or image_index < 1 or image_index > S \
                or attempt_index == convergence - 1 :
            return None

    # reject flats
    if abs(contrast)*S < CONTRAST_THRESHOLD: return None
    # reject edges
    if EIGENVALUE_RATIO*(trace_xy**2) > ((EIGENVALUE_RATIO+1)**2)*det_xy or det_xy<0: return None
    # create keypoint cv2 object
    keypoint = cv2.KeyPoint()
    keypoint.pt = ((j + offset[0]) * (2 ** octave_index), (i + offset[1]) * (2 ** octave_index))
    keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((offset[2] + 0.5) * 255)) * (2 ** 16)
    keypoint.size = SIGMA * (2 ** ((image_index + offset[2]) / float32(S))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
    keypoint.response = abs(contrast)

    return keypoint, image_index

def calculate_magnitude_orientation(image, x,y):
    image = np.array(image)
    dx = image[y, x + 1] - image[y, x - 1]
    dy = image[y - 1, x] - image[y + 1, x]
    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
    return gradient_magnitude, gradient_orientation

def keypoint_orientation(keypoint, nth_octave, image):
    keypoints = []

    scale = SCALE_FACTOR * keypoint.size / float32(2.0 ** (nth_octave + 1))
    radius = int(round(RADIUS_FACTOR * scale))
    weight_factor = -0.5 / (scale ** 2)
    histogram = np.zeros(TOTAL_BIN)

    keypoint_x = keypoint.pt[0]/float32(2.0**nth_octave)
    keypoint_y = keypoint.pt[1]/float32(2.0**nth_octave)

    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            x,y = keypoint_x + j, keypoint_y + i
            if x < 0 or x > image.shape[1] -1 or y < 0 or y > image.shape[0] -1:
                continue
            gradient_mag, gradient_ori = calculate_magnitude_orientation(image,int(x),int(y))
            weight = np.exp(weight_factor * (i ** 2 + j ** 2))
            histogram_index = int(round(gradient_ori * TOTAL_BIN / 360.))
            histogram[histogram_index % TOTAL_BIN] += weight * gradient_mag
    max_orientation = np.max(histogram)

    peak_orientation_index = []
    for i in range(1, TOTAL_BIN-1):
        if histogram[i]> histogram[i-1] and histogram[i] > histogram[i+1]:
            peak_orientation_index.append(i)
    for peak_index in peak_orientation_index:
        peak_value = histogram[peak_index]
        if peak_value > GOOD_PEAK_RATIO*max_orientation:
            left_value = histogram[(peak_index - 1) % TOTAL_BIN]
            right_value = histogram[(peak_index + 1) % TOTAL_BIN]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % TOTAL_BIN
            orientation = 360. - interpolated_peak_index * 360. / TOTAL_BIN
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints.append(new_keypoint)
    return keypoints

def find_all_keypoints(all_image, all_DoG):

    keypoints = []
    for octave_index, nth_DoG in enumerate(all_DoG):
        for image_index, (slide0, slide1, slide2) in enumerate(zip(nth_DoG, nth_DoG[1:], nth_DoG[2:])):
            for i in range(BORDER_WIDTH, slide0.shape[0] - BORDER_WIDTH):
                for j in range(BORDER_WIDTH, slide0.shape[1] - BORDER_WIDTH):
                    if extrema_detection(slide0[i-1:i+2, j-1:j+2], slide1[i-1:i+2, j-1:j+2], slide2[i-1:i+2, j-1:j+2]):
                        finding_result = find_one_keypoint(nth_DoG,i, j, image_index + 1, octave_index)
                        if finding_result is not None:
                            keypoint, localized_image_index = finding_result
                            keypoints_with_orientations = keypoint_orientation(keypoint, octave_index, all_image[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                # convert to keypoint format
                                keypoint_with_orientation.pt = tuple(0.5 * np.array(keypoint_with_orientation.pt))
                                keypoint_with_orientation.size *= 0.5
                                keypoint_with_orientation.octave = (keypoint_with_orientation.octave & ~255) | ((keypoint_with_orientation.octave - 1) & 255)
                                keypoints.append(keypoint_with_orientation)
    return keypoints

##################
# gen descriptor #
##################
def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1.0 / (1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, all_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        image = all_images[octave + 1, layer]
        num_rows,num_cols = image.shape[0],image.shape[1]
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list,col_bin_list,magnitude_list,orientation_bin_list = [],[],[],[]

        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = image[window_row, window_col + 1] - image[window_row, window_col - 1]
                        dy = image[window_row - 1, window_col] - image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

def sift(image):
    image = image.astype('float32')
    gaussian_images = gen_all_image(image)
    dog_images = gen_all_DoG(gaussian_images)
    keypoints = find_all_keypoints(gaussian_images, dog_images)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors

###########################
# plot all images pyramid #
###########################
def plot_all_image(image):
    all_images = gen_all_image(image)
    fx, plots = plt.subplots(TOTAL_OCTAVE, TOTAL_IMAGE_EACH_OCTAVE,gridspec_kw={  'height_ratios': [8,4,2,1]}, figsize=(10,8))
    for nth_octave in range(TOTAL_OCTAVE):
        for image_index in range(TOTAL_IMAGE_EACH_OCTAVE):
            plotted_image = all_images[nth_octave, image_index]
            plots[nth_octave][image_index].axis('off')
            plots[nth_octave][image_index].imshow(plotted_image, cmap='gray')
    plt.subplots_adjust(left=0.067, bottom=0.062, right=0.964, top=0.948, wspace=0.057, hspace=0.13)
    plt.show()

################
# plot nth DoG #
################
def plot_nth_DoG(image, octave_order):
    all_image = gen_all_image(image)
    nth_DoG = []
    nth_octave = all_image[octave_order]
    for image_index in range(1,TOTAL_IMAGE_EACH_OCTAVE):
        last_image = nth_octave[image_index].astype('uint8')
        this_image = nth_octave[image_index-1].astype('uint8')
        nth_DoG.append(this_image-last_image)
    fx, plots = plt.subplots(1,TOTAL_DOG_EACH_OCTAVE)
    for i in range(TOTAL_DOG_EACH_OCTAVE):
        plots[i].axis('off')
        plots[i].imshow(nth_DoG[i], cmap= 'gray')
    plt.show()

############################
# plot image with keypoint #
############################
def plot_keypoint(image, key):
    img_1 = cv2.drawKeypoints(image, key, image, color=(255, 0, 0))
    plt.imshow(img_1)
    plt.show()

####################
# plot match image #
####################
FLANN_INDEX_KDTREE = 0
def plot_match_image(img1, img2, kp1, kp2, des1, des2):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7* n.distance:
            good.append(m)
    print(len(good))
    # Estimate homography between template and scene
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.imshow(newimg)
    plt.show()

