import urllib.request
import cv2
import imutils
import numpy as np
from skimage.feature import canny
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import probabilistic_hough_line
from imutils.perspective import four_point_transform


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def remove_horizontal_line(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    return result

def remove_dot_noise(image):
    _, blackAndWhite = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 4, cv2.CV_32S)
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 8:  # filter small dotted regions
            img2[labels == i + 1] = 255
    res = cv2.bitwise_not(img2)
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.erode(res,kernel, iterations=1)
    return res

def skew_correction(gray_image):
    orig = gray_image
    thresh = threshold_otsu(gray_image)
    normalize = gray_image > thresh
    blur = gaussian(normalize,3)
    edges = canny(blur)
    hough_lines = probabilistic_hough_line(edges)
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]
    rad_angles = [np.arctan(x) for x in slopes]
    deg_angles = [np.degrees(x) for x in rad_angles]
    histo = np.histogram(deg_angles, bins=100)
    rotation_number = histo[1][np.argmax(histo[0])]
    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_number, 1.0)
    rotated = cv2.warpAffine(orig, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return np.array(rotated)

def align_images(image, template,maxFeatures=7000, keepPercent=0.2,debug=False):
   imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
   orb = cv2.ORB.create(maxFeatures)
   (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
   (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
   method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
   matcher = cv2.DescriptorMatcher.create(method)
   matches = matcher.match(descsA, descsB, None)
   matches = sorted(matches, key=lambda x: x.distance)
   keep = int(len(matches) * keepPercent)
   matches = matches[:keep]
   if debug:
      matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
      matchedVis = imutils.resize(matchedVis, width=1000)
      cv2.imshow("Matched Keypoints", matchedVis)
      cv2.waitKey(0)

   ptsA = np.zeros((len(matches), 2), dtype="float")
   ptsB = np.zeros((len(matches), 2), dtype="float")
   for (i, m) in enumerate(matches):
      ptsA[i] = kpsA[m.queryIdx].pt
      ptsB[i] = kpsB[m.trainIdx].pt

   (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
   (h, w) = template.shape[:2]
   aligned = cv2.warpPerspective(image, H, (w, h))
   return aligned

def resize(img):
   dim = (600, 350)
   resizing = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)
   return resizing

def remove_noise(image):
   return cv2.medianBlur(image, 3)
def border(img):
   return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)

def remove_black(image):
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   lower = np.array([0, 0, 0])
   upper = np.array([100, 175, 110])
   mask = cv2.inRange(hsv, lower, upper)
   invert  = 255-mask
   return invert

def readIMG(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    result = 255 - opening
    result = cv2.GaussianBlur(result, (3, 3), 0)
    return result


def four_point(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([100, 175, 110])
    mask = cv2.inRange(hsv, lower, upper)

    # Morph close to connect individual text into a single contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find rotated bounding box then perspective transform
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (36, 255, 12), 2)
    warped = four_point_transform(255 - mask, box.reshape(4, 2))
    return warped

def url_to_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    return img
