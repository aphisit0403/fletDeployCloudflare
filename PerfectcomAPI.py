from skimage import io
from pytesseract import Output
import cv2
from PIL import Image
import pytesseract
import numpy as np
import binascii
import string
import re
from collections import namedtuple
from pathlib import Path

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
# define the locations of each area of the document we wish to OCR
OCR_LOCATIONS = [
    OCRLocation("card_number", (249, 30, 517, 62), ["เลขประจําตัวประชาชน", "Identification Number", "oe oon"]),
    OCRLocation("FullnameTH", (165, 61, 505, 109), ["WIE", "ชื่อตัวและชื่อสกุล"]),
    OCRLocation("nameEN", (232, 103, 400, 131), ["Name", "name", "oe ean"]),
    OCRLocation("LastnameEN", (257, 126, 449, 157), ["Last name", "Lastname"]),
    OCRLocation("dobTH", (255, 152, 431, 185), ["4.0", "AA"]),
    OCRLocation("dobEN", (293, 177, 443, 210), ["Date", "Birht"]),
    OCRLocation("Religion", (245, 202, 320, 231), ["ศาสนา"]),
    OCRLocation("Address", (54, 224, 345, 282), ["ที่อu"]),
    OCRLocation("Issue_dateTH", (53, 278, 160, 299), [":", "วะนออกบัตร", "วันออกบัตร"]),
    OCRLocation("Issue_dateEN", (51, 310, 163, 331), [":", "วะนออกบัตร", "วันออกบัตร", "วันขอกบัตร"]),
    OCRLocation("expiry_dateTH", (333, 275, 429, 299), [":", "วันหมดอายุ"]),
    OCRLocation("expiry_dateEN", (323, 311, 442, 329), [":", "วันหมดอายุ", "บัตรหมดยอาย"]),
    OCRLocation("office", (153, 301, 321, 349), ["พนักงงาน", "พนักงงาน", "สคง", "เจ้าพนักงานออกบัตร"])
]


class OcrReader:
    def __init__(self,template_threshold: float = 0.7,
                 tessdata_dir_config: str = r'--tessdata-dir "Tesseract-OCR/tessdata"'):
        self.tessdata_dir_config = tessdata_dir_config
        self.template_threshold = template_threshold
        self.root_path = Path(__file__).parent.parent
        self.image = None
        self.shift_rate = 25000
        self.good = []
        self.parsingResults = []
        #pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
    def flan(self, img_soruce):
        template = cv2.imread("template_img/personal-card-template.jpg")
        sift = cv2.SIFT.create(25000)
        kp1, des1 = sift.detectAndCompute(img_soruce, None)
        kp2, des2 = sift.detectAndCompute(template, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > 30:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape[:2]
            image_scan = cv2.warpPerspective(img_soruce, M, (w, h))
            return image_scan

    def __readImage(self, image=None):
        try:
            try:
                # handler if image params is base64 encode.
                img = io.imread(image)
                # img = cv2.imdecode(np.fromstring(base64.b64decode(image, validate=True), np.uint8), cv2.IMREAD_COLOR)
                # img = url_to_image(image)
            except binascii.Error:
                # handler if image params is string path.
                img = cv2.imread(image)
            dim = (600, 350)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            return img
        except cv2.error as e:
            raise ValueError(f"Can't read image from source. cause {e.msg}")

    def __extractItems(self, image_scan):
        self.parsingResults = []
        for loc in OCR_LOCATIONS:
            (x, y, w, h) = loc.bbox
            roi = image_scan[y:h, x:w]
            imgCrop = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            try:
                if loc.id in 'card_number':
                    card_number = pytesseract.image_to_string(imgCrop, lang="tha+eng", config="--oem 1 -c tessedit_char_whitelist=0123456789",
                                                              output_type=Output.STRING)
                    card_number = re.sub(r'[^\w\s\n]', ' ', card_number)
                    idc_regex = re.compile(r"\d")
                    idc_matches = idc_regex.findall(card_number)
                    self.parsingResults.append("".join(idc_matches))

                elif loc.id in 'FullnameTH':
                    imgCrop = cv2.resize(imgCrop, (0, 0), fx=1.2, fy=1.2)
                    imgCrop = cv2.adaptiveThreshold(imgCrop[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY,
                                                    11,
                                                    8) + cv2.adaptiveThreshold(imgCrop[:, :, 1], 255,
                                                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                               cv2.THRESH_BINARY, 11,
                                                                               8) + cv2.adaptiveThreshold(
                        imgCrop[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
                    NameTH = pytesseract.image_to_string(imgCrop, lang="tha", config="--dpi 300 --psm 7 --oem 1",
                                                         output_type=Output.STRING)
                    NameTH = " ".join(NameTH.split())
                    NameTH = re.sub(r',|เ', '.', NameTH)
                    pre_regex = re.compile(r"(นาย|นาง|นางสาว|น.ส.)")
                    pre_matches = pre_regex.findall(NameTH)
                    first_name = NameTH.split(" ")[1]
                    lastTH_regex = re.compile(r"([\u0E00-\u0E7F]+$)")
                    last_name = lastTH_regex.findall(NameTH)
                    if pre_matches:
                        self.parsingResults.append(" ".join(pre_matches))
                    if first_name:
                        self.parsingResults.append(first_name)
                    if last_name:
                        self.parsingResults.append(" ".join(last_name))
                elif loc.id in 'nameEN':
                    imgCrop = cv2.adaptiveThreshold(imgCrop[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY,
                                                    11,
                                                    8) + cv2.adaptiveThreshold(imgCrop[:, :, 1], 255,
                                                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                               cv2.THRESH_BINARY, 11,
                                                                               8) + cv2.adaptiveThreshold(
                        imgCrop[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
                    nameEN = pytesseract.image_to_string(imgCrop,
                                                         config="-l eng --dpi 300 --psm 13 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.",
                                                         output_type=Output.STRING)
                    nameEN = re.sub(r'[^\w\s\n|\n]', ' ', nameEN)
                    preEN_regex = re.compile(r"(Mr|Miss|Mrs|Ms)")
                    preEN_matches = preEN_regex.findall(nameEN)
                    nameen_regex = re.compile(r"([a-zA-Z']+$)")
                    first_nameEN = nameen_regex.findall(nameEN)
                    if preEN_matches:
                        self.parsingResults.append(" ".join(preEN_matches))
                        if first_nameEN:
                            first_nameEN = " ".join(first_nameEN)
                            first_nameEN = re.sub(r'(Mr|Miss|Mrs|Ms)', '', first_nameEN)
                            self.parsingResults.append(first_nameEN)
                elif loc.id in 'LastnameEN':
                    LastnameEN = pytesseract.image_to_string(imgCrop,
                                                             config="-l eng --dpi 300 --psm 13 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.",
                                                             output_type=Output.STRING)
                    last_nameEN = LastnameEN.strip()
                    self.parsingResults.append(last_nameEN)
                elif loc.id in 'dobTH':
                    dobTH = pytesseract.image_to_string(imgCrop, config="-l tha --psm 7 --oem 1",
                                                        output_type=Output.STRING)
                    dobTH = " ".join(dobTH.split())
                    dob = re.sub(r',ุ', '.', dobTH)
                    self.parsingResults.append(dob.strip())
                elif loc.id in 'dobEN':
                    dobEN = pytesseract.image_to_string(imgCrop, config="-l eng --psm 3 --oem 1",
                                                        output_type=Output.STRING)
                    dobEN = " ".join(dobEN.split())
                    doben = re.sub(r',', '.', dobEN)
                    self.parsingResults.append(doben.strip())
                elif loc.id in 'Religion':
                    Religion = pytesseract.image_to_string(imgCrop,
                                                           config="-l tha --dpi 300 --psm 7 --oem 1",
                                                           output_type=Output.STRING)
                    Religion = re.sub(r'[a-zA-Z0-9\._-]', '', Religion)
                    religion = Religion.strip()
                    self.parsingResults.append(religion)
                elif loc.id in 'Address':
                    imgCrop = cv2.resize(imgCrop, (0, 0), fx=1.25, fy=1.25)
                    imgCrop = cv2.adaptiveThreshold(imgCrop[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY,
                                                    11,
                                                    8) + cv2.adaptiveThreshold(imgCrop[:, :, 1], 255,
                                                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                               cv2.THRESH_BINARY, 11,
                                                                               8) + cv2.adaptiveThreshold(
                        imgCrop[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
                    add = pytesseract.image_to_string(imgCrop,lang="tha", config="--dpi 300, --psm 3",
                                                      output_type=Output.STRING)
                    address = " ".join(add.split())
                    punc = '''!()-[]{};:'"\|,<>?@#$%^&*_~'''
                    for ele in address:
                        if ele in punc:
                            address = address.replace(ele, "")
                    addr_clean = re.sub(r'แมุ่ที|หมุ่ที', 'หมู่ที่', address)
                    addregex = re.compile(r'\b[0-9].+')
                    addr_result = addregex.findall(addr_clean)
                    self.parsingResults.append(" ".join(addr_result).strip())
                elif loc.id in 'Issue_dateTH':
                    img_issudate = cv2.resize(imgCrop, (0, 0), fx=1.2, fy=1.2)
                    Issue_date = pytesseract.image_to_string(img_issudate, lang="tha", config="--dpi 300 --psm 6",
                                                             output_type=Output.STRING)
                    issuedate = " ".join(Issue_date.split())
                    punc = '''!()-[]{};:'"\|,<>/?@#$%^&*_~'''
                    for ele in issuedate:
                        if ele in punc:
                            issuedate = issuedate.replace(ele, "")
                    issuedate = re.sub(r',ุ', '.', issuedate)
                    self.parsingResults.append(issuedate.strip())
                elif loc.id in 'Issue_dateEN':
                    imgCrop = cv2.resize(imgCrop, (0, 0), fx=1.2, fy=1.2)
                    imgCrop = cv2.adaptiveThreshold(imgCrop[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY,
                                                   11,
                                                   8) + cv2.adaptiveThreshold(imgCrop[:, :, 1], 255,
                                                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                              cv2.THRESH_BINARY, 11,
                                                                              8) + cv2.adaptiveThreshold(
                        imgCrop[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
                    IssueEN = pytesseract.image_to_string(imgCrop, lang="eng", config="--dpi 300 --psm 6",
                                                          output_type=Output.STRING)
                    issuedateEN = " ".join(IssueEN.split())
                    punc = '''!()-[]{};:'"\|—.,+<>/?@#$%^&*_~'''
                    for ele in issuedateEN:
                        if ele in punc:
                            issuedateEN = issuedateEN.replace(ele, "")
                    self.parsingResults.append(issuedateEN.strip())
                elif loc.id in 'expiry_dateTH':
                    expiry_date = pytesseract.image_to_string(imgCrop, lang="tha", config="--dpi 400 --psm 6",
                                                              output_type=Output.STRING)
                    expirydate = " ".join(expiry_date.split())
                    expiryTH = re.sub(r'ว', '7', expirydate)
                    punc = '''!()-[]{};:'"\|—,+<>/?@#$%^&*_~'''
                    for ele in expiryTH:
                        if ele in punc:
                            expiryTH = expiryTH.replace(ele, "")
                    self.parsingResults.append(expiryTH.strip())
                elif loc.id in 'expiry_dateEN':
                    expiry_dateEN = pytesseract.image_to_string(imgCrop, lang="eng", config="--dpi 400 --psm 6",
                                                                output_type=Output.STRING)
                    expirydateEN = " ".join(expiry_dateEN.split())
                    expiryEN = expirydateEN.translate(str.maketrans('', '', string.punctuation))
                    self.parsingResults.append(expiryEN.strip())
                else:
                    imgCrop = cv2.resize(imgCrop, (0, 0), fx=1.2, fy=1.2)
                    office = pytesseract.image_to_string(imgCrop, lang="tha+dilleniaupc", config="--dpi 300 --psm 3 --oem 3",
                                                         output_type=Output.STRING)
                    id1 = office.index("(")
                    id2 = office.index(")")
                    res = ''
                    for i in range(id1 + len("("), id2):
                        res = res + office[i]
                    office_name = res
                    self.parsingResults.append(office_name.strip())
            except:
                pass
        return self.parsingResults

    def extract_data(self, image):
        img_soruce = self.__readImage(image)
        image_scan = self.flan(img_soruce)
        result = self.__extractItems(image_scan)
        return result
