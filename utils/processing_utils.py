import cv2
import numpy as np
import os


# =========================
# 1. LIGHTING FIX
# =========================
def fix_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# =========================
# 2. EDGE DETECTION (FIXED)
# =========================
def detect_edges(image):
    image = fix_lighting(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # FIX: stronger edges for printed papers
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return edges


# =========================
# 3. FIND DOCUMENT CONTOUR (FIXED)
# =========================
def find_contours(edges, image):

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    best_contour = None
    best_score = 0

    for c in contours:

        area = cv2.contourArea(c)

        if area < 0.02 * img_area:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        # FIX: relaxed filter (important for printed docs)
        if aspect_ratio < 0.5 or aspect_ratio > 6:
            continue

        area_ratio = area / img_area

        if area_ratio > 0.98:
            continue

        if area_ratio < 0.02:
            continue

        score = area * (2 / aspect_ratio)

        if score > best_score:
            best_score = score
            best_contour = box

    return best_contour


# =========================
# 4. ORDER POINTS
# =========================
def order_points(pts):

    pts = np.array(pts).reshape(-1, 2)

    if len(pts) != 4:
        return None

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# =========================
# 5. PERSPECTIVE TRANSFORM
# =========================
def perspective_transform(image, pts):

    rect = order_points(pts)

    if rect is None:
        return image

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # crop borders
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(255 - thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        warped = warped[y:y+h, x:x+w]

    return warped




def clean_scan(image):

    # 🔥 STEP 1: grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔥 STEP 2: enhance contrast (VERY IMPORTANT)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 🔥 STEP 3: light denoise (DO NOT blur text)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 🔥 STEP 4: adaptive threshold (soft)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,   # bigger block size = safer text
        5
    )

    return binary

# =========================
# 7. ENHANCE FINAL OUTPUT (FIXED SINGLE VERSION)
# =========================
def enhance_scan(image):

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # PRINTED TEXT MODE
    if lap_var > 120:

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        sharp = cv2.filter2D(image, -1, kernel)

        scan = cv2.adaptiveThreshold(
            sharp,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21,
            7
        )

    # HANDWRITTEN MODE
    else:

        clahe = cv2.createCLAHE(2.5, (8, 8))
        image= clahe.apply(image)

        scan = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5
        )

    return scan


# =========================
# 8. MAIN PIPELINE
# =========================
if __name__ == "__main__":

    input_path = r"E:\desktop data\mentorship program devsil\document_scanner\dataset\input_images"

    for image_file in os.listdir(input_path):

        image = cv2.imread(os.path.join(input_path, image_file))

        if image is None:
            continue

        edges = detect_edges(image)
        contour = find_contours(edges, image)

        debug = image.copy()

        if contour is None:
            print(f"No document found: {image_file}")
            continue

        cv2.drawContours(debug, [contour], -1, (0, 255, 0), 3)

        warped = perspective_transform(image, contour)
        cleaned = clean_scan(warped)
        final_scan = enhance_scan(warped)

        cv2.imshow("Original", image)
        cv2.imshow("Edges", edges)
        cv2.imshow("Contour Debug", debug)
        cv2.imshow("Scanned Output", final_scan)

        print(f"Processed: {image_file}")

        cv2.waitKey(0)

    cv2.destroyAllWindows()