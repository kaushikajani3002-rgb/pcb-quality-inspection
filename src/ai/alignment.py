import cv2
import numpy as np
from PIL import Image
from typing import Any
from src.utils.logger import logger

def align_pcb_image(uploaded_image: Any, reference_image: Any) -> Any:
    """
    Aligns the uploaded target image to a reference template PCB image using feature detection (ORB),
    calculating homography, and applying perspective warping.
    
    :param uploaded_image: Target image to be aligned (PIL Image or numpy array).
    :param reference_image: Reference PCB template image (PIL Image or numpy array).
    :return: Perspectively warped and aligned image.
    """
    if uploaded_image is None:
        raise ValueError("Input target image is invalid or None")
    if reference_image is None:
        raise ValueError("Reference image is invalid or None")
        
    # Convert uploaded image to OpenCV format (numpy BGR array)
    if isinstance(uploaded_image, Image.Image):
        img_target = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    elif isinstance(uploaded_image, np.ndarray):
        img_target = uploaded_image.copy()
    else:
        # Try loading
        img_pil = Image.open(uploaded_image).convert("RGB")
        img_target = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    if img_target is None or img_target.size == 0:
        raise ValueError("Invalid target image dimensions or payload")

    # Convert reference image to OpenCV format
    if isinstance(reference_image, Image.Image):
        img_ref = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    elif isinstance(reference_image, np.ndarray):
        img_ref = reference_image.copy()
    else:
        img_pil_ref = Image.open(reference_image).convert("RGB")
        img_ref = cv2.cvtColor(np.array(img_pil_ref), cv2.COLOR_RGB2BGR)

    if img_ref is None or img_ref.size == 0:
        raise ValueError("Invalid reference image dimensions or payload")

    # Convert to grayscale
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors
    kp_target, des_target = orb.detectAndCompute(gray_target, None)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

    if des_target is None or des_ref is None:
        raise ValueError("Insufficient features detected: missing descriptor patterns in image frame")

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_target, des_ref)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # We need at least 4 matches to calculate Homography, but set a safer minimum of 10
    min_matches = 10
    if len(matches) < min_matches:
        raise ValueError(f"Insufficient matching points found. Need at least {min_matches}, found {len(matches)}")

    # Extract location of good matches
    src_pts = np.float32([kp_target[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if h_matrix is None:
        raise ValueError("Homography calculation failed: reference alignment matrix is singular")

    # Warp perspective to match the dimensions and coordinate space of the reference image
    h, w, c = img_ref.shape
    aligned_img_bgr = cv2.warpPerspective(img_target, h_matrix, (w, h))

    # Convert back to PIL Image if the original was PIL
    if isinstance(uploaded_image, Image.Image):
        aligned_img_rgb = cv2.cvtColor(aligned_img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(aligned_img_rgb)
        
    return aligned_img_bgr
