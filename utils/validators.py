import io
from pathlib import Path
from typing import Any, Dict, Union, Tuple
from utils.logger import logger

class ImageValidator:
    """
    Validates image payload parameters, headers, and extensions to protect
    against malicious file injections or corrupt payloads.
    """
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    MAX_FILE_SIZE_MB = 15.0

    # Hex signatures for images
    MAGIC_SIGNATURES = {
        b"\x89PNG\r\n\x1a\n": "PNG",
        b"\xff\xd8\xff": "JPEG"
    }

    @classmethod
    def validate_file_metadata(cls, filename: str, file_size_bytes: int) -> Tuple[bool, str]:
        """
        Validates basic file metadata including size and suffix extensions.
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in cls.ALLOWED_EXTENSIONS:
            err_msg = f"Unsupported file extension '{suffix}'. Allowed: {cls.ALLOWED_EXTENSIONS}"
            logger.warning(err_msg)
            return False, err_msg
        
        max_bytes = cls.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size_bytes <= 0:
            err_msg = "Payload is empty (0 bytes)."
            logger.warning(err_msg)
            return False, err_msg
        
        if file_size_bytes > max_bytes:
            err_msg = f"File size exceeds the limit of {cls.MAX_FILE_SIZE_MB}MB."
            logger.warning(err_msg)
            return False, err_msg

        return True, "Metadata valid"

    @classmethod
    def validate_image_bytes(cls, image_bytes: bytes) -> Tuple[bool, str]:
        """
        Inspects raw bytes to ensure they match valid image headers (magic numbers).
        """
        if not image_bytes or len(image_bytes) < 8:
            return False, "Insufficient payload size to verify file signature."

        for signature, format_name in cls.MAGIC_SIGNATURES.items():
            sig_len = len(signature)
            if image_bytes[:sig_len] == signature:
                return True, f"Valid {format_name} signature matched."

        err_msg = "Image content verification failed. Header does not match PNG or JPEG magic numbers."
        logger.warning(err_msg)
        return False, err_msg


class ConfigurationValidator:
    """
    Validates config parameters against boundaries to ensure numerical safety.
    """
    @staticmethod
    def validate_inspection_parameters(confidence: float, iou: float, position_tolerance: float) -> Tuple[bool, str]:
        """
        Validates confidence slider, IoU slider, and pixel alignment tolerances.
        """
        try:
            if not (0.0 <= confidence <= 1.0):
                return False, f"Confidence threshold {confidence} is out of bounds [0.0, 1.0]."
            
            if not (0.0 <= iou <= 1.0):
                return False, f"IoU threshold {iou} is out of bounds [0.0, 1.0]."

            if position_tolerance <= 0:
                return False, f"Position tolerance {position_tolerance} must be a positive value."

            return True, "Configuration values within valid bounds."
        except Exception as e:
            logger.error(f"Error executing ConfigurationValidator: {e}")
            return False, f"Internal validation error: {e}"
