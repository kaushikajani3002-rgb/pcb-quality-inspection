import io
import time
from datetime import datetime
from PIL import Image
from typing import Union, Tuple

class Helper:
    """
    General helper class containing format conversions and execution timers.
    """
    @staticmethod
    def get_current_timestamp() -> str:
        """
        Returns string representation of current date and time.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def get_log_timestamp() -> str:
        """
        Returns log directory friendly string.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Formats float seconds into a human-readable string (e.g., '0.18 sec').
        """
        return f"{seconds:.3f} sec"

    @staticmethod
    def convert_bytes_to_pil(image_bytes: bytes) -> Image.Image:
        """
        Converts uploaded binary payload to PIL Image object.
        """
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def convert_pil_to_bytes(pil_img: Image.Image, format_type: str = "PNG") -> bytes:
        """
        Converts PIL Image object back to binary payload.
        """
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format=format_type)
        return byte_arr.getvalue()
