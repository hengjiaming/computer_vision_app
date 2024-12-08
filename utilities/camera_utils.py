from PIL import Image
from PIL.ExifTags import TAGS


def extract_camera_details(image):
    """
    Extract camera details from the image's EXIF data.
    """
    try:
        # Get EXIF data
        exif_data = image._getexif()
        if not exif_data:
            return {"error": "No EXIF data found in the image."}

        # Map EXIF tags
        exif_tags = {TAGS.get(
            tag, tag): value for tag, value in exif_data.items()}

        # Extract relevant details
        details = {
            "Image Size": image.size,
            "Focal Length": exif_tags.get("FocalLength"),
            "Camera Model": exif_tags.get("Model"),
            "Camera Make": exif_tags.get("Make"),
            "Exposure Time": exif_tags.get("ExposureTime"),
            "ISO Speed": exif_tags.get("ISOSpeedRatings"),
        }

        # Format focal length if it's a tuple
        if details["Focal Length"] and isinstance(details["Focal Length"], tuple):
            details["Focal Length"] = float(details["Focal Length"][0]) / float(
                details["Focal Length"][1]
            )

        return details
    except Exception as e:
        return {"error": f"Error extracting camera details: {e}"}
