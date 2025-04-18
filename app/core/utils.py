import logging
import os
from io import BytesIO

import boto3
import requests
from botocore.exceptions import ClientError
from PIL import Image


def save_image(app_id: str, blob_path: str, save_dir: str = "temp") -> str:
    """Downloads binary image data and saves it as PNG/JPG for 3D conversion.

    Args:
        app_id (str): The application ID for the remote service
        blob_path (str): The blob path from text-to-image response
        save_dir (str): Directory to save temporary images

    Returns:
        str: Path to the saved image file

    Raises:
        Exception: If the image download fails
    """
    os.makedirs(save_dir, exist_ok=True)

    # Construct blob URL
    url = f"https://{app_id}/resource?reid={blob_path}"

    # Download and process image
    response = requests.get(url)
    if response.status_code == 200:
        # Save binary data as image
        image = Image.open(BytesIO(response.content))
        save_path = os.path.join(save_dir, "image.png")
        image.save(save_path, "PNG")
        return save_path
    else:
        raise Exception(f"Failed to download image resource: {response.status_code}")


def save_3d_model(app_id: str, webgl_path: str, save_dir: str = "temp") -> None:
    """Downloads and saves the 3D model file.

    Args:
        app_id (str): The application ID for the remote service
        webgl_path (str): The path to the WebGL file from the response
        save_dir (str): Directory to save temporary files

    Returns:
        None: This function does not return anything

    Raises:
        Exception: If the file download fails
    """
    os.makedirs(save_dir, exist_ok=True)

    # Construct blob URL
    url = f"https://{app_id}/resource?reid={webgl_path}"

    # Download and save file
    response = requests.get(url)
    if response.status_code == 200:
        save_path = os.path.join(save_dir, "model.glb")
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download 3D model resource: {response.status_code}")


def upload_to_s3(file_path: str) -> str | None:
    """Upload a file to an S3 bucket and return the public URL.

    Args:
        ctx (Context): The agent context object
        file_path (str): File to upload
        bucket (str): Bucket to upload to
        object_name (str): S3 object name. If not specified, file_path is used

    Returns:
        str | None: Public URL of the uploaded file if successful, None otherwise
    """
    session = boto3.Session()
    s3_client = session.client(service_name="s3")

    try:
        s3_client.upload_file(
            file_path,
            "forge-projects",
            "projects/model.glb",
            ExtraArgs={"ACL": "public-read"},
        )
        url = "https://forge-projects.s3.amazonaws.com/projects/model.glb"
        logging.info(f"{url} uploaded to S3")
        return url
    except ClientError as e:
        logging.info(f"Error uploading to S3: {e}")
        return None
