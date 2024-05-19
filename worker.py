from mpi4py import MPI
import boto3
import cv2
import numpy as np
import logging
import socket

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get the private IP address of the current node
hostname = socket.gethostname()
private_ip = socket.gethostbyname(hostname)

# Set up logging
logging.basicConfig(filename=f'/tmp/node_{rank}.log', level=logging.DEBUG)
logger = logging.getLogger(f'node_{rank}')
logger.info(f"Node {rank} started execution.")

# S3 Configuration
bucket_name = "mybucket1000101"
output_prefix = "/tmp"

# Initialize the S3 client
s3 = boto3.client('s3')

def edge_detection(img):
    return cv2.Canny(img, 100, 200)

def color_manipulation(img):
    return cv2.bitwise_not(img)

def histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def noise_reduction(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def morphological_transform(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def process_image_tasks(task, img):
    try:
        logger.info(f"Starting task {task}")
        if img is None:
            logger.warning(f"Skipping task {task} due to empty image.")
            return None
        if task == 'edge_detection':
            return edge_detection(img)
        elif task == 'color_manipulation':
            return color_manipulation(img)
        elif task == 'histogram_equalization':
            return histogram_equalization(img)
        elif task == 'noise_reduction':
            return noise_reduction(img)
        elif task == 'morphological_transform':
            return morphological_transform(img)
        logger.info(f"Completed task {task}")
    except Exception as e:
        logger.error(f"Error processing task {task}: {e}")
    return None

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    if object_name is None:
        object_name = file_name
    try:
        s3.upload_file(file_name, bucket, object_name)
        logger.info(f"Uploaded {file_name} to {bucket}/{object_name}")
    except Exception as e:
        logger.error(f"Error uploading {file_name}: {str(e)}")
        return False
    return True

while True:
    task = comm.recv(source=0)
    s3_bucket, img_name, operation = task
    local_image_path = f"/tmp/{img_name}"
    
    # Download image from S3
    s3 = boto3.client('s3', region_name='eu-north-1')
    s3.download_file(s3_bucket, img_name, local_image_path)
    img = cv2.imread(local_image_path)

    # Process task
    result = process_image_tasks(operation, img)

    # Save processed result locally and upload back to S3
    output_path = f"/tmp/{operation}_{img_name}"
    cv2.imwrite(output_path, result)
    s3.upload_file(output_path, s3_bucket, f"{operation}_{private_ip}_{img_name}")

    # Send result back to master
    comm.send((private_ip, {operation: result}), dest=0)
    logger.info(f"Worker {rank} processed task {operation} and sent results to master.")