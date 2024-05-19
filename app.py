from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import threading
import time
import signal
import boto3
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# AWS S3 and SQS settings
AWS_REGION = "eu-north-1"
S3_BUCKET_NAME = "mybucket1000101"
SQS_QUEUE_URL = "https://sqs.eu-north-1.amazonaws.com/211125754953/taskqueue"
s3_client = boto3.client('s3', region_name=AWS_REGION)
sqs_client = boto3.client('sqs', region_name=AWS_REGION)

# Worker statuses
worker_status = {}
tasks_queue = []
tasks_queue_lock = threading.Lock()
stop_event = threading.Event()

@app.route('/')
def index():
    return render_template('index.html', worker_status=worker_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    operation = request.form.get('operation')
    if not files or not operation:
        return 'No file or operation selected', 400
    
    for file in files:
        if file.filename == '':
            return 'No selected file', 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        upload_url = upload_to_s3(file_path, filename)
        if upload_url:
            task_message = f"{S3_BUCKET_NAME},{filename},{operation}"
            try:
                sqs_client.send_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MessageBody=task_message
                )
                print(f"Task added to queue: {filename}, {operation}")
            except Exception as e:
                print(f"Failed to add task to queue: {e}")
        else:
            print(f"Failed to upload file to S3: {filename}")

    return redirect(url_for('index'))

def upload_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, file_name)
        os.remove(file_path)
        print(f'{file_name} uploaded to S3 and local file removed.')
        return f"s3://{S3_BUCKET_NAME}/{file_name}"  # Return the URL of the uploaded blob
    except Exception as e:
        print(f"Failed to upload {file_name} to S3: {e}")
        return None

@app.route('/files')
def list_files():
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
    if 'Contents' in response:
        files = [content['Key'] for content in response['Contents']]
    else:
        files = []
    return render_template('files.html', files=files)

@app.route('/download/<filename>')
def download_file(filename):
    file_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET_NAME, 'Key': filename},
        ExpiresIn=3600
    )
    return redirect(file_url)

@app.route('/status', methods=['POST'])
def update_status():
    data = request.get_json()
    worker_id = data.get('worker_id').split(':')[0]  # Use only the IP address as the key
    status = data.get('status')
    if worker_id not in worker_status:
        worker_status[worker_id] = []
    worker_status[worker_id].append(status)
    return jsonify({'message': 'Status updated'}), 200

@app.route('/status', methods=['GET'])
def status():
    statuses = [{'id': worker, 'statuses': status_list} for worker, status_list in worker_status.items()]
    return jsonify(statuses)

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify([message['Body'] for message in sqs_client.receive_message(QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=10).get('Messages', [])])

@app.route('/clear_tasks', methods=['POST'])
def clear_tasks():
    while True:
        messages = sqs_client.receive_message(QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=10).get('Messages', [])
        if not messages:
            break
        for message in messages:
            sqs_client.delete_message(
                QueueUrl=SQS_QUEUE_URL,
                ReceiptHandle=message['ReceiptHandle']
            )
    return jsonify({'message': 'Tasks cleared'}), 200

@app.route('/clear_all', methods=['POST'])
def clear_all():
    global worker_status
    worker_status = {}
    clear_tasks()
    return jsonify({'message': 'All statuses and results cleared'}), 200

def continuous_task_fetch(stop_event):
    while not stop_event.is_set():
        tasks_fetched = fetch_tasks_from_sqs()
        if tasks_fetched > 0:
            print("New tasks fetched and queued")
        stop_event.wait(10)  # Delay between fetch attempts

def fetch_tasks_from_sqs():
    global stop_event
    if stop_event.is_set():
        print("Stop event set, stopping fetch_tasks_from_sqs")
        return 0
    try:
        messages = sqs_client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5
        ).get('Messages', [])
        fetched_tasks = 0
        with tasks_queue_lock:
            for msg in messages:
                if stop_event.is_set():
                    print("Stop event set, breaking message processing loop")
                    break
                task = msg['Body'].split(',')
                tasks_queue.append(task)
                sqs_client.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=msg['ReceiptHandle']
                )
                fetched_tasks += 1
        if fetched_tasks > 0:
            print(f"Fetched {fetched_tasks} tasks from SQS")
        return fetched_tasks
    except Exception as e:
        print(f"Failed to fetch tasks from SQS: {e}")
        return 0

def update_worker_status(stop_event):
    while not stop_event.is_set():
        stop_event.wait(5)

def signal_handler(sig, frame):
    print("Interrupt received, shutting down...")
    stop_event.set()
    shutdown_server()

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    status_thread = threading.Thread(target=update_worker_status, args=(stop_event,))
    status_thread.start()

    task_fetch_thread = threading.Thread(target=continuous_task_fetch, args=(stop_event,))
    task_fetch_thread.start()

    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down.")
    finally:
        print("Stopping threads...")
        stop_event.set()
        task_fetch_thread.join()
        status_thread.join()
        print("Threads stopped, exiting.")
