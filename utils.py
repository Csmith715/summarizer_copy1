import boto3
import os
import pathlib

cta_root_path = os.getenv('CTA_ROOT_PATH', '/tmp')
qg_root_path = os.getenv('QG_ROOT_PATH', '/tmp')
cta_path = os.getenv('CTA_PATH', 'CTA_Bullets')
qg_path = os.getenv('QG_PATH', 'question-generation')

s3 = boto3.client('s3')

def download_s3_folder(bucket_name, s3_folder):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
    """
    local_dir = f'{qg_root_path}/{qg_path}'
    if pathlib.Path(local_dir).is_dir():
        return
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

def download_file(path, file, bucket_name):
    tdir = pathlib.Path(path)
    target_dir = f'{cta_root_path}/{path}'
    if tdir.joinpath(file).is_file():
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    return s3.download_file(bucket_name, f'{path}/{file}', f'{target_dir}/{file}')
