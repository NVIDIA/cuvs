from minio import Minio
from minio.deleteobjects import DeleteObject

from config import (OBJECT_STORAGE_PORT, OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY)

def create_object_storage_client(object_storage_host, object_storage_port=OBJECT_STORAGE_PORT):
    endpoint = object_storage_host + ':' + str(object_storage_port)
    client = Minio(endpoint, OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY, secure=False)
    return client

def list_buckets(object_storage_host, object_storage_port=OBJECT_STORAGE_PORT):
    client = create_object_storage_client(object_storage_host, object_storage_port)
    buckets = client.list_buckets()
    for bucket in buckets:
        print(bucket.name)

def list_objects(object_storage_host, bucket_name, object_storage_port=OBJECT_STORAGE_PORT, prefix = None):
    client = create_object_storage_client(object_storage_host, object_storage_port)
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    for obj in objects:
        obj_size_kb = obj.size / 1024
        obj_size_mb = obj_size_kb / 1024
        if obj_size_mb > 1:
            obj_size_str = str(int(round(obj_size_mb, 1))) + 'MiB'
        else:
            obj_size_kb_flat = int(round(obj_size_kb, 1))
            obj_size_str = str(obj_size_kb_flat) + 'KiB'
        obj_siz_pad = obj_size_str.ljust(7)
        print('{}   {}    {}'.format(obj.last_modified.strftime("%Y-%m-%d %H:%M:%S"), obj_siz_pad, obj.object_name))

def accumulate_folder_and_display(folder_total_size, folder_file_count, last_folder):
    folder_size_kb = folder_total_size / 1024
    folder_size_mb = folder_size_kb / 1024
    folder_size_gb = folder_size_mb / 1024
    if folder_size_gb > 1:
        folder_size_str = str(round(folder_size_gb, 1)) + 'GiB'
    elif folder_size_mb > 1:
        folder_size_str = str(round(folder_size_mb, 1)) + 'MiB'
    else:
        folder_size_str = str(round(folder_size_kb, 1)) + 'KiB'

    folder_size_pad = folder_size_str.ljust(12)
    folder_file_count_str = str(folder_file_count) + ' objects'
    folder_file_count_pad = folder_file_count_str.ljust(16)
    print('{}{}   {}'.format(folder_size_pad, folder_file_count_pad, last_folder))

def disk_usage(object_storage_host, bucket_name, object_storage_port=OBJECT_STORAGE_PORT, prefix = None):
    client = create_object_storage_client(object_storage_host, object_storage_port)
    objects_path = bucket_name if prefix is None else bucket_name + '/' +prefix
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    last_folder = None
    total_size = 0
    total_file_count = 0
    folder_total_size = 0
    folder_file_count = 0
    last_obj_size = 0
    for obj in objects:
        if prefix is None: # no prefix - summarize by primary folders
            if "files" not in obj.object_name: # uploaded vectors folder
                index = obj.object_name.find("/")
            else: # milvus created folders
                index = obj.object_name.find("/", obj.object_name.find("/") + 1)
        else:
            index = obj.object_name.rfind("/")
        folder = obj.object_name[:index]
        # print('folder: {} last_folder: {}'.format(folder, last_folder))
        if folder == last_folder or last_folder is None:
            folder_total_size += obj.size
            folder_file_count += 1
        else: # new folder
            if folder_file_count ==1: # passed folder with just one file
                folder_total_size += obj.size
            accumulate_folder_and_display(folder_total_size, folder_file_count, last_folder)
            folder_file_count = 1
            folder_total_size = 0
        last_obj_size = obj.size
        last_folder = folder
        total_size += obj.size
        total_file_count += 1
    # last folder
    if folder_file_count ==1: # passed folder with just one file
        folder_total_size = last_obj_size
    accumulate_folder_and_display(folder_total_size, folder_file_count, last_folder)
    # total
    print('Total {} size:'.format(objects_path))
    accumulate_folder_and_display(total_size, total_file_count, '')

def remove_objects(object_storage_host, bucket_name, object_storage_port=OBJECT_STORAGE_PORT, dry_run=False, prefix = None):
    client = create_object_storage_client(object_storage_host, object_storage_port)
    objects = client.list_objects(bucket_name, prefix, recursive=True)
    objects_copy = list(objects).copy()
    delete_object_list = map(
        lambda x: DeleteObject(x.object_name),
        client.list_objects(bucket_name, prefix, recursive=True),
    )

    if dry_run:
        for obj in objects_copy:
            print('DRYRUN: Removing {}.'.format(obj.object_name))
        return

    errors = client.remove_objects(bucket_name, delete_object_list)
    if len(list(errors)) > 0:
        for error in errors:
            print("error occurred when deleting object", error)
        return
    for obj in objects_copy:
        print('Removed {}.'.format(obj.object_name))