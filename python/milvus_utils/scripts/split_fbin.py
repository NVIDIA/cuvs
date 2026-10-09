import pymilvus
from pymilvus import utility
import time
import os
import mmap
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, Manager

MILVUS_PORT = 19530
MILVUS_IP = os.getenv("MILVUS_IP")
MILVUS_HOST = f"http://{MILVUS_IP}:{MILVUS_PORT}"

def get_milvus_client():
    return pymilvus.MilvusClient(uri=MILVUS_HOST)

def check_size(collection_name):
    coll=pymilvus.Collection(collection_name, using=get_milvus_client()._using)

    return coll.is_empty

def wait_index(collection_name):
    start_time = time.time()
    progress = utility.index_building_progress(collection_name, using=get_milvus_client()._using)
    nrows = progress.get("total_rows", -1)
    with tqdm(total=int(nrows), desc="Indexing vectors") as pbar:
        while True:
            progress = utility.index_building_progress(collection_name, using=get_milvus_client()._using)
            pbar.n = int(progress.get("indexed_rows", 0))
            pbar.refresh()
            time.sleep(1)
            if int(progress.get("indexed_rows", 0)) == int(nrows):
                break
    print(f"Index building finished in {time.time() - start_time} seconds")

def create_fbin_subset(input_file, output_file, num_vectors=1000000):
    """
    Create a new fbin file from an existing one, keeping only the first num_vectors vectors.
    
    Args:
        input_file: Path to the input fbin file
        output_file: Path to the output fbin file
        num_vectors: Number of vectors to keep (default: 1M)
    
    Returns:
        tuple: (num_vectors_kept, dimension) - The actual number of vectors kept and the dimension
    """
    ## Alternative bash: 
    ## N=$((8 + 120000000 * 1024 * 4))
    ## dd if=/home/nvidia/falcon_1024_240M_dataset_backup/base1_falcon_1024_240M.fbin \
    ## of=/raid/data/falcon_1024_120M_dataset/base1_falcon_1024_120M.fbin \
    ## bs=1M iflag=count_bytes count=$N status=progress
    with open(input_file, 'rb') as f_in:
        # Read header: first 8 bytes contain [num_vectors, dimension] as int32
        fbin_attrs = np.fromfile(f_in, count=2, dtype=np.int32)
        original_num_vectors, dimension = fbin_attrs[0], fbin_attrs[1]
        
        # Determine how many vectors to keep
        vectors_to_keep = min(num_vectors, original_num_vectors)
        
        print(f"Original file: {original_num_vectors} vectors, dimension: {dimension}")
        print(f"Creating new file with first {vectors_to_keep} vectors")
        
        # Read the first num_vectors vectors
        vector_size = dimension * 4  # Each float32 is 4 bytes
        data = np.fromfile(f_in, count=vectors_to_keep * dimension, dtype=np.float32)
        
        # Verify we read the correct amount
        actual_vectors_read = len(data) // dimension
        if actual_vectors_read < vectors_to_keep:
            print(f"Warning: Only read {actual_vectors_read} vectors instead of {vectors_to_keep}")
            vectors_to_keep = actual_vectors_read
    
    # Write the new file
    with open(output_file, 'wb') as f_out:
        # Write header: [num_vectors, dimension]
        header = np.array([vectors_to_keep, dimension], dtype=np.int32)
        header.tofile(f_out)
        
        # Write the vectors
        data.tofile(f_out)
    
    print(f"Successfully created {output_file} with {vectors_to_keep} vectors")
    return vectors_to_keep, dimension

def split_fbin_file(input_file, num_vectors=2000000000):
    """
    Create a new fbin file from an existing one, keeping only the first num_vectors vectors.
    
    Args:
        input_file: Path to the input fbin file
        output_file: Path to the output fbin file
        num_vectors: Number of vectors to keep (default: 1M)
    
    Returns:
        tuple: (num_vectors_kept, dimension) - The actual number of vectors kept and the dimension
    """
    with open(input_file, 'rb') as f_in:
        # Read header: first 8 bytes contain [num_vectors, dimension] as int64
        fbin_attrs = np.fromfile(f_in, count=2, dtype=np.int64)
        original_num_vectors, dimension = fbin_attrs[0], fbin_attrs[1]
        print(f"Original file: {original_num_vectors/1000000}M vectors, dimension: {dimension}")
        print(f"Splitting into {num_vectors/1000000}M vectors per file")
        print(f"Total number of files: {(original_num_vectors+num_vectors-1)//num_vectors}")
        n_part = 1
        for i in range(0, original_num_vectors, num_vectors):
            # Determine how many vectors to keep
            vectors_to_keep = min(num_vectors, original_num_vectors - i)
            output_file = f"{input_file.split('.')[0]}_part_{n_part}.fbin"
            print(f"Creating new file with {vectors_to_keep/1000000}M vectors")
            with open(output_file, 'wb') as f_out:
                header = np.array([vectors_to_keep, dimension], dtype=np.int32)
                header.tofile(f_out)
                batch_size = 1000000
                with tqdm(total=int(vectors_to_keep), desc=f"Splitting vectors part {n_part}") as pbar:
                    for batch_index in range(0, vectors_to_keep, batch_size):
                        data = np.fromfile(f_in, count=batch_size * dimension, dtype=np.float32)
                        data.tofile(f_out)
                        pbar.n = int(batch_index)
                        pbar.refresh()
            print(f"Successfully created {output_file} with {vectors_to_keep/1000000}M vectors")
            n_part += 1

def _process_part(args):
    """
    Worker function to process a single part of the split operation.
    
    Args:
        args: tuple of (input_file, n_part, start_index, vectors_to_keep, dimension, num_vectors, progress_counter)
    
    Returns:
        tuple: (n_part, output_file, vectors_to_keep)
    """
    input_file, n_part, start_index, vectors_to_keep, dimension, num_vectors, progress_counter = args
    
    # Calculate file positions
    header_size = 2 * 8  # 2 int64 values = 16 bytes
    vector_size_bytes = dimension * 4  # Each float32 is 4 bytes
    start_byte = header_size + start_index * vector_size_bytes
    
    output_file = f"{input_file.split('.')[0]}_part_{n_part}.fbin"
    
    batch_size = 1000000
    actual_vectors = 0
    
    # Open both files - write header first, then stream data
    with open(output_file, 'wb') as f_out:
        # Write placeholder header (will update with actual count at the end)
        header = np.array([vectors_to_keep, dimension], dtype=np.int32)
        header.tofile(f_out)
        
        # Open input file and seek to the correct position
        with open(input_file, 'rb') as f_in:
            f_in.seek(start_byte)
            # Read and write batches simultaneously
            remaining = vectors_to_keep
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                data = np.fromfile(f_in, count=current_batch * dimension, dtype=np.float32)
                
                if len(data) == 0:
                    break
                
                # Write batch immediately
                data.tofile(f_out)
                batch_vectors = len(data) // dimension
                actual_vectors += batch_vectors
                remaining -= current_batch
                
                # Update shared progress counter
                progress_counter.value += batch_vectors
    
    return (n_part, output_file, actual_vectors)

def split_fbin_file_mp(input_file, num_vectors=1000000000, num_processes=10):
    """
    Create multiple fbin files from an existing one using multiprocessing.
    Each output file contains num_vectors vectors (except possibly the last one).
    
    Args:
        input_file: Path to the input fbin file
        num_vectors: Number of vectors per output file (default: 1B)
        num_processes: Number of parallel processes to use (default: cpu_count())
    
    Returns:
        list: List of tuples (part_number, output_file, vectors_kept)
    """
    # Read header to get file info
    with open(input_file, 'rb') as f_in:
        fbin_attrs = np.fromfile(f_in, count=2, dtype=np.int64)
        original_num_vectors, dimension = fbin_attrs[0], fbin_attrs[1]
    
    print(f"Original file: {original_num_vectors/1000000}M vectors, dimension: {dimension}")
    print(f"Splitting into {num_vectors/1000000}M vectors per file")
    
    # Calculate number of parts
    num_parts = (original_num_vectors + num_vectors - 1) // num_vectors
    print(f"Total number of files: {num_parts}")
    
    # Create shared counter for progress tracking
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    # Prepare arguments for each part
    tasks = []
    n_part = 1
    for i in range(0, original_num_vectors, num_vectors):
        vectors_to_keep = min(num_vectors, original_num_vectors - i)
        tasks.append((input_file, n_part, i, vectors_to_keep, dimension, num_vectors, progress_counter))
        n_part += 1
    
    # Process parts in parallel
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Processing {num_parts} parts using {num_processes} processes...")
    
    
    with Pool(processes=num_processes) as pool:
        # Start async map operation
        async_result = pool.map_async(_process_part, tasks)
        
        # Monitor progress with tqdm while waiting for results
        with tqdm(total=int(original_num_vectors), desc="Splitting file", unit="vectors", unit_scale=True) as pbar:
            last_count = 0
            while not async_result.ready():
                # Poll the shared counter and update progress bar
                current_count = progress_counter.value
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)  # Small sleep to avoid busy waiting
            
            # Get results (this will return immediately since ready() is True)
            results = async_result.get()
            
            # Final update to ensure we're at 100%
            final_count = progress_counter.value
            if final_count > last_count:
                pbar.update(final_count - last_count)
            pbar.n = int(original_num_vectors)
            pbar.refresh()
    
    # Sort results by part number
    results.sort(key=lambda x: x[0])
    
    # Print results
    for n_part, output_file, vectors_kept in results:
        print(f"Successfully created {output_file} with {vectors_kept/1000000}M vectors")
    
    return results

if __name__ == "__main__":
    # print(check_size(os.getenv("COLLECTION_NAME")))
    # wait_index(os.getenv("COLLECTION_NAME"))
    #split_fbin_file("/raid/data/falcon_1024_5b_mock_dataset/falcon_extract_10M_seed42_representative_normalized_sample9999360_n30000_lowrankpca32_300/mock_vectors.fbin64")
    #split_fbin_file_mp("/raid/data/falcon_1024_5b_mock_dataset/falcon_extract_10M_seed42_representative_normalized_sample9999360_n30000_lowrankpca32_300/mock_vectors.fbin64",
    #                   num_vectors=100000000, num_processes=50)
    
    create_fbin_subset("/home/nvidia/falcon_1024_240M_dataset_backup/base1_falcon_1024_240M.fbin",
                       "/raid/data/falcon_1024_120M_dataset/base1_falcon_1024_120M.fbin", num_vectors=120000000)