import argparse
import sys

import h5py
import numpy
import pandas as pd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='groundtruth file (fbin or parquet)')
    parser.add_argument('--base', type=str, help='fbin base file')
    parser.add_argument('--query', type=str, help='fbin query file')
    parser.add_argument('--hdf5', type=str, help='output hdf5 file')
    args = parser.parse_args()
    gnd_file_name = args.gt
    fbin_base_file_name = args.base
    query_base_file_name = args.query
    hdf5_file_name = args.hdf5

    if gnd_file_name.endswith('parquet'):
        print('gt parquet file')
        gt_ids = []
        gt_distances = []
        df = pd.read_parquet(gnd_file_name)
        sz_query = len(df)
        print('sz_query:' + str(sz_query))
        neighbors_substring = 'neighbors_id'
        distances_substring = 'distances'
        matching_columns = [col for col in df.columns if (neighbors_substring in col or distances_substring in col) ]
        df_matching = df[matching_columns]
        gt_lists = df_matching.values.tolist()
        for a_list in gt_lists:
            gt_ids.append(a_list[0])
            gt_distances.append(a_list[1])
        max_k = len(gt_ids[0])
        print('max_k:' + str(max_k))
        arr_neighbors = numpy.reshape(gt_ids, [sz_query, max_k])
        arr_distances = numpy.reshape(gt_distances, [sz_query, max_k])
    elif gnd_file_name.endswith(('fbin', 'ibin')):
        print('gt fbin file')
        gnd_file = open(gnd_file_name, 'rb')
        attrs = numpy.fromfile(gnd_file, count=2, dtype=numpy.int32)
        print(attrs)
        sz_query, max_k = attrs[0], attrs[1]
        data = numpy.fromfile(gnd_file, count=sz_query * max_k, dtype=numpy.int32)
        arr_neighbors = numpy.reshape(data, [sz_query, max_k])
        data = numpy.fromfile(gnd_file, count=sz_query * max_k, dtype=numpy.float32)
        arr_distances = numpy.reshape(data, [sz_query, max_k]) # this is actually the square distances -  (arr_distances**0.5) should apply on write
    else:
        print(f"Error: Only fbin or parquet groundtruth file format is supported")
        sys.exit(0)

    if query_base_file_name.endswith('fbin'):
        query_file = open(query_base_file_name, 'rb')
        attrs = numpy.fromfile(query_file, count=2, dtype=numpy.int32)
        print(attrs)
        num_query_vectors, dimension = attrs[0], attrs[1]
        query_data = numpy.fromfile(query_file, count=num_query_vectors * dimension, dtype=numpy.float32)
        arr_queries = numpy.reshape(query_data, [num_query_vectors, dimension])
    else:
        print(f"Error: Only fbin query file format is supported")
        sys.exit(0)

    if fbin_base_file_name.endswith('fbin'):
        base_file = open(fbin_base_file_name, 'rb')
        attrs = numpy.fromfile(base_file, count=2, dtype=numpy.int32)
        print(attrs)
        num_base_vectors, dimension = attrs[0], attrs[1]
    else:
        print(f"Error: Only fbin base file format is supported")
        sys.exit(0)

    fout = h5py.File(hdf5_file_name, "w")
    fout.create_dataset('distances', data=arr_distances**0.5)  # squared of the real distances
    fout.create_dataset('neighbors', data=arr_neighbors)
    fout.create_dataset('test', data=numpy.float32(arr_queries))
    print('create_dataset num_base_vectors: ' + str(num_base_vectors))
    if num_base_vectors >= 1e6:
        sz_b = int(1e6)
    else:
        sz_b = num_base_vectors
    fout.create_dataset("train", (num_base_vectors, dimension), chunks=(sz_b, dimension))
    num_full_chunks = num_base_vectors // sz_b
    remaining = num_base_vectors % sz_b

    for i in range(num_full_chunks):
        data = numpy.fromfile(base_file, count=sz_b * dimension, dtype=numpy.float32)
        arr_base = numpy.reshape(data, (sz_b, dimension))
        fout["train"][i * sz_b:(i + 1) * sz_b, :] = arr_base
        print(f'finished chunk {i}')

    if remaining > 0:
        data = numpy.fromfile(base_file, count=remaining * dimension, dtype=numpy.float32)
        arr_base = numpy.reshape(data, (remaining, dimension))
        fout["train"][num_full_chunks * sz_b:num_base_vectors, :] = arr_base
        print('finished remaining chunk')


    fout.attrs.create('distance', 'euclidean')
    fout.close()
    print('hdf5 file created: ' + hdf5_file_name)
    print('END')


if __name__ == "__main__":
    main()
