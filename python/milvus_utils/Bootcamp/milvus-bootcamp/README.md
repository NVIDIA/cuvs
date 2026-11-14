# Scripts about Milvus

## Prerequisites
* Python >= 3.10
* Running Milvus 2.6.*  
* Proxy is configured if required
* For the upload command - platform memory >= 64GB

### Installation notes
Bootcamp can be installed either on the same machine where Milvus 
runs or on another machine in the same network. If Milvus runs on 
another machine, ensure that the Milvus port is open on the server 
machine and there are no firewall or proxy rules dropping the 
connection between the machines.

## Getting Started with the Milvus Bootcamp Project
### Data files for testing
The data files for bootcamp testing can be downloaded from the [TEXMEX](http://corpus-texmex.irisa.fr/) site
### Installation
Download the Milvus Bootcamp from [From Jenkins Build](https://10.93.66.62:31648/job/Milvus%20Bootcamp/) and unzip it.
```shell
unzip milvus-bootcamp-1.1.<release>.zip	
cd milvus-bootcamp
./install.sh
source venv/bin/activate
```
### Uninstall
The most secure way to uninstall bootcamp is to delete the directory containing all the bootcamp files.
### Upgrade
Remove the directory containing the current bootcamp files. Download and install the new version following 
the instructions above. 
## Running Milvus Bootcamp with Customized Options
`python3 main.py [<OPTIONS>] <COMMAND>`

### For advanced users
To see examples of how to run Milvus Bootcamp with customized options, refer to the scripts `index_cycle.sh` and `options-test.sh`.
Edit the variable values in the `index_cycle.sh` script to suite your environment. 
## Scripts

**Commands**

| Procedure         | Short | Description                                                                          |
|-------------------|-------|--------------------------------------------------------------------------------------|
| create            | c     | Perform the operation of creating a collection.                                      |
| insert            | a     | Perform the operation of writing data from fvecs/fbin file.                          |
| upload            | u     | Perform the operation of uploading data from fvecs/fbin file.                        |
| complete_upload   | cu    | Perform the operation of completing uploading data from bulk_writer folder.          |
| import            | im    | Imports the prepared data files from the object storage bucket to a Milvus instance. |
| import_progress   | imp   | Gets the progress of the specified bulk-import job.                                  |
| list_imports      | lim   | Lists all import jobs for a given collection.                                        |
| create_index      | I     | Perform indexing operations.                                                         |
| drop_index        | O     | Drop index.                                                                          |
| performance       | P     | Execute performance test.                                                            |
| desc              | d     | Describe the collection metadata.                                                    |
| index_info        | i     | View the index information of for the provided collection.                           |
| index_progress    | p     | View the index progress.                                                             |
| has               | h     | Determine whether a collection exists.                                               |
| rows              | r     | View the number of vectors in a collection.                                          |
| search            | S     | Search the k nearest neighbors for each query vector                                 |
| drop              | D     | Delete the specified collection.                                                     |
| load              | L     | Load the specified collection data to memory                                         |
| compact           | C     | Compact and merge small segments in the current collection.                          |
| list              | l     | List all collections.                                                                |
| release           | R     | Release the specified collection data from memory                                    |
| mc_list_buckets   | lib   | List information of all accessible buckets                                           |
| mc_list_objects   | lio   | Lists objects information of a bucket                                                |
| mc_disk_usage     | du    | Summarizes the disk usage of given bucket and its folders                            |
| mc_remove_objects | ro    | Remove (delete) all objects (files) from a bucket                                    |
| rename            | n     | Rename a collection                                                                  |
| segments          | g     | Show segments info of a collection                                                   |

<i>If parameters are not passed in the CLI, default values from the ```config.py``` file 
will be used. For the default value of each option, see the "Config" and "Default" 
columns of the tables below.<i>

**Parameters for all commands**

| Option          | Default      | Description                                                                |
|-----------------|--------------|----------------------------------------------------------------------------|
| --collection    |              | Specify the name of the collection to be operated on                       |
| --host          | 127.0.0.1    | Specify the IP where the Milvus server / Object Storage service is located |
| --port          | 19530 / 9000 | Specify the port provided by Milvus server / Object Storage service        |
 <i>* The <b>list</b> command doesn't require the <b>--collection</b> parameter</i>

**Parameters for collection creation**

| Option                | Default | Description                                                                                                |
|-----------------------|---------|------------------------------------------------------------------------------------------------------------|
| --dim                 |         | Specify the vector dimension when creating a collection                                                    |
| --shards_num          | 1       | Specify number of shards to split the collection                                                           |
| --additional_fields   |         | Specify list of additional fields                                                                          |
| --additional_types    |         | Specify list of fields types correlated to the List of additional fields                                   |
| --not_clustering_key  | False   | Whether the collection vectors field is not clustering key in Milvus open source contribution collections. |

**Parameters for load collection**

| Option    | Default | Description                                           |
|-----------|---------|-------------------------------------------------------|
| --exclude |         | Specify the fields to exclude from collection loading |


**Parameters for indexing**

| Option                            | Default  | Description                                                                                                                                                                                                                        | 
|-----------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --index_type                      | IVF_FLAT | When creating an index, you need to specify the index type<FLAT, IVF_FLAT, HNSW, DISKANN, AISAQ, HNSW_PQ, HNSW_SQ>                                                                                                                 |
| --metric_type                     | L2       | Index metric type (<b>L2</b> for Euclidean Distance, <b>IP</b> for Inner Product)                                                                                                                                                  |
| --nlist                           |          | Specify the number of buckets during clustering when creating IVF_FLAT/IVF_PQ index (Generally, it is recommended that nlist can be 4 * sqrt(n))                                                                                   |
| --hnsw_m                          | 16       | The max number of connections that each vector in the graph can have during the HNSW index construction phase                                                                                                                      |
| --efconst                         | 500      | Size of the priority queue that is used to determine which vectors will be connected to a new vector being inserted into the graph of HNSW index                                                                                   |
| --max_degree                      | 56       | Max number of edges that each node in the DISKANN/AISAQ graph can have                                                                                                                                                             |
| --search_list_size                | 100      | Size of the candidate list                                                                                                                                                                                                         |
| --pq_code_budget_gb_ratio         | 0.125    | Ratio of memory allocated to storing PQ (Product Quantization) codes relative to the total memory budget for the AISAQ index                                                                                                       |
| --search_cache_budget_gb_ratio    | 0        | Ratio of cached node numbers to raw data  for the DISKANN/AISAQ index                                                                                                                                                              |
| --disk_pq_code_budget_gb_ratio    | 0.25     | For the DISKANN/AISAQ index - Size limit on the vector code. Set value of 0 to store full precision vectors.                                                                                                                       | 
| --inline_pq                       | -1       | Set the number of pq vectors to be stored inline as part of the index node for the AISAQ index. Valid values -1, 0...max_degree                                                                                                    |
| --rearrange                       | True     | For the AISAQ index - Enable vectors rearrangement during build, when enabled, each vector will be assigned and stored with a new id, in a way that the number of IOs needed to read the PQ vectors during search will be minimal. |
| --num_entry_points                | 100      | For the AISAQ index - Number of entry points that should be generated to be used as a search start points. Value must be between 0 and 1000.                                                                                       |
| --pq_cache_size                   | 0        | For the AISAQ index - PQ vectors cache DRAM size in MiB. Valid values 0-1024 (1GB)                                                                                                                                                 |
| --pq_m                            | 12       | Number of subquantizers for the HNSW_PQ index.                                                                                                                                                                                     |
| --nbits                           | 8        | Number of bits per subquantizer for the HNSW_PQ index.                                                                                                                                                                             |
| --sq_type                         | None     | Scalar quantizer type the HNSW_SQ index. Valid values are "sq6", "sq8", "fp16", "bf16".                                                                                                                                            |
| --refine                          | False    | Whether the refine is used during the train for the HNSW_PQ and HNSW_SQ indexes.                                                                                                                                                   |
| --refine_type                     | None     | The type of refine for the HNSW_PQ and HNSW_SQ indexes. Valid values are value "sq6", "sq8", "fp16", "bf16", "fp32", "flat"                                                                                                        |

**Parameters for data insert**

| Option                     | Default | Description                                                                                                                                                                               |
|----------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --base                     |         | Path to the fvecs file                                                                                                                                                                    |
| --total_vectors            | 20,000  | When the data format is bvecs or fvecs, the amount of data to be written                                                                                                                  |
| --chunk_size               | 20,000  | When the data format is bvecs  or fvecs, the amount of data written into milvus each time(<=256MB)                                                                                        |
| --shards_num               | 1       | Specify number of shards to split the collection (if the collection doesn't exist and created automatically upon insert)                                                                  |
| --additional_columns_files |         | Specify list of columns data files, each file is provide as npy file format, specified with full Path+file name which is the column name and include data related to the specified column |


**Parameters for data upload**

| Option                     | Default      | Description                                                                                                                                                                               |
|----------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --base                     |              | Path to the fvecs/fbin file                                                                                                                                                               |
| --total_vectors            | 0 (all)      | Total number of vectors to be upload                                                                                                                                                      |
| --num_tasks                | 0 (all cpus) | Number of parallel tasks for upload                                                                                                                                                       |
| --max_file_size_mb         | 1024 (1GB)   | The maximum size of each uploaded file in MB (Large files over 2GB might decrease uplod/import time duration)                                                                             | 
| --bucket_name              |              | Object Storage bucket name to upload the file vectors to                                                                                                                                  |
| --upload_desc_file         |              | File (full path) for the upload output list of parquet/json files                                                                                                                         |
| --file_start_index         | 0            | Start index in the given base vectors file to start upload from                                                                                                                           |
| --collection_start_index   | 0            | Start index in the future collection to start import to                                                                                                                                   |
| --bulk_file_type           | parquet      | parquet / json                                                                                                                                                                            |
| --additional_columns_files |              | Specify list of columns data files, each file is provide as npy file format, specified with full Path+file name which is the column name and include data related to the specified column |
| --additional_types         |              | Specify list of fields types correlated to the List of additional fields                                                                                                                  |

**Parameters for complete upload**

| Option                     | Default      | Description                                                               |
|----------------------------|--------------|---------------------------------------------------------------------------|
| --bucket_name              |              | Object Storage bucket name to complete upload the file vectors to         |
| --upload_desc_file         |              | File (full path) for the complete upload input list of parquet/json files |

**Parameters for data import**

| Option                   | Default | Description                                                                                                 |
|--------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| --upload_desc_file       |         | File (full path) contain list of parquet/json files (from pre advanced upload command output) to be import  |

**Parameters for get import progress**

| Option    | Default | Description                                           |
|-----------|---------|-------------------------------------------------------|
| --job_id  |         | The ID of the bulk-import job of your interest        |
| --details | False   | Print progress in more details for each imported file |

**Parameters for performance test**

| Option                    | Default                                | Description                                                                                                                                                                      |
|---------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --query_file              |                                        | Path to the query vectors file                                                                                                                                                   |
| --gt_file                 |                                        | Path to the ground truth file - supported format are fbin, ivecs and parquet (in case of parquet there shoud be 'neighbors_id' column containing each query topk neighbors id's) |
| --result_path             | performance                            | Path to the performance test result file                                                                                                                                         |
| --metric_type             | L2                                     | Index metric type (<b>L2</b> for Euclidean Distance, <b>IP</b> for Inner Product)                                                                                                |
| --nq                      | [10000]                                | Number of queries to be tested                                                                                                                                                   |
| --search_param            |                                        | When querying, specify the parameter value when querying (When the index is of type Ivf, this parameter refers to Nprobe. When the index is HNSW, this parameter refers to EF)   |
| --topk                    | [1,1,10]                               | The topk value to be tested in each np (Here means testing multiple topk values)                                                                                                 |
| --beamwidth               | Calculated according to BeamWidthRatio | The beam width of the indexes (only for DISKANN family index types). Valid values 1..128                                                                                         |
| --vectors_beamwidth       | 1                                      | The beam width of the compressed vectors (only for AISAQ index type). Valid values 1..4 (must be <= beamwidth)                                                                   |
| --pq_read_page_cache_size | 5MB                                    | For the AISAQ index - PQ vectors read page cache DRAM size in bytes - per thread. Valid values 0-32MB                                                                            |

**Parameters for search vector distance**

| Option                  | Default | Description                                                                             |
|-------------------------|---------|-----------------------------------------------------------------------------------------|
| --query_file            |         | Path to the query vectors file                                                          |
| --metric_type           | L2      | Index metric type (<b>L2</b> for Euclidean Distance, <b>IP</b> for Inner Product)       |
| --k                     | 4       | The number of nearest neighbors to retrieve for each query vector                       |
| --probe                 | 10      | Determines how many clusters are visited during the search in IVF Flat index            |
| --refine_k              | 0       | Refine factor (0 value leads to a search without a refine) search in HNSW PQ/SQ indexes |


**Parameters for collection rename**

| Option     | Default | Description                     |
|------------|---------|---------------------------------|
| --new_name |         | The new name for the collection |

**Parameters for collection segments**

| Option    | Default | Description                                |
|-----------|---------|--------------------------------------------|
| --sum_by  |         | Sum segment info by node or by collection  |

## Instructions

**1.Create a Collection**

```shell
python main.py --collection=<collection_name> --dim=<DIMENSION> c
# Use default values from the configuration files
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --dim=<DIMENSION> --additional_fields=<ADDITIONAL_FIELDS> --additional_types=<ADDITIONAL_TYPES> --shards_num=<SHARDS_NUM> c
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --dim=128 --additional_fields=url,content --additional_types=varchar512,varchar62400 --shards_num=2 c
```

**2. Create an index**

```shell
# Use default values from the configuration files
python main.py create_index
python main.py --collection=<COLLECTION NAME> --index_type <index_type> create_index
# Example:
python main.py --collection=my_collection --index_type DISKANN create_index

# Use parameters for AISAQ Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --index_type=AISAQ \
   --metric_type=<METRIC TYPE> --max_degree=<MAX DEGREE> --pq_code_budget_gb_ratio=<PQ_CODE_BUDGET_GB_RATIO> \
   --search_list_size=<SEARCH LIST SIZE> --search_cache_budget_gb_ratio=<SEARCH CACHE BUDGET GB RATIO> \
   --disk_pq_code_budget_gb_ratio=<DISK PQ_CODE_BUDGET_GB_RATIO> create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=AISAQ \
   --metric_type=IP --max_degree=56 --pq_code_budget_gb_ratio=0.05 --search_list_size=100 \ 
   --search_cache_budget_gb_ratio=0.1 --disk_pq_code_budget_gb_ratio=128 create_index

# Use parameters for DISKANN Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --index_type=DISKANN \
   --metric_type=<METRIC TYPE> create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=DISKANN \
   --metric_type=IP create_index
   
# Use parameters for HNSW Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --index_type=HNSW \
   --metric_type=<METRIC TYPE> --hnsw_m=<M> --efconstr=<EF_CONSTRUCTION> create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=HNSW \
   --metric_type=IP --hnsw_m=16 --efconstr=500 create_index
# Use parameters for HNSW PQ Index
# --refine is a flag - omit it for False value, otherwise it will be True 
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --index_type=HNSW_PQ \
   --metric_type=<METRIC TYPE> --hnsw_m=<M> --efconstr=<EF_CONSTRUCTION> --pq_m=<PQ_M> --nbits=<NBITS> --refine --refine_type=<REFINE_TYPE> create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=HNSW_PQ \
   --metric_type=IP --hnsw_m=16 --efconstr=500 --pq_m=64 --nbits=10 --refine --refine_type=flat create_index
# Use parameters for HNSW SQ Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --index_type=HNSW_SQ \
   --metric_type=<METRIC TYPE> --hnsw_m=<M> --efconstr=<EF_CONSTRUCTION> --sq_type=<SQ_TYPE> --refine --refine_type=<REFINE_TYPE> create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=HNSW_SQ \
   --metric_type=IP --hnsw_m=16 --efconstr=500 --sq_type=sq6 --refine --refine_type=flat create_index
# Use parameters for Ivf Flat Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --nlist=<NLIST>--index_type=IVF_FLAT \
   --metric_type=<METRIC TYPE> --dim=<VECTOR DIMENSION>  create_index
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --index_type=IVF_FLAT \
   --metric_type=IP --nlist=1024  create_index
```

**3. Drop index**

Bootcamp implies that only one column can be indexed in a collection, therefore this operation expects only the collection name should be provided. 
```shell
python main.py --collection=<COLLECTION NAME> drop_index
# Example:
python main.py --collection=my_collection drop_index
```

**4. Data insert**

```shell
# Use default values from the configuration files
python main.py --collection <collection_name> insert
# Use parameters
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --dim=<VECTOR DIMENSION> \
--total_vectors=<NUMBER OF VECTORS TO INSERT> --chunk_size=<INSERT CHUNK SIZE> --base=<BASE VECTORS FILE> --additional_columns_files=<ADDITIONAL_COLUMN_FILES> --shards_num=<NUMBER OF SHARDS> insert
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --dim=128 \
--total_vectors=1000000 --chunk_size=1000 --base=/home/potter/data/sift_base.ivf --additional_columns_files=/path/to/url.npy,/path/to/content.npy --shards_num=3 insert
```

**5. Load data to memory**
```shell
python main.py --collection=<COLLECTION NAME> load
# Example:
python main.py --collection=my_collection load
python main.py --collection=my_collection --exclude=source_content load
```

**6. Compact collection**
```shell
python main.py --collection=<COLLECTION NAME> compact
# Example:
python main.py --collection=my_collection compact
```

**7. Performance Test**
```shell
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --search_param=<SEARCH PARAM> \
--result_path=<RESULT PATH> --gt_file=<GROUND TRUTH FILE> --nq=<NUMBER OF QUERIES> --topk <K> \
--query_file=<QUERY FILE PATH> --metric_type=<METRIC TYPE> performance
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --search_param=10 \
--result_path=/tmp --gt_file=/home/potter/data/sift_gt.ivecs --nq=1000 --topk 4 \
--query_file=/home/potter/data/sift_qry.fvecs --metric_type=L2 performance

# Use parameters for HNSW Index
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --search_param=<SEARCH PARAM> \
--result_path=<RESULT PATH> --gt_file=<GROUND TRUTH FILE> --nq=<NUMBER OF QUERIES> --topk <K> \
--query_file=<QUERY FILE PATH> --metric_type=<METRIC TYPE> performance
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --search_param=10 \
--result_path=/tmp --gt_file=/home/potter/data/sift_gt.ivecs --nq=1000 --topk 4 \
--query_file=/home/potter/data/sift_qry.fvecs --metric_type=L2 performance

# Use parameters for HNSW PQ/SQ Indexes
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION NAME> --search_param=<SEARCH PARAM> --refine_k=<REFINE_K> \
--result_path=<RESULT PATH> --gt_file=<GROUND TRUTH FILE> --nq=<NUMBER OF QUERIES> --topk <K> \
--query_file=<QUERY FILE PATH> --metric_type=<METRIC TYPE> performance
# Example:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --search_param=10 --refine_k=1.6 \
--result_path=/tmp --gt_file=/home/potter/data/sift_gt.ivecs --nq=1000 --topk 4 \
--query_file=/home/potter/data/sift_qry.fvecs --metric_type=L2 performance
# Example for DISKANN index:
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --search_param=10 \
--result_path=/tmp --gt_file=/home/potter/data/sift_gt.ivecs --nq=1000 --topk 4 \
--query_file=/home/potter/data/sift_qry.fvecs --metric_type=L2 --beamwidth=2 performance
# Example for AISAQ index::
python main.py --host=172.28.70.64 --port=19530 --collection=my_collection --search_param=10 \
--result_path=/tmp --gt_file=/home/potter/data/sift_gt.ivecs --nq=1000 --topk 4 \
--query_file=/home/potter/data/sift_qry.fvecs --metric_type=L2 --beamwidth=2 --vectors_beamwidth=1 performance
```

**8. Describe the collection metadata**
```shell
python main.py --collection <collection_name> desc
# Example:
python main.py --collection my_collection desc
```

**9. View collection index information**

```shell
python main.py --collection <collection_name> index_info
# Example:
python main.py --collection my_collection index_info
```

**10.Determine whether the collection exists**

```shell
python main.py --collection <collection_name> has
# Example:
python main.py --collection my_collection has
```

**11.View the number of vectors in the collection**

```shell
python main.py --collection <collection_name> rows
# Example:
python main.py --collection my_collection rows
```

**12.Delete the collection**

```shell
python main.py --collection <collection_name> drop
# Example:
python main.py --collection my_collection drop
```

**13. List collections**

```shell
python main.py list
```

**14. Collection's Index Progress**
```shell
python main.py --collection <collectoin_name> index_progress 
# Example:
python main.py --collection my_collection index_progress 
```

**15. Collection's load Progress**
```shell
python main.py --collection <collectoin_name> load_progress
# Example: 
python main.py --collection my_collection load_progress 
```


### Steps for import data to milvus collection
1) Create the collection
2) Upload the relevant data to the Object Storage service
3) Import the uploaded data to the collection
4) You can monitor the import data process with the list_imports and import_progress commands


**16. Data upload**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --base <BASE VECTORS FILE> --upload_desc_file <RESULTS FILE> --bucket_name=<BUCKET NAME> upload
# Use parameters
python main.py --host=<HOST> --port=<PORT> --upload_desc_file <RESULTS FILE> --bucket_name=<BUCKET NAME> \
--total_vectors=<NUMBER OF VECTORS TO INSERT> --file_start_index=<FILE START INDEX> --collection=<COLLECTION> --collection_start_index=<COLLECTION START INDEX> --base=<BASE VECTORS FILE> --bulk_file_type=<BULK FILE TYPE> --additional_columns_files=<ADDITIONAL_COLUMN_FILES> --additional_types=<ADDITIONAL_TYPES> upload
# Example:
python main.py --host=172.28.x.y --port=9000 --total_vectors=1000000 --upload_desc_file=upload_results.txt --collection=my_collection --base=/home/potter/data/sift_base.vecs --additional_columns_files=/path/to/url.npy,/path/to/content.npy --additional_types=varchar512,varchar62400 --file_start_index=100000 --bucket_name=milvus7 --bulk_file_type=json upload
```

**17. Data import**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --collection <COLLECTION> --upload_desc_file <VECTORS TO IMPORT FILE> import
# Use parameters
python main.py --host=<HOST> --port=<PORT> --upload_desc_file <VECTORS TO IMPORT FILE> --collection=<COLLECTION> import
# Example:
python main.py --host=172.28.x.y --port=19530 --upload_desc_file=upload_results.txt --collection=my_collection import
```

**18. Get import progress**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --job_id <JOB ID> import_progress
# Use parameters
python main.py --host=<HOST> --port=<PORT> --job_id <JOB ID> import_progress
# Example:
python main.py --host=172.28.x.y --port=19530 --job_id=46575477 import_progress
```
**19. List imports**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --collection <COLLECTION> list_imports
# Use parameters
python main.py --host=<HOST> --port=<PORT> --collection <COLLECTION> list_imports
# Example:
python main.py --host=172.28.x.y --port=19530 --collection=my_collection list_imports
```

**20. List buckets**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> mc_list_buckets
# Use parameters
python main.py --host=<HOST> --port=<PORT> mc_list_buckets
# Example:
python main.py --host=172.28.x.y --port=9000 mc_list_buckets
```
**21. List objects**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --bucket_name=<BUCKET NAME> mc_list_objects
# Use parameters
python main.py --host=<HOST> --port=<PORT> --bucket_name=<BUCKET NAME> --prefix <PREFIX> mc_list_objects
# Example:
python main.py --host=172.28.x.y --port=9000 --bucket_name=milvus3 --prefix falcon786d mc_list_objects
```
**22. Disk usage**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --bucket_name=<BUCKET NAME> mc_disk_usage
# Use parameters
python main.py --host=<HOST> --port=<PORT> --bucket_name=<BUCKET NAME> --prefix <PREFIX> mc_disk_usage
# Example:
python main.py --host=172.28.x.y --port=9000 --bucket_name=milvus3 --prefix falcon786d mc_disk_usage
```
**23. Remove objects**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --bucket_name=<BUCKET NAME> mc_remove_objects
# Use parameters
python main.py --host=<HOST> --port=<PORT> --bucket_name=<BUCKET NAME> --prefix <PREFIX> --dry_run mc_remove_objects
# Example:
python main.py --host=172.28.x.y --port=9000 --bucket_name=milvus3 --prefix falcon786d --dry_run mc_remove_objects
```

**24. Rename a collection**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --collection=<OLD NAME> --new_name <New Name> rename
python main.py --host=172.28.x.y --port=19530 --collection=Benchmark1 --new_name Benchmark rename
```

**25. Collection segments**

```shell
# Use default values from the configuration files
python main.py --host=<HOST> --collection=<COLLECTION> segments
# Use parameters
python main.py --host=<HOST> --port=<PORT> --collection=<COLLECTION> segments
# Example:
python main.py --host=172.28.x.y --port=19530 --collection=my_collection --sum_by=collection/node segments
```

## Project Sources
Clone the project `git clone https://dssd-bitbucket.us.kioxia.com/scm/tes/bootcamp.git`

```