# VectorDBBench wrapper Tool
This utility is designated to create graphic representation of VectorDBBench performance test results.

## Prerequisites
* Python >= 3.11
* Running Milvus > 2.4.*  
* Proxy is configured if required


### Installation
Download the VectorDBBench from [From Jenkins Build](https://10.93.66.62:31648/job/VectorDBBench/) and unzip it.
```shell
unzip vectordbbench-1.1.<release>.zip	
cd vectordbbench
./install.sh
source venv/bin/activate
```
### Uninstall
The most secure way to uninstall VectorDBBench is to delete the directory containing all the VectorDBBench files.

### Upgrade
Remove the directory containing the current VectorDBBench files. Download and install the new version following the instructions above. 

### Preparations Steps 
1) Create queries test.parquet file AND groundtruth neighbors.parquet file using convert_to_parquet.py script - see usage below
2) Create Milvus collection and insert/upload data to it
3) When creating Milvus collection, please adhere to the following naming convention:
   - Only alphanumeric characters or dashes should be used for collection name
   - The words 'search' or 'upload' <b>must not</b> be used
4) Create index (DISKANN, AISAQ, HNSWPQ or IVF_FLAT) on the collection and load it

## Create queries test.parquet and groundtruth neighbors.parquet files
`python3 convert_to_parquet.py --queries_file=<queries fbin source file full path> --gt=<groundtruth source file full path> --parquet_path=<output test.parquet/neighbors.parquet full path> --num_queries=<number of queries to convert>
`

**Arguments for convert_to_parquet**

| Option         | Description                                     |
|----------------|-------------------------------------------------|
| --queries_file | queries fbin source file full path              |
| --gt           | groundtruth source file full path               |
| --parquet_path | output test.parquet/neighbors.parquet full path |
| --num_queries  | number of queries to convert                    |



### Running VectorDBBench performance tests
`python3 run_vectordbbench.py --case=<milvus index type case> --config_file=milvus_config.yaml --num-concurrency=<num processes> --k=<topk> --search_list/ef_search/nprobe=<list of search params>`

**Arguments for performance tests**

| Option        | Default                     | Description                                                                                            |
|---------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| --case        |                             | Enum of the index type on the tested collection: milvusdiskann, milvusaisaq, milvushnsw, milvusivfflat |
| --config_file | vectordb_bench/config-files | milvus_config.yaml to be placed on vectordb_bench/config-files project path                            |
| --out         | local path                  | Full path where to copy output results json files for the bmchart tool                                 |
| --search_list |                             | List of search parameters (comma separated) for DISKANN and AISAQ                                      |
| --ef_search   |                             | List of search parameters (comma separated) for HNSWPQ                                                 |
| --nprobe      |                             | List of search parameters (comma separated) for IVF_FLAT                                               |
 
Output results json files will be placed in vectordb_bench/results/Milvus project path.  


**Parameters to config in milvus_config.yaml**

| Option                     | Description                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|
| skip_search_serial         | Skip recall tests - should be always False                                                    |
| case_type                  | Which dataset to use - should be always PerformanceCustomDataset                              |
| num_concurrency            | Number of concurrency search processes                                                        |
| concurrency_duration       | Adjusts the duration in seconds of each concurrency search  [default: 30]                     |
| k                          | top-k results from Milvus                                                                     |
| custom_case_name           | Test name                                                                                     |
| custom_dataset_dim         | Dataset dimension                                                                             |
| custom_dataset_dir         | Path of test.parquet and neighbors.parquet files                                              |
| custom_dataset_file_count  | Number of train parquet files. Not relevant to us – should be always 1                        |
| custom_dataset_metric_type | Index metric type: L2/IP                                                                      |
| custom_dataset_name        | Collection name                                                                               |
| custom_dataset_size        | Collection number of vectors                                                                  |
| drop_old                   | Drop exists collection - Not relevant to us – should be always False!                         |
| load                       | Load data to collection - Not relevant to us – should be always False!                        |
| uri                        | Milvus instance URI - http://[IP]:19530                                                       |
| beamwidth                  | The beamwidth to be used for search - relevant only for milvusdiskann and milvusaisaq cases   |
| vectors_beamwidth          | PQ vector beam width - relevant only for milvusaisaq case                                     |
| pq_read_page_cache_size    | PQ vectors read page cache DRAM size in bytes per thread - relevant only for milvusaisaq case |


## Instructions

**Running performance tests**

```shell
# Use default values
python3 run_vectordbbench.py --case=<milvus index type case> --search_list/ef_search/nprobe =<list of search params>
# Full usage
python3 run_vectordbbench.py --case=<milvus index type case> --config_file=milvus_config.yaml --out=<output files full path> --search_list/ef_search/nprobe=<list of search params>
# Example:
python3 run_vectordbbench.py --case=milvusdiskann --config_file=milvus_config.yaml --out=/path/to/output --search_list=10,30,80,100,130,160,200
```


## Project Sources
Clone the project `git clone https://dssd-bitbucket.us.kioxia.com/scm/tes/vectordbbench.git`


### Create charts
## Using VectorDBBench with bmchart utility

### Prerequisites
1. Python 3.7 or later
2. The <b>Poetry</b> Python dependency management tool

#### Installation ON A SEPARATED PATH OTHER THAN THE VectorDBBench INSTALLATION  
1. Clone the Qdrant benchmark tool from 
the [Bitbucket repository](https://dssd-bitbucket.us.kioxia.com/projects/TES/repos/vector-db-benchmark/browse)
2. Set project root as a working directory: ```cd vector-db-benchmark```
3. Create python virtual environment and activate it:
```bash
python3 -m venv venvqdrant
source ./venvqdrant/bin/activate
```
4. Install [Poetry dependency management tool](https://python-poetry.org) ```pip3 install poetry```  
5. Install dependencies: ```poetry install --no-root```
6. Make sure, that the Poetry environment is activated by ``` poetry shell```
7. Run ```python3 ./bmchart.py -h```, to see the available tool options:
```bash
python ./bmchart.py -h
usage: bmchart.py [-h] -i INPUT [-o OUTPUT] [-p {y,Y,yes,Yes,n,N,no,No}] [-a {y,Y,yes,Yes,n,N,no,No}] [-m {engine,dataset}] [-d {3,4,5,6,7,8}] [-r THREADS] [-x SUFFIX] [-f FILTER]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory with json files generated by Qdrant benchmark
  -o OUTPUT, --output OUTPUT
                        Directory to save the chart
  -p {y,Y,yes,Yes,n,N,no,No}, --print {y,Y,yes,Yes,n,N,no,No}
                        Print benchmark json to standard output
  -a {y,Y,yes,Yes,n,N,no,No}, --annotate {y,Y,yes,Yes,n,N,no,No}
                        Annotate dots with Y value instead of printing Y axe labels
  -m {engine,dataset}, --mult {engine,dataset}
                        Create multi-experiment curves chart
  -d {3,4,5,6,7,8}, --decimal {3,4,5,6,7,8}
                        Number of decimal digits on the Y axe ticks 3>=n<=10
  -r THREADS, --threads THREADS
                        Filter experiments by number of threads
  -x SUFFIX, --suffix SUFFIX
                        Chart file name suffix
  -f FILTER, --filter FILTER
                        Filter benchmarks to render by parameter: dataset|experiment|engine|parallel|top=<regex> e.g. dataset=glove-25*.
```
### Running a single performance test
1. Run the VectorDBBench wrapper run_vectordbbench.py as described above. Specify --out param for the output json results files which will be the input for the bmcahrt
2. Run the bmchart tool
```bash
python ./scripts/bmchart.py -i <VectorDBBench wrapper output folder>/ -o ./charts/
```


To be able using the bmchart filter charts mechanism it is possible to repeat the same serial running test to the same output folder with various custom_case_name values on the same dataset and use the bmchart with the filter option accordingly:
### Running multiple performance tests to compare results
1. Run the VectorDBBench wrapper several times. For each test specify different custom_case_name on the milvus_config.yaml. 
2. Run the bmchart tool. <b>Caution</b>: Running the bmchart tool over a large number of 
results may produce a cluttered and unreadable chart.
3. To avoid cluttered charts use filters 
```bash
python ./bmchart.py -i ../examples/results/ -o ../examples/charts/ -x double.curve -f "experiment=falcon*" -p y
```
The number next to each dot represents the search parameter. In case of the diskann index it's the ```searchList``` parameter.

### Running multiple performance tests to compare results on a single curve
It is possible to create an aggregated chart, where each experiment is represented by dots on a curve instead of 
the entire curve. To generate an aggregated chart, use the ```-m dataset|engine``` option. 
This option instructs the bmchart tool to group the data by either dataset or engine:
1. Run the VectorDBBench wrapper several times. For each test specify different custom_case_name on the milvus_config.yaml. 
2. Run the bmchart tool
```bash
python ./bmchart.py -i ../results/ -o ../charts/ -m engine
```
3. Use this command to create an aggregated chart based on a single dataset:
```bash
python ./bmchart.py -i ../results/ -o ../charts/ -f "dataset=falcon2M" -m engine
```
The bmchart tool uses the following mechanism to determine which result of which experiment is represented by a dot:
- Sort and group all experiment results by the ```mean_precisions``` attribute. 
- Filter the results to retain only those with the highest Y-axis value (e.g., rps, mean_time, etc.) within each group.

For more information about the bmchart tool - See bmchart.py usage (from qdrant vector DB benchmark tool):
https://dssd-bitbucket.us.kioxia.com/projects/TES/repos/vector-db-benchmark/browse/scripts/README.md
