#! /bin/bash

set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
# export COLLECTION_NAME=falcon_1024_240M
export COLLECTION_NAME=falcon_1024_1M
export DESC_FILE=falcon_desc_1M.txt
# export BASE_FILE=/raid/data/falcon_1024_240M_dataset/base1_falcon_1024_240M.fbin
export BASE_FILE=/raid/data/falcon_1024_960M_dataset/base_falcon_1024_1M.fbin

#microk8s kubectl create namespace kioxia
#microk8s kubectl create -f Milvus/milvus-cluster.yaml
#sleep 200
cd Bootcamp/milvus-bootcamp
source venv/bin/activate
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --dim 1024 create
rm $DESC_FILE
python3 main.py --host=$MINIO_IP --collection $COLLECTION_NAME --base $BASE_FILE --bucket_name milvus3 --upload_desc_file $DESC_FILE upload

export MILVUS_MIXCOORD_LOG_LOC=$(sudo ls /var/log/pods | grep mixcoord | head -n 1)
export ZERO_LOG=/var/log/pods/$MILVUS_MIXCOORD_LOG_LOC/mixcoord/0.log

import_logs=$(python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --upload_desc_file $DESC_FILE import)
echo $import_logs
if [ $? -ne 0 ]; then
    echo "Import failed"
    exit 1
fi
export IMPORT_JOBID=$(echo $import_logs | grep "jobId" | cut -d "\"" -f 8)
echo "IMPORT_JOBID: $IMPORT_JOBID"
sleep 30

python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --job_id $IMPORT_JOBID wait_import_complete
date;
sudo ../../grep_log_simple.sh "clustering compaction task total elapse" $ZERO_LOG;
date
exit 0 ## CLUSTERING COMPACTION





sleep 5
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
python check_index.py
sleep 5
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME load
#
## export MILVUS_DATANODE_LOG_LOC=$(sudo ls /var/log/pods | grep datanode | head -n 1)
## export ZERO_LOG=/var/log/pods/$MILVUS_DATANODE_LOG_LOC/datanode/0.log
### SEARCH
#
cd ../../VectorDBBench/vectordbbench
source venv/bin/activate

export MILVUS_CONFIG_FILE=/raid/kioxia-dev/Version-2.6-23273/Benchmark_Instructions/VectorDBBench_9_instances_960M_milvus_config.yaml
sleep 300
cd Bootcamp/milvus-bootcamp
source venv/bin/activate
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME load
cd ../../VectorDBBench/vectordbbench
source venv/bin/activate
export RESULTS_DIR=result_retry
python3 run_vectordbbench.py --case=milvusaisaq --config_file=$MILVUS_CONFIG_FILE --out results_960M --search_list=5,10,20,30,40,50,60,70,75,80,85,90,95,100
python3 ../../Benchmark_Instructions/json_files_to_csv.py 'vectordb_bench/results/Milvus/result_20260224*.json' results_960M/${RESULTS_DIR}.csv
mkdir vectordb_bench/results/Milvus/$RESULTS_DIR
mv vectordb_bench/results/Milvus/result_20260224*.json vectordb_bench/results/Milvus/$RESULTS_DIR
