#! /bin/bash

set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
export COLLECTION_NAME=falcon_1024_240M
export DESC_FILE=falcon_desc_240M.txt
export BASE_FILE=/raid/data/falcon_1024_240M_dataset/base1_falcon_1024_240M.fbin

#microk8s kubectl create namespace kioxia
#microk8s kubectl create -f Milvus/milvus-cluster.yaml
#sleep 200

export MILVUS_MIXCOORD_LOG_LOC=$(sudo ls /var/log/pods | grep mixcoord | head -n 1)
export ZERO_LOG_MIXCOORD=/var/log/pods/$MILVUS_MIXCOORD_LOG_LOC/mixcoord/0.log

cd Bootcamp/milvus-bootcamp
source venv/bin/activate
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --dim 1024 create
##rm $DESC_FILE
##python3 main.py --host=$MINIO_IP --collection $COLLECTION_NAME --base $BASE_FILE --bucket_name milvus3 --upload_desc_file $DESC_FILE upload
import_logs=$(python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --upload_desc_file $DESC_FILE import)
echo $import_logs
export IMPORT_JOBID=$(echo $import_logs | grep "jobId" | cut -d "\"" -f 8)
echo "IMPORT_JOBID: $IMPORT_JOBID"
sleep 600 # 10 minutes
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --job_id $IMPORT_JOBID wait_import_complete
date;
sudo ../../grep_log_simple.sh $ZERO_LOG_MIXCOORD "clustering compaction task total elapse";
date
##exit 0
sleep 60
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
sleep 60
python check_index.py
sleep 30
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME load
#
## export MILVUS_DATANODE_LOG_LOC=$(sudo ls /var/log/pods | grep datanode | head -n 1)
## export ZERO_LOG=/var/log/pods/$MILVUS_DATANODE_LOG_LOC/datanode/0.log
### SEARCH
#
cd ../../VectorDBBench/vectordbbench
source venv/bin/activate

export MILVUS_CONFIG_FILE=/raid/kioxia-dev/Version-2.6-23273/Benchmark_Instructions/VectorDBBench_9_instances_240M_milvus_config.yaml

export RESULTS_DIR=result_240M_segment_filter_1_5
python3 run_vectordbbench.py --case=milvusaisaq --config_file=$MILVUS_CONFIG_FILE --out results_240M --search_list=10,20,30,40,50,60,70,75,80,85,90,95,100
python3 ../../Benchmark_Instructions/json_files_to_csv.py 'vectordb_bench/results/Milvus/result_20260310*.json' results_240M/${RESULTS_DIR}.csv
mkdir vectordb_bench/results/Milvus/$RESULTS_DIR
mv vectordb_bench/results/Milvus/result_20260310*.json vectordb_bench/results/Milvus/$RESULTS_DIR
