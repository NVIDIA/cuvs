#! /bin/bash

set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
export COLLECTION_NAME=falcon_1024_5B
export DESC_FILE=falcon_desc_5B.txt
export BASE_FILE=/raid/data/falcon_1024_5b_mock_dataset/falcon_extract_10M_seed42_representative_normalized_sample9999360_n30000_lowrankpca32_300/mock_vectors.fbin

#microk8s kubectl create namespace kioxia
#microk8s kubectl create -f Milvus/milvus-cluster.yaml
#sleep 200

export MILVUS_MIXCOORD_LOG_LOC=$(sudo ls /var/log/pods | grep mixcoord | head -n 1)
export ZERO_LOG_MIXCOORD=/var/log/pods/$MILVUS_MIXCOORD_LOG_LOC/mixcoord/0.log

cd Bootcamp/milvus-bootcamp
source venv/bin/activate

sleep 10800 # 3 hours

date
echo "Import"
import_logs=$(python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --upload_desc_file $DESC_FILE import)
echo $import_logs
export IMPORT_JOBID=$(echo $import_logs | grep "jobId" | cut -d "\"" -f 8)
echo "IMPORT_JOBID: $IMPORT_JOBID"
sleep 3600 # 60 minutes
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --job_id $IMPORT_JOBID wait_import_complete
date;
sudo ../../grep_log_simple.sh $ZERO_LOG_MIXCOORD "clustering compaction task total elapse" "\[lastState=analyzing\] \[currentState=failed\]"
date
sleep 60
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
sleep 60
python check_index.py
sleep 5
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME load
date
### SEARCH

export MILVUS_CONFIG_FILE=/raid/kioxia-dev/Version-2.6-23273/Benchmark_Instructions/VectorDBBench_9_instances_5B_milvus_config.yaml
export RESULTS_DIR=result_5B

cd ../../VectorDBBench/vectordbbench
source venv/bin/activate
python3 run_vectordbbench.py --case=milvusaisaq --config_file=$MILVUS_CONFIG_FILE --out results_5B --search_list=10,20,30,40,50,60,70,75,80,85,90,95,100
python3 ../../Benchmark_Instructions/json_files_to_csv.py 'vectordb_bench/results/Milvus/result_20260311*.json' results_5B/${RESULTS_DIR}.csv
mkdir vectordb_bench/results/Milvus/$RESULTS_DIR
mv vectordb_bench/results/Milvus/result_20260311*.json vectordb_bench/results/Milvus/$RESULTS_DIR
