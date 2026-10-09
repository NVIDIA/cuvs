#! /bin/bash

# set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
export COLLECTION_NAME=falcon_1024_120M
export DESC_FILE=falcon_desc_120M.txt
export BASE_FILE=/raid/data/falcon_1024_120M_dataset/base1_falcon_1024_120M.fbin

export MILVUS_MIXCOORD_LOG_LOC=$(sudo ls /var/log/pods | grep mixcoord | head -n 1)
export ZERO_LOG_MIXCOORD=/var/log/pods/$MILVUS_MIXCOORD_LOG_LOC/mixcoord/0.log

cd Bootcamp/milvus-bootcamp
source venv/bin/activate

# microk8s kubectl create -f Milvus/milvus-cluster-cpu-160m.yaml; sleep 200
date
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --dim 1024 create
### 
# Import subset of the original dataset
###
rm -f $DESC_FILE; python3 main.py --num_tasks 120 --host=$MINIO_IP --collection $COLLECTION_NAME --base $BASE_FILE --bucket_name milvus-cpu-120 --upload_desc_file $DESC_FILE --max_file_size_mb 200 upload
date
echo "Import"
import_logs=$(python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --upload_desc_file $DESC_FILE import)
echo $import_logs
export IMPORT_JOBID=$(echo $import_logs | grep "jobId" | cut -d "\"" -f 8)
echo "IMPORT_JOBID: $IMPORT_JOBID"
sleep 120 # 2 minutes
import_exit_code=2
while [ $import_exit_code -eq 2 ]; do
    python3 ../../scripts/wait_import_complete.py
    import_exit_code=$?
    sleep 60 # 1 minutes
done
date;
sudo ../../scripts/grep_log_simple.sh $ZERO_LOG_MIXCOORD "clustering compaction task total elapse" "\[lastState=analyzing\] \[currentState=failed\]"
date
sleep 120
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
sleep 60
python check_index.py
date
cd ../../
bash scripts/extract_logs_datanode.sh
python scripts/extract_aisaq_build_metrics.py datanode -o aisaq_original_gpu_120_15m.csv
mv datanode datanode_original_cpu_120_15m

python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME drop_index
date

microk8s kubectl delete -f Milvus/milvus-cluster-cpu-160m.yaml
sleep 30
microk8s kubectl create -f Milvus/milvus-cluster-gpu-160m.yaml
sleep 230

python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
sleep 60
python check_index.py
date

bash scripts/extract_logs_datanode.sh
python scripts/extract_aisaq_build_metrics.py datanode -o aisaq_original_gpu_120_15m.csv
mv datanode datanode_original_gpu_120_15m