#! /bin/bash

# set -e

export MINIO_IP=10.185.121.71
export MILVUS_IP=10.185.121.72
export COLLECTION_NAME=falcon_1024_160M
export DESC_FILE=falcon_desc_160M.txt
export BASE_FILE=/raid/data/falcon_1024_160m_mock_dataset/cluster_stats_falcon_extract_10M_seed42_representative_normalized_sample9999360_n30000_lowrankpca16_300/mock_vectors.fbin

export MILVUS_MIXCOORD_LOG_LOC=$(sudo ls /var/log/pods | grep mixcoord | head -n 1)
export ZERO_LOG_MIXCOORD=/var/log/pods/$MILVUS_MIXCOORD_LOG_LOC/mixcoord/0.log

cd Bootcamp/milvus-bootcamp
source venv/bin/activate

date
python3 main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --dim 1024 create
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
