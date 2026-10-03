#! /bin/bash

set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
export COLLECTION_NAME=falcon_1024_120M
export DESC_FILE=falcon_desc_120M.txt


cd Bootcamp/milvus-bootcamp
source venv/bin/activate

date
python main.py --host=$MILVUS_IP --collection $COLLECTION_NAME --index_type AISAQ --metric_type L2 --max_degree 64 --search_list 256 --pq_code_budget_gb_ratio 0.0417 --inline_pq 0 --rearrange True --num_entry_points=1000 --disk_pq_code_budget_gb_ratio=0.25 create_index
sleep 60
python check_index.py
date
cd ../../
bash scripts/extract_logs_datanode.sh
python scripts/extract_aisaq_build_metrics.py datanode -o aisaq_latest_cpu.csv
mv datanode datanode_gpu