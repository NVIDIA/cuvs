#! /bin/bash

set -e

export MINIO_IP=10.185.121.70
export MILVUS_IP=10.185.121.71
export COLLECTION_NAME=falcon_1024_5B



source Bootcamp/milvus-bootcamp/venv/bin/activate

#python3 Bootcamp/milvus-bootcamp/main.py --host=$MILVUS_IP --collection $COLLECTION_NAME release
#microk8s kubectl delete -f Milvus/milvus-cluster-5B.yaml
#microk8s kubectl create -f Milvus/milvus-cluster-5B.yaml
#sleep 200
python3 Bootcamp/milvus-bootcamp/main.py --host=$MILVUS_IP --collection $COLLECTION_NAME load