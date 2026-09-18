#!/usr/bin/env bash
set -euo pipefail

SCALE=10
NOISE=0.02
SCHEME=PARTIAL
# SCHEME=FULL
# NORMALIZE=""
NORMALIZE=-n

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"

BC_PATH=$SCRIPT_DIR/../..
QD_PATH=$BC_PATH/../../../vector-db-benchmark
ARTEFACTS=$HOME/work/artefacts
SCALED_PATH=$ARTEFACTS/SCALED_$SCALE
DS=/ANN-Repo2/liorf/falcon1024_1M_dataset/base_falcon_1024_1M.fbin
SCALED_DS=$SCALED_PATH/base_falcon_1024_"$SCALE"M
QRY_PATH=/ANN-Repo2/liorf/falcon1024_1M_dataset/query_falcon_1024_10K.fbin
GT_PATH="$SCALED_DS"_gt.fbin

COLLECTION=falcon"$SCALE"m


source $BC_PATH/../../vnvbc/bin/activate

rm -rf $SCALED_PATH
mkdir $SCALED_PATH

echo "======================== SCALING IS ABOUT TO START $(date) ======================================="
cd $BC_PATH
# python -m scale.cli -i $DS -o $SCALED_DS.fbin -s $SCALE -a $NOISE -m $SCHEME $NORMALIZE -t 5e-4 -x 123
python -m scale.cli -i $DS -o $SCALED_DS.fbin -s $SCALE -a $NOISE -m $SCHEME $NORMALIZE -t 3e-4 -x 123 -w 50


SCALED_BYTES=$(ls -l $SCALED_DS.fbin | awk ' { print $5 } ')
NNORM=$(echo "($SCALED_BYTES - 8) / 1024 / 4" | bc)

echo 
echo "***************** NORMALIZED DATASET SAVED TO $SCALED_DS.fbin **************"
echo "======================================== SCALING DONE NOW $(date) ======================================================"


ls $SCALED_DS.fbin


DESC_FILE=falcon2m.desc
rm -rf $DESC_FILE

python main.py --host=172.28.55.131 --collection=$COLLECTION --dim=1024 create

echo "======================================== COLLECTION $COLLECTION IS CREATED ======================================================"
echo "======================================== UPLOAD IS ABOUT TO START  $(date) ======================================================"
python main.py \
    --host=172.28.55.130 \
    --base $SCALED_DS.fbin \
    --upload_desc_file $DESC_FILE \
    --collection $COLLECTION \
    --bucket_name=milvus3 \
    --max_file_size_mb 635 \
    upload
echo "======================================== UPLOAD IS DONE ======================================================"
echo "======================================== IMPORT IS ABOUT TO START  $(date) ======================================================"
python main.py --host=172.28.55.131 --collection=$COLLECTION --dim 1024 --upload_desc_file $DESC_FILE import
python $SCRIPT_DIR/wait-import.py -a 172.28.55.131 -c $COLLECTION -r $NNORM
echo "======================================== IMPORT IS ABOUT DONE ======================================================"
echo "======================================== INDEXING IS ABOUT TO START  $(date) ======================================================"
python main.py  -a 172.28.55.131 \
    --metric_type=L2 \
    --max_degree=64 \
    --search_list=200 \
    --pq_code_budget_gb_ratio=0.25 \
    --disk_pq_code_budget_gb_ratio=0 \
    -c $COLLECTION \
    --inline_pq -1 \
    --num_entry_points=0 \
    --index_type=AISAQ \
    create_index

python $SCRIPT_DIR/wait-load.py -a 172.28.55.131 -c $COLLECTION

echo "======================================== INDEXING IS FINISHED  COLLECTION $COLLECTION IS LOADED ======================================================"

echo "======================================== COMPUTE GROUNDTRUTH IS ABOUT TO START  $(date) ======================================================"
compute_groundtruth \
	--data_type float \
	--dist_fn l2 \
	--base_file  $SCALED_DS.fbin \
	--query_file $QRY_PATH \
	--gt_file "$GT_PATH" \
	--K 100


echo 
echo "======================================== GT COMPUTATION DONE: "$GT_PATH" =========================================================="
echo

echo "======================================== BOOTCAMP PERFORMANCE TEST IS ABOUT TO START  $(date) ======================================================"
python $BC_PATH/main.py \
    --host=172.28.55.131 \
    --collection=$COLLECTION \
    --search_param=10 \
    --result_path=$SCALED_PATH/res \
    --gt_file="$GT_PATH" \
    --nq=10 --topk 10 \
    --query_file=$QRY_PATH \
    --metric_type=L2 \
    performance
echo "======================================== BOOTCAMP PERFORMANCE TEST IS FINISHED ======================================================"


echo "======================================== ABOUT TO CREATE HDF5  $(date) ======================================================"
python $BC_PATH/hdf5/create_hdf5.py \
    --gt "$GT_PATH" \
    --base $SCALED_DS.fbin \
    --query $QRY_PATH \
    --hdf5 $SCALED_DS.hdf5

echo "======================================== $(ls $SCALED_DS.hdf5) IS CREATED $(date) ======================================================"




cd $QD_PATH
source $QD_PATH/vnvqdrant/bin/activate

sed -i "s#FALCON_SCALED_NORM_PLACEHOLDER#$SCALED_DS.hdf5#" $QD_PATH/datasets/datasets.json

echo "======================================== ABOUT TO RUN QDRANT SEARCH $(date) ======================================================"
python run.py \
    --host 172.28.55.131 \
    --collection $COLLECTION \
    --engines milvus-on-disk-aisaq-inlinepq-auto \
    --datasets falcon-scaled-norm \
    --skip-upload
echo "======================================== QDRANT SEARCH IS FINISHED $(date) ======================================================"
sed -i "s#$SCALED_DS.hdf5#FALCON_SCALED_NORM_PLACEHOLDER#" $QD_PATH/datasets/datasets.json