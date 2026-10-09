#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' # No Color

usage(){
    echo "Usage: $0 <HNSW | DISKANN | IVF_FLAT> <L2 | IP> <short | long>"
    exit 1
}

if [ "$#" -ne 3 ]; then
  usage
fi

if [ "$3" == "short" ]; then
    create="c"
    list="l"
    insert="a"
    rows="r"
    create_index="I"
    index_progress="p"
    index_info="i"
    load="L"
    load_progress="s"
    search="S"
    performance="P"
    release="R"
    has="h"
    drop="D"
    describe="d"
elif [ "$3" == "long" ]; then
    create="create"
    list="list"
    insert="insert"
    create_index="create_index"
    rows="rows"
    index_progress="index_progress"
    index_info="index_info"
    load="load"
    load_progress="load_progress"
    search="search"
    performance="performance"
    release="release"
    has="has"
    drop="drop"
    describe="desc"
else
    usage
fi

if [ "$2" == "L2" ] || [ "$2" == "IP" ]; then
  METRIC_TYPE=$2
else
  echo "Bad metric type: $2"
  usage
fi

if [ "$1" == "IVF_FLAT" ] || [ "$1" == "DISKANN" ] || [ "$1" == "HNSW" ]; then
  INDEX_TYPE=$1
else
  echo "Bad index type: $1"
  usage
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST=127.0.0.1
PORT=19530
P='python'
MAIN=$SCRIPT_DIR'/main.py'
COLL=sift_small
BASE_VECTORS_FILE=$SCRIPT_DIR'/data/siftsmall_base.fvecs'
QRY_VECTORS_FILE=$SCRIPT_DIR'/data/siftsmall_query.fvecs'
GT_FILE=$SCRIPT_DIR'/data/siftsmall_groundtruth.ivecs'
RES_PATH='/tmp'

if [ ! -f "$BASE_VECTORS_FILE" ]; then
    echo "Error: File $BASE_VECTORS_FILE does not exist."
    exit 1
elif [ ! -f "$QRY_VECTORS_FILE" ]; then
    echo "Error: File $QRY_VECTORS_FILE does not exist."
    exit 1
elif [ ! -f "$GT_FILE" ]; then
    echo "Error: File $GT_FILE does not exist."
    exit 1
fi

echo "About to test $INDEX_TYPE index"
echo
#echo "This will be search"
#echo $P $MAIN --host=$HOST --port=$PORT --collection=$COLL --k=1 --input=$QRY_VECTORS_FILE $search
echo
printf "Test drop collection (1). ${GREEN}Possibly the collection does not exist. It's not an error${NC}\n"
$P $MAIN --collection=$COLL --host=$HOST --port=$PORT $drop
echo ---------------------------------------------------------
echo

echo "Test create collection (1)."
echo "$MAIN --collection=$COLL --dim=128 --host=$HOST --port=$PORT $create"
$P $MAIN --collection=$COLL --dim=128 --host=$HOST --port=$PORT $create
echo ---------------------------------------------------------
echo

echo "Test list collections."
$P $MAIN --host=$HOST --port=$PORT $list
echo ---------------------------------------------------------
echo

echo "Test Describe collection (1)."
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $describe
echo ---------------------------------------------------------
echo

echo "Test insert vectors from file (1)."
  $P $MAIN --host=$HOST --port=$PORT --collection=$COLL --dim=128 --base=$BASE_VECTORS_FILE $insert
echo ---------------------------------------------------------
echo

echo "Test rows in collection (1). Expected to be 10,000 rows"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $rows
echo ---------------------------------------------------------
echo

echo "Test insert vectors from file (2)."
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL --dim=128 --base=$BASE_VECTORS_FILE \
  --total_vectors=9999 --chunk_size=3333 $insert
echo ---------------------------------------------------------
echo

echo "Test rows in collection (2). Expected to be 19,999 rows"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $rows
echo ---------------------------------------------------------
echo

echo "Test create $INDEX_TYPE index."
if [ "$INDEX_TYPE" == "IVF_FLAT" ]; then
  $P $MAIN --host=$HOST --port=$PORT --collection=$COLL --index_type=$INDEX_TYPE \
  --dim=128 --metric_type=$METRIC_TYPE  $create_index
elif [ "$INDEX_TYPE" == "DISKANN" ]; then
  $P $MAIN --host=$HOST --port=$PORT --collection=$COLL --index_type=$INDEX_TYPE \
   --metric_type=$METRIC_TYPE --max_degree=64 --search_list=128 --pq_code_budget_gb_ratio=0.25 \
   $create_index
elif [ "$INDEX_TYPE" == "HNSW" ]; then
  $P $MAIN --host=$HOST --port=$PORT --collection=$COLL --index_type=$INDEX_TYPE \
  --dim=128 --metric_type=$METRIC_TYPE --efconst=200 --hnsw_m=16 $create_index
else
  echo The $INDEX_TYPE is not supported
  exit 1
fi
echo ---------------------------------------------------------
echo

echo "Test Describe collection (2)."
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $describe
echo ---------------------------------------------------------
echo



echo "Test index progress"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $index_progress
echo ---------------------------------------------------------
echo

echo "Test index info"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $index_info
echo ---------------------------------------------------------
echo


echo "Test load collection"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $load
echo ---------------------------------------------------------
echo

echo "Test load progress"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $load_progress
echo ---------------------------------------------------------
echo

echo "Test search"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL --k=1 --query_file=$QRY_VECTORS_FILE $search
echo ---------------------------------------------------------
echo

echo "Test performance"
$P $MAIN --collection=$COLL --host=$HOST --port=$PORT --search_param=10 --result_path=$RES_PATH \
--gt_file=$GT_FILE --query_file=$QRY_VECTORS_FILE --nq=10 --topk=1,2,4,10 performance
echo ---------------------------------------------------------
echo

echo "Test release collection"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $release
echo ---------------------------------------------------------
echo

echo "Test has collection (1)"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $has
echo ---------------------------------------------------------
echo

echo "Test drop collection"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $drop
echo ---------------------------------------------------------
echo

echo "Test has collection (2)"
$P $MAIN --host=$HOST --port=$PORT --collection=$COLL $has
echo ---------------------------------------------------------
echo

