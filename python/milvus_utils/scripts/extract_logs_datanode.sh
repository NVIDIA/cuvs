#! /bin/bash

set -e

DATANODES=$(sudo ls /var/log/pods/ | grep datanode)
for DATANODE in $DATANODES; do
    sudo cp -r /var/log/pods/$DATANODE/ datanode/
done