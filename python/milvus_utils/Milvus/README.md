# kioxia milvus

## Installation

Supported OS:
-------------
Ubuntu 20.x 22.x


DEB installation:
-----------------
Before installation:
You need to prepare the milvus working folder /var/lib/milvus and mount it on fast storage
according to performance need.

Prerequisite - cuda:

You need to install cuda rt libraries(if you have cuda toolkit or cuda_rt >= 12.8 you can skip this step):

```bash
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt install cuda-cudart-12-8
```

Install:

```bash
sudo dpkg -i milvus_<major>.<minor>-<release>-1_amd64.deb
```

The milvus service will be starting automatically after the installation, you can verify it by:

```bash
systemctl status milvus
```




## Upgrade


DEB upgrade:
------------

```bash
sudo dpkg -i milvus_<major>.<minor>-<release>-1_amd64.deb
```

## Uninstall


DEB uninstall:
--------------

```bash
sudo dpkg --purge milvus
rm -rf /default.etcd/member/
rm -rf /var/lib/milvus/*

```


## Kubernetes Image Installation
Install Load balancer
----------------------
Install a load balance and assign external ip pool (for example metallb)
This is required for external access to milvus and prometheus/grafana.
1) kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.8/config/manifests/metallb-native.yaml
2) Edit MetallbPool.yaml and set the assigned IP Pool, consult IT manager for free IPs:
```yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: first-pool
  namespace: metallb-system
spec:
  addresses:
  - 172.16.100.10-172.16.100.20## this is example - set to your own assigned ips

apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: l2adv
  namespace: metallb-system
spec:
  ipAddressPools:
  - first-pool
```    
3) kubectl create -f MetallbPool.taml

Install Milvus Operator:
-----------------
If the Milvus operator wasn’t installed yet – install the operator
https://milvus.io/docs/install_cluster-milvusoperator.md (how to install operator)
Steps used:
1) Install cert manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.5.3/cert-manager.yaml
Wait until pods are ready
2) Install milvus operator
kubectl apply -f https://raw.githubusercontent.com/zilliztech/milvus-operator/main/deploy/manifests/deployment.yaml
Wait until pods are ready


Install NVIDIA's operator(optional)
------------------
NVIDIA's gpu operator is required if you want to use gpu(s) in k8s clusters.
To install the NVIDIA's operator from rancher, you need to:
1) In Rancher UI add nvidia helm repo via Apps -> Repositories
    The repo is https://helm.ngc.nvidia.com/nvidia
2) In Rancher UI, go to: Apps & Marketplace → Charts → gpu-operator → Install
   note: latest gpu operator version 25.10.x had issues but 25.4.3 works.
2)  Set Namespace: Choose gpu-operator and check Create namespace.

3)  Scroll to the big YAML editor labeled Values (YAML).

Find or add this section under toolkit: (if not present, add it manually):
toolkit:
env:
- name: CONTAINERD_SOCKET
  value: /run/k3s/containerd/containerd.sock
4) Click Install
5) Make sure your GPUs are listed when running kubectl describe node <node with gpu>

Using more than one GPU per host
--------------------------------
You can increase the replica count of data node to the GPU count.
In order to use a different GPU from each datanode you should limit
the gpu resource to 1 for each datanode replica by adding:
resources:
    limits:
        nvidia.com/gpu: 1
Under the datanode component.
Please note that the host disk will become a bottleneck in such configuration and expect
long delays while writing or fetching index files.



Prepare Storage
-----------------

Remote Storage
--------------
The cr requires 2 storage classes: 
standard - used by etcd
standard-thin - used by minio and pulsar.
You need to provide this storage classes to work 

Local Storage
-------------
To work with local storage we'll use dynamic local storage provisioner
called local-path-provisioner:
1) Prepare a directory for the provisioner on each worker node:
Create a folder: /opt/local-path-provisioner/
Mount it on a local drive - use ext4 fs
update fstab to make the mount persistent.
 
2) Install local path provisioner:
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.28/deploy/local-path-storage.yaml

This will deploy a provisioner pod which allocates volumes from /opt/local-path-provisioner/
and will create a storage class called local-path
Set the storage class local-path as default.

kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

3) Edit milvus cluster yaml - replace standard and standard-thin with local-path storage class
Minio is configured with 10Ti volumes, if its not available, reduce it accordingly (for example 1Ti)
If 3 workers are not available - set replicaCount to 1 for all dependencies and reduce replicas to number of available
workers for all milvus node component (noot coordinators- for them its always 1)
from 3 to what's available. 
If you don't have 3 nodes for minio, set storage mode to standalone and set replicaCount to 1.
Also, when running pulsar only with one node, uncomment managedLedger* fields under broker section.

External Minio
--------------
When storage requirements are very high, for example for a 10b dataset, 110T are required just for minio, you might
have to use an external Minio.
This section will explain how to do it.
First you need to install an external Minio , you can use this guide: https://www.atlantic.net/dedicated-server-hosting/how-to-deploy-minio-on-ubuntu-24-04-an-open-source-object-storage-application/
You'll need to do some Minio configuration so install also mc on the Minio host. (linux minio client)
You'll need to edit the CR and switch to external Minio as explained here: https://milvus.io/docs/object_storage_operator.md
The section you need is: "Use Amazon Web Service (AWS) S3 as external object storage".
When creating the secret you need to:
1) replace my-release with your milvus name
2) Set the namespace kioxia for the secret

When specifying the storage type in the CR , you need to specify Minio.
You just need the secret and the cr editing they show, no need for other role/service configs.
After Minio is up, you need to configure stale_uploads_expiry=720h (this is required since we work with
long uploads during clustering compaction).

Run:

   mc alias set local http://127.0.0.1:9000 <minio user> <minio secret key>
   mc admin config set local api stale_uploads_expiry=720h

To make it a permanent config, edit the minio service:

sudo nano /etc/systemd/system/minio.service

add a line:
   Environment=MINIO_API_STALE_UPLOADS_EXPIRY=720h

In the environment area.

When done Run:

   sudo systemctl daemon-reload

indexSliceSize
---------------
We currently set indexSliceSize to 1 which leads to 128/1 threads which fetch data from minio.
Milvus uses default of 16 which leads to 8 threads.
   
Disable primary key index in RAM
--------------------------------
You can disable this index with queryNode.enablePKIndex: false configuration.
This can save RAM when loading collections and is only feasible if you dont require
Hybrid search and you don't do updates or deletes.

Multiple query nodes per Node
-----------------------------
To work with more than one query node on a host, you need to permanently configure
max aio ctx:
sudo sysctl -w fs.aio-max-nr=64k X num query nodes
and also add a line in /etc/sysctl.conf:
For example:
fs.aio-max-nr = 262144

Query Node/Index Node storage
------------------------------
In addition - querynode and indexnode requires fast storage under /var/lib/milvus/data,
So you need to mount it on a fast nvme drive/array.


Install Milvus CR
------------------
1) Create namespace kioxia if it doesnt exist.
kubectl create namespace kioxia

2) Enable clustering compaction
If you want to use the clustering compaction feature you need to set config->dataCoord->clusterin->enable
and config->dataCoord->clusterin->autoEnable to true in the milvus cluster cr.

3) Config DiskANN params:
   config->common:

```YAML
...
DiskIndex:
  MaxDegree: 56
  SearchListSize: 100
  PQCodeBudgetGBRatio: 0.125
  SearchCacheBudgetGBRatio: 0.125
  BeamWidthRatio: 4.0
...
```

4) Config AiSAQ params:
   config->common:

```YAML
...
AiSAQIndex:
  MaxDegree: 56
  SearchListSize: 100
  InlinePQ: -1
  Rearrange: true
  NumEntryPoints: 1
  PQCodeBudgetGBRatio: 0.125
  DiskPQCodeBudgetGBRatio: 0.25
  PQCacheSize: 0
  SearchCacheBudgetGBRatio: 0
  BeamWidthRatio: 4.0
...
```

5) After doing any editing required for local storage and/or clustering compaction as described above - do:
```bash
kubectl create -f milvus-cluster-<major>.<minor>.<release>.yaml
```

Upgrade / Modify:
-----------------

```bash
kubectl delete -f milvus-cluster-<major>.<minor>.<current_release>.yaml
kubectl create -f milvus-cluster-<major>.<minor>.<new_release>.yaml
```

Uninstall:
-----------------
1. Get the name of the deployed Milvus resource:<br/>
  Run the following command to list the deployed <b>milvus</b> instance:
  ```bash
  kubectl get milvus -n kioxia
  ```
  Example output:
  ```bash
  NAME      MODE      STATUS      UPDATED   AGE
  milvus3   cluster   Healthy   False     7d2h
  ```
  In this example, the name of the installed Milvus instance is <b>milvus3</b>.
2. Delete the milvus instance<br/> 
Use the name of the milvus resource (e.g., <b>milvus3</b>) from the output above to delete the instance:
```bash
kubectl delete milvus milvus3 -n kioxia
```
**Note: Deleting the Milvus instance using the above command does not remove its dependencies. If needed, you must delete them separately or follow the instructions below.**

3. Delete Milvus Along with Its Dependencies
If you wish to delete Milvus along with its dependencies (note that this will also delete the associated data), run the following commands **before** executing the delete command:
```bash
kubectl patch milvus milvus3 -n kioxia --type='merge' -p '{"spec":{"dependencies":{"etcd":{"inCluster":{"deletionPolicy": "Delete", "pvcDeletion": true}}}}}'
kubectl patch milvus milvus3 -n kioxia --type='merge' -p '{"spec":{"dependencies":{"storage":{"inCluster":{"deletionPolicy": "Delete", "pvcDeletion": true}}}}}'
kubectl patch milvus milvus3 -n kioxia --type='merge' -p '{"spec":{"dependencies":{"pulsar":{"inCluster":{"deletionPolicy": "Delete", "pvcDeletion": true}}}}}'
```
After applying these patches, proceed with the delete command:
```bash
  kubectl delete milvus milvus3 -n kioxia
```

Install prometheus and grafana
------------------------------
1) Install the install operator:
kubectl create namespace kumo-services
kubectl create -f ks-install-operator-v3.24-2264.yaml
Wait until operator is ready
Due to permmision issue: https://github.com/aws/eks-charts/issues/21
local-path storage class cannot be used for prometheus pod (it can be used for grafana and alert manager),
so by default- the promethus install will not persist data.
If it's important to persist prometheus data in your setup, provide a different storage class
and uncomment storage spec for prometheus with valid values.
If you can't provide a storage class - look for other options in:
https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/user-guides/storage.md
 
Install prometheus and grafana:
2) If you use local storage you need to replace standard and standard-thin storage classes with local-path.
Also edit the additional scrap config with your IP's if any, remove this section if it is not needed.
After editing perform:

kubectl create -f prometheus_local_rancher.yaml

3) Install milvus dashboard
Open grafana (user admin ,passw ksAdmin)
Go to Dashobards->New->Import - import the artifact dashboard.json

