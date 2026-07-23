# Opensearch + cuVS multi-node benchmarking setup

This guide sets up the three-node version of the OpenSearch GPU benchmark stack.

```text
client node       -->   runs the benchmark submitter
opensearch node   -->   runs OpenSearch
builder GPU node  -->   runs the remote index build service
```

The examples use `us-west-2`, but that region is not required. Choose any AWS region where your needed instance types, GPU AMI, and quotas are available, then keep all resources in that same region.

The goal is to keep the deployment simple while separating the three major components onto their own EC2 instances. Docker Compose still runs locally on each instance; it does not create a cross-host network. Cross-node communication uses EC2 private DNS names or private IPv4 addresses.

## 1. Choose one AWS region

In the AWS Console, set the region selector to your chosen region. The examples below use:

```text
US West (Oregon) us-west-2
```

Use this same region for S3, EC2, security groups, and the instances' IAM roles. The EC2 instances and security groups should also be in the same VPC so private DNS, private IPs, and security-group source rules work as expected.

When running commands, set both AWS region variables from one value:

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_REGION="$AWS_DEFAULT_REGION"
```

`AWS_REGION` is used when registering the OpenSearch S3 repository region. `AWS_DEFAULT_REGION` is used by AWS CLI and boto-style tooling. Setting both from the same value avoids accidental mismatches.

## 2. Create the S3 bucket

Go to **S3 > Create bucket**.

Use:

```text
Bucket name: globally unique name
Region: your chosen AWS region, for example US West (Oregon) us-west-2
Object Ownership: ACLs disabled
Block Public Access: block all public access
Encryption: SSE-S3 is fine
Versioning: optional
```

This bucket stores the remote-build staging objects, including vectors and
generated index artifacts. Benchmark datasets and result plots stay on the
client node under `DATASET_PATH`.

## 3. Create an IAM policy for S3

Go to **IAM > Policies > Create policy > JSON**.

Use this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::opensearch-cuvs-bench"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::opensearch-cuvs-bench/*"
    }
  ]
}
```

Name it:

```text
opensearch-cuvs-bench-s3-policy
```

The delete permission lets the snapshot repository and staging workflow clean up
temporary objects when needed.

## 4. Create the EC2 IAM role

Go to **IAM > Roles > Create role**.

Choose:

```text
Trusted entity: AWS service
Use case: EC2
```

Attach:

```text
opensearch-cuvs-bench-s3-policy
AmazonSSMManagedInstanceCore
```

Name it:

```text
opensearch-cuvs-bench-ec2-role
```

This role gives the instances refreshable S3 credentials and enables Session Manager access. Prefer this over fixed `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` values.

## 5. Create security groups

Go to **EC2 > Security Groups > Create security group**.

Create these three security groups in the same VPC:

```text
sg-cuvs-client
sg-cuvs-opensearch
sg-cuvs-builder
```

Inbound rules for `sg-cuvs-client`:

```text
No inbound rules needed
```

Inbound rules for `sg-cuvs-opensearch`:

```text
TCP 9200 from sg-cuvs-client
TCP 9200 from sg-cuvs-builder
```

Inbound rules for `sg-cuvs-builder`:

```text
TCP 1025 from sg-cuvs-opensearch
TCP 1025 from sg-cuvs-client
```

Leave outbound as the default `allow all`. Use Session Manager instead of SSH if possible. If you need SSH, add TCP `22` only from your own IP.

## 6. Launch the OpenSearch node

Go to **EC2 > Instances > Launch instance**.

Use:

```text
Name: opensearch-cuvs-db
AMI: Ubuntu 22.04 or Amazon Linux 2023
Instance type: r7i.xlarge, r7i.2xlarge, or m7i.2xlarge
Security group: sg-cuvs-opensearch
IAM role: opensearch-cuvs-bench-ec2-role
Storage: 100+ GiB gp3, larger if needed
```

Under **Advanced details > Metadata options**, use:

```text
IMDS endpoint: Enabled
IMDSv2: Required
Hop limit: 2
```

After launch, copy the instance's **Private IPv4 DNS** or **Private IPv4 address**. You will use it as `OPENSEARCH_URL`.

## 7. Launch the GPU builder node

Launch a second EC2 instance:

```text
Name: opensearch-cuvs-builder
AMI: AWS Deep Learning Base AMI with CUDA, Ubuntu 22.04
Instance type: g5.xlarge or g6.xlarge
Security group: sg-cuvs-builder
IAM role: opensearch-cuvs-bench-ec2-role
Storage: 100+ GiB gp3
```

Use the same metadata options:

```text
IMDS endpoint: Enabled
IMDSv2: Required
Hop limit: 2
```

After launch, copy the instance's **Private IPv4 DNS** or **Private IPv4 address**. You will use it as `REMOTE_INDEX_BUILDER_URL`.

## 8. Launch the client node

Launch the benchmark submitter instance:

```text
Name: opensearch-cuvs-client
AMI: Ubuntu 22.04 or Amazon Linux 2023
Instance type: c7i.xlarge or m7i.xlarge
Security group: sg-cuvs-client
IAM role: opensearch-cuvs-bench-ec2-role
Storage: 50-100 GiB gp3
```

Use the same metadata options:

```text
IMDS endpoint: Enabled
IMDSv2: Required
Hop limit: 2
```

## 9. Install Docker and Compose on each node

Connect to each instance with **EC2 > Instances > Connect > Session Manager**.

Install Docker and Docker Compose if they are not already installed. Then verify:

```bash
docker compose version
```

On the GPU builder node, also verify GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

If you want to avoid `sudo docker`, add your login user to the `docker` group and reconnect:

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker "$(whoami)"
```

## 10. Copy the deployment checkout to each node

On all three nodes, clone or copy the `deploy-opensearch-tmp` branch of the
`jrbourbeau/cuvs` fork:

```bash
git clone --branch deploy-opensearch-tmp https://github.com/jrbourbeau/cuvs.git
cd cuvs
```

Make sure this file is present:

```text
docker-compose.multinode.yml
```

## 11. Start OpenSearch

On the OpenSearch node:

```bash
export S3_BUCKET=opensearch-cuvs-bench
export AWS_DEFAULT_REGION=us-west-2
export AWS_REGION="$AWS_DEFAULT_REGION"
docker compose -f docker-compose.multinode.yml --profile opensearch up -d
```

Verify:

```bash
curl http://localhost:9200
```

## 12. Start the remote index builder

On the GPU builder node:

```bash
export S3_BUCKET=opensearch-cuvs-bench
export AWS_DEFAULT_REGION=us-west-2
export AWS_REGION="$AWS_DEFAULT_REGION"
docker compose -f docker-compose.multinode.yml --profile builder up -d
```

Verify the container is running:

```bash
docker ps
```

From the OpenSearch node, verify that the builder is reachable. OpenSearch is
the service that calls the remote builder:

```bash
python3 -c 'import socket; socket.create_connection(("BUILDER_PRIVATE_DNS_OR_IP", 1025), 5).close(); print("builder reachable")'
```

From the client node, run the same check. The benchmark container also waits
for the builder before starting a remote-build run:

```bash
python3 -c 'import socket; socket.create_connection(("BUILDER_PRIVATE_DNS_OR_IP", 1025), 5).close(); print("builder reachable")'
```

## 13. Configure OpenSearch from the client node

On the client node:

```bash
export OPENSEARCH_HOST=OPENSEARCH_PRIVATE_DNS_OR_IP
export OPENSEARCH_URL=http://${OPENSEARCH_HOST}:9200
export REMOTE_INDEX_BUILDER_URL=http://BUILDER_PRIVATE_DNS_OR_IP:1025
export S3_BUCKET=opensearch-cuvs-bench
export AWS_DEFAULT_REGION=us-west-2
export AWS_REGION="$AWS_DEFAULT_REGION"
export S3_PREFIX=knn-indexes
export REMOTE_VECTOR_REPOSITORY=vector-repo
```

Then run the one-shot configure profile:

```bash
docker compose -f docker-compose.multinode.yml --profile configure run --rm configure-remote-index-build
```

This registers the S3-backed remote vector repository and tells OpenSearch where the remote index builder service lives.

## 14. Run the benchmark

Still on the client node:

```bash
export REMOTE_INDEX_BUILD=true
export DATASET_PATH="$(pwd)/opensearch-cuvs-datasets"
export DATASET=sift-128-euclidean
export BENCH_GROUPS=test
export K=10
export BATCH_SIZE=
export BUILD_BATCH_SIZE=

mkdir -p "${DATASET_PATH}"

docker compose -f docker-compose.multinode.yml --profile client build bench
docker compose -f docker-compose.multinode.yml --profile client run --rm bench
```

## 15. Debug checklist

From the client node, these should all work:

```bash
curl "$OPENSEARCH_URL"
python3 -c 'import os, socket; from urllib.parse import urlparse; url = urlparse(os.environ["REMOTE_INDEX_BUILDER_URL"]); socket.create_connection((url.hostname, url.port or 1025), 5).close(); print("builder reachable")'
aws s3 ls s3://$S3_BUCKET --region "$AWS_DEFAULT_REGION"
```

If S3 fails, check the IAM role and S3 policy. If OpenSearch or builder
connectivity fails, check private DNS/IP values and security group source rules.

## Notes

- Keep the S3 bucket and EC2 resources in the same AWS region. The examples use `us-west-2`, but the setup is not region-specific.
- Within that region, keep all three instances and security groups in the same VPC. Prefer the same Availability Zone for the first benchmark run.
- Use private DNS or private IPs, not public IPs, for cross-node service traffic.
- Docker Compose networks are local to one host; Compose service names do not resolve across EC2 instances.
- The EC2 IAM role credentials rotate automatically. Avoid freezing temporary credentials into `.env` unless the application absolutely requires literal `AWS_*` variables.
