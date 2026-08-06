#!/bin/bash
#
# If static S3 credentials are provided, write them to the OpenSearch keystore,
# then start OpenSearch. Doing this at runtime (not image build time) avoids
# baking credentials into image layers. If static credentials are not provided,
# repository-s3 can fall back to the AWS default credential provider chain, such
# as an EC2 instance role.
#
# Static credential environment variables:
#   AWS_ACCESS_KEY_ID       AWS access key ID
#   AWS_SECRET_ACCESS_KEY   AWS secret access key
#   AWS_SESSION_TOKEN       STS session token (required for temporary credentials)
#
set -e

if [ -n "${AWS_ACCESS_KEY_ID}" ] && [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
    rm -f /usr/share/opensearch/config/opensearch.keystore
    /usr/share/opensearch/bin/opensearch-keystore create
    printf '%s' "${AWS_ACCESS_KEY_ID}"       | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.access_key
    printf '%s' "${AWS_SECRET_ACCESS_KEY}"   | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.secret_key
    if [ -n "${AWS_SESSION_TOKEN}" ]; then
        printf '%s' "${AWS_SESSION_TOKEN}"   | /usr/share/opensearch/bin/opensearch-keystore add --stdin s3.client.default.session_token
    fi
elif [ -n "${AWS_ACCESS_KEY_ID}" ] || [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together" >&2
    exit 1
else
    echo "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set; using the AWS default credential provider chain for S3" >&2
fi

exec /usr/share/opensearch/bin/opensearch
