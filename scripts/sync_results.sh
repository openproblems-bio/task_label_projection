#!/bin/bash

# download results
aws s3 sync \
  s3://openproblems-data/resources/task_label_projection/ \
  resources \
  --exclude '*.h5ad' \
  --delete --dryrun

# UPLOAD RESULTS
# WARNING: USE WITH CARE!
aws s3 sync \
  --profile op \
  resources \
  s3://openproblems-data/resources/task_label_projection/ \
  --exclude '*.h5ad' \
  --delete --dryrun
