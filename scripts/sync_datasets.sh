#!/bin/bash

# download datasets
aws s3 sync \
  s3://openproblems-data/resources/task_label_projection/datasets \
  resources/datasets \
  --delete
