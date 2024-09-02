#!/bin/bash

cat > /tmp/params.yaml << 'HERE'
input_states: s3://openproblems-data/resources/datasets/**/state.yaml
rename_keys: 'input:output_dataset'
output_state: "state.yaml"
settings: '{"output_train": "$id/train.h5ad", "output_test": "$id/test.h5ad"}'
publish_dir: s3://openproblems-data/resources/task_label_projection/datasets/
HERE

tw launch https://github.com/openproblems-bio/task_label_projection.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_label_projection,create_resources
