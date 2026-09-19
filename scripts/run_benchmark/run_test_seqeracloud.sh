#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

# write the parameters to file
cat > /tmp/params.yaml << 'HERE'
input_states: s3://openproblems-data/resources_test/task_label_projection/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test;input_solution:output_solution'
output_state: "state.yaml"
publish_dir: s3://openproblems-work/temp/task_label_projection/
HERE

tw launch https://github.com/openproblems-bio/task_label_projection.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 7gRyww9YNGb0c6BUBtLhDP \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_label_projection,test
