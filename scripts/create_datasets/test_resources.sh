#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

RAW_DATA=resources_test/common
DATASET_DIR=resources_test/task_label_projection

mkdir -p $DATASET_DIR

# process dataset
echo Running process_dataset
nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  --publish_dir "$DATASET_DIR" \
  --id "cxg_immune_cell_atlas" \
  --input "$RAW_DATA/cxg_immune_cell_atlas/dataset.h5ad" \
  --output_train '$id/train.h5ad' \
  --output_test '$id/test.h5ad' \
  --output_solution '$id/solution.h5ad' \
  --output_state '$id/state.yaml' \
  -c common/nextflow_helpers/labels_ci.config

# run one method
viash run src/methods/knn/config.vsh.yaml -- \
    --input_train $DATASET_DIR/cxg_immune_cell_atlas/train.h5ad \
    --input_test $DATASET_DIR/cxg_immune_cell_atlas/test.h5ad \
    --output $DATASET_DIR/cxg_immune_cell_atlas/prediction.h5ad

# run one metric
viash run src/metrics/accuracy/config.vsh.yaml -- \
    --input_prediction $DATASET_DIR/cxg_immune_cell_atlas/prediction.h5ad \
    --input_solution $DATASET_DIR/cxg_immune_cell_atlas/solution.h5ad \
    --output $DATASET_DIR/cxg_immune_cell_atlas/score.h5ad

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile op \
  "$DATASET_DIR" s3://openproblems-data/resources_test/task_label_projection \
  --delete --dryrun
