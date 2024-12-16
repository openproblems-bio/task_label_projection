cat(">> Loading dependencies\n")
library(anndata, warn.conflicts = FALSE)
requireNamespace("SingleR", quietly = TRUE)
library(Matrix, warn.conflicts = FALSE)

## VIASH START
par <- list(
  input_train = "resources_test/task_label_projection/cxg_immune_cell_atlas/train.h5ad",
  input_test = "resources_test/task_label_projection/cxg_immune_cell_atlas/test.h5ad",
  output = "output.h5ad"
)
meta <- list(
  name = "seurat_transferdata"
)
## VIASH END

cat(">> Load input data\n")
input_train <- read_h5ad(par$input_train)
input_test <- read_h5ad(par$input_test)

cat(">> Run method\n")
pred <- SingleR::SingleR(
  test = t(input_test$layers[["normalized"]]),
  ref = t(input_train$layers[["normalized"]]),
  labels = input_train$obs$label
)

cat(">> Create output data\n")
output <- anndata::AnnData(
  obs = data.frame(
    row.names = input_test$obs_names,
    label_pred = pred$labels
  ),
  uns = list(
    method_id = meta$name,
    dataset_id = input_test$uns[["dataset_id"]],
    normalization_id = input_test$uns[["normalization_id"]]
  ),
  shape = c(input_test$n_obs, 0L)
)

cat(">> Write output to file\n")
output$write_h5ad(par$output, compression = "gzip")
