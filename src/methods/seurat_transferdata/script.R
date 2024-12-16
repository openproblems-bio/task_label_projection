cat(">> Loading dependencies\n")
library(Matrix, warn.conflicts = FALSE)
library(anndata, warn.conflicts = FALSE)
requireNamespace("Seurat", quietly = TRUE)
library(magrittr, warn.conflicts = FALSE)

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

packageVersion("Matrix")

cat(">> Load input data\n")
input_train <- read_h5ad(par$input_train)
input_test <- read_h5ad(par$input_test)

cat(">> Converting AnnData to Seurat\n")
anndataToSeurat <- function(adata) {
  # interpreted from https://github.com/satijalab/seurat/blob/v3.1.0/R/objects.R
  obj <-
    SeuratObject::CreateSeuratObject(
      counts = as(Matrix::t(adata$layers[["counts"]]), "CsparseMatrix")
    ) %>%
    SeuratObject::SetAssayData(
      layer = "data",
      new.data = as(Matrix::t(adata$layers[["normalized"]]), "CsparseMatrix")
    ) %>%
    SeuratObject::AddMetaData(
      adata$obs
    )

  # set hvg
  SeuratObject::VariableFeatures(obj) <- adata$var_names[adata$var[["hvg"]]]

  # set embedding
  # could add loadings and stdev
  X_pca <- adata$obsm[["X_pca"]]
  dimnames(X_pca) <- list(rownames(adata), paste0("PC_", seq_len(ncol(X_pca))))
  embed <- SeuratObject::CreateDimReducObject(
    embeddings = X_pca,
    key = "PC_"
  )
  obj[["pca"]] <- embed

  # return
  obj
}

obj_train <- anndataToSeurat(input_train)
obj_test <- anndataToSeurat(input_test)

cat(">> Find transfer anchors\n")
npcs <- ncol(obj_train[["pca"]])
anchors <- Seurat::FindTransferAnchors(
  reference = obj_train,
  query = obj_test,
  npcs = npcs,
  dims = seq_len(npcs),
  verbose = FALSE
)

cat(">> Predict on test data\n")
query <- Seurat::TransferData(
  anchorset = anchors,
  reference = obj_train,
  query = obj_test,
  refdata = list(labels = "label"),
  verbose = FALSE
)

cat(">> Create output data\n")
output <- anndata::AnnData(
  obs = data.frame(
    row.names = input_test$obs_names,
    label_pred = query$predicted.labels
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
