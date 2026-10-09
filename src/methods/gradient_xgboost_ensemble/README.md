# Gradient XGBoost ensemble

This adapts ten frozen parameter recipes recovered from the Gradient kidney
label-projection campaign: top 3,000 supplied HVGs plus AQP3, normalized sparse
expression, and majority voting. Only training-cell labels are used at inference
fit time. It accepts the OpenProblems training/test API and emits predictions as
AnnData. AQP3 is resolved through feature names as well as the gene index.

The recipes were selected after repeated feedback on the original test set
(263 attempts); they were subsequently reused without retuning in donor-fold
validation. The exact original ensemble membership is a source-supported
reconstruction and its original tie rule is unverified. This implementation
uses lexicographically smallest labels on tied votes, as in the frozen donor
validation. Dependency versions match the recovered producer Dockerfile.
Supplied cohort-wide HVG scores and normalization are reused; this component
does not fit preprocessing separately within folds.

This contribution does not claim that the campaign score is an independently
validated OpenProblems leaderboard result. OpenProblems must evaluate this
adaptation on its own datasets.

```sh
viash test src/methods/gradient_xgboost_ensemble/config.vsh.yaml
```
