
from sklearn.linear_model import LinearRegression
from viewser import Queryset, Column, fetch
from views_runs import Storage, StepshiftedModels, DataPartitioner
from views_runs.run_result import RunResult

(Queryset("production-run-example-queryset", "country_month")
        .with_column(Column("ged_dep", "ged2_cm", "ged_ns_best_sum_nokgi").transform.missing.fill())
        .with_column(Column("ged_indep", "ged2_cm", "ged_ns_best_sum_nokgi").transform.missing.fill())
        .publish())

store = Storage()
data = fetch("production-run-example-queryset")

result = RunResult.retrain_or_retrieve(
        store              = store,
        partitioner        = DataPartitioner({"A": {"train": (1,100)}}),
        stepshifted_models = StepshiftedModels(LinearRegression(), [*range(1,13)], "ged_dep"),
        dataset            = data,
        queryset_name      = "production-run-example-queryset",
        partition_name     = "A",
        storage_name       = "production-run-example",
        author_name        = "example")

print(result.retrained)
print(result.data)

result = RunResult.retrain_or_retrieve(
        store              = store,
        partitioner        = DataPartitioner({"A": {"train": (1,100)}}),
        stepshifted_models = StepshiftedModels(LinearRegression(), [*range(1,13)], "ged_dep"),
        dataset            = data,
        queryset_name      = "production-run-example-queryset",
        partition_name     = "A",
        storage_name       = "production-run-example",
        author_name        = "example",
        retrain            = True)

print(result.retrained)
print(result.data)
