
from sklearn.linear_model import LinearRegression
from viewser import Queryset, Column
from views_runs import operations
from views_runs import Storage, StepshiftedModels, DataPartitioner

(Queryset("production-run-example-queryset", "country_month")
        .with_column(Column("ged_dep", "ged2_cm", "ged_ns_best_sum_nokgi").transform.missing.fill())
        .with_column(Column("ged_indep", "ged2_cm", "ged_ns_best_sum_nokgi").transform.missing.fill())
        .publish())

store = Storage()

run, data = operations.retrain_or_retrieve(
        store              = store,
        partitioner        = DataPartitioner({"A": {"train": (1,100)}}),
        stepshifted_models = StepshiftedModels(LinearRegression(), [*range(1,13)], "ged_dep"),
        queryset_name      = "production-run-example-queryset",
        partition_name     = "A",
        storage_name       = "production-run-example",
        author_name        = "example")

print(run)
print(data)
