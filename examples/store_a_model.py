"""
store_a_model
=============

This example script shows you how to train a basic model object and store it
using the provided storage class.
"""

from sklearn.linear_model import LinearRegression
from viewser import Queryset,Column
from views_runs import ViewsRun, StepshiftedModels, DataPartitioner, Storage

# We instantiate the store, which we'll use to save our model object
store = Storage()

# Make a basic dataset, and duplicate the ged column so it can be used as both dep. and indep in the model.
data = (Queryset("views-runs-example-queryset", "country_month")
    .with_column(Column("ged_dependent", "ged2_cm", "ged_ns_best_sum_nokgi").transform.missing.fill())
    .publish().fetch())
data["ged_independent"] = data["ged_dependent"]

# Make and fit a run object

run = ViewsRun(
        DataPartitioner({"A":{"train":(1,300), "test": (301,400)}}),
        StepshiftedModels(LinearRegression(), [*range(1,13)], "ged_dependent"))
run.fit("A", "train", data)

# We want to add metadata for our object

metadata = run.create_model_metadata(
        author = "testuser",
        queryset_name = "views-runs-example-queryset",
        training_partition_name = "A",
        )

#  We can store both the model object and the metadata at the same time

store.store("views-runs-example-run", run, metadata, overwrite = True)

# You can assert the equivalence of the stored metadata

assert metadata == store.fetch_metadata("views-runs-example-run")

# Here's how you could choose to store a new model object only if the existing
# object has the same metadata and steps

if store.exists_with_metadata("views-runs-example-run", metadata):
    print("The existing stored data is equivalent, I won't bother overwriting it")
else:
    print("Overwriting store data, since it differs from what I got...")
    store.store("views-runs-example-run", run, metadata, overwrite = True)
