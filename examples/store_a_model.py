"""
store_a_model
=============

This example script shows you how to train a basic model object and store it
using the provided storage class.
"""

import datetime
from sklearn.linear_model import LinearRegression
from viewser import Queryset,Column
from views_runs import ViewsRun, StepshiftedModels, DataPartitioner, ModelMetadata, Storage

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

metadata = ModelMetadata(
        author = "testuser",
        run_id = "testing",
        queryset_name = "views-runs-example-queryset",
        train_start = 1,
        train_end = 300,
        training_date = datetime.datetime.now())

#  We can store both the model object and the metadata at the same time

store.store("views-runs-example-run", run, metadata, overwrite = True)

# You can assert the equivalence of the stored metadata

assert metadata == store.fetch_metadata("views-runs-example-run")

# Here's how you could choose to store a new model object only if the existing
# object has the same metadata and steps

existing_is_equivalent = (
        store.exists("views-runs-example-run") and
        metadata == store.fetch_metadata("views-runs-example-run") and
        store.retrieve("views-runs-example-run").steps == run.steps
        )

if existing_is_equivalent:
    print("The existing stored data is equivalent, I won't bother overwriting it")
else:
    print("Overwriting store data, since it differs from what I got...")
    store.store("views-runs-example-run", run, metadata, overwrite = True)
