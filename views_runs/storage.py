
from typing import Any
from viewser.storage import model_object

_model_object_store = model_object.ModelObjectStorage()
_metadata_store     = model_object.ModelMetadataStorage()

def store(name: str, model: Any, metadata: Any):
    _model_object_store.write(name, model)
    _metadata_store.write(name, metadata)

def retrieve(name: str):
    return _model_object_store.read(name)

def list():
    return _model_object_store.list()

def fetch_metadata(name: str):
    return _metadata_store.read(name)
