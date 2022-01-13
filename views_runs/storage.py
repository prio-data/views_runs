
import warnings
from typing import Any, List
from views_storage.serializers.json import JsonSerializable
from viewser.storage import model_object

class Storage():
    """
    Storage
    =======

    A storage driver which can be used to store and retrieve (model) objects
    and associated metadata. Works as a key-value store, with each (model)
    object being associated with a unique name, which is then used to fetch
    both the object and its metadata.

    """

    def __init__(self):
        self._model_object_store = model_object.ModelObjectStorage()
        self._metadata_store = model_object.ModelMetadataStorage()

    def store(self, name: str, model: Any, metadata: JsonSerializable = {}, overwrite: bool = False) -> None:
        """
        store
        =====

        parameters:
            name (str): Unique name associated with the model
            model (Any): A trained model object
            metadata (JsonSerializable): A dictionary containing metadata
            overwrite (bool) = False: Whether to overwrite the model if it already exists

        Store a model object and its associated metadata under a unique name.

        """
        self._model_object_store.write(name, model, overwrite = overwrite)
        self._metadata_store.write(name, metadata, overwrite = overwrite)

    def retrieve(self, name: str) -> Any:
        """
        retrieve
        ========

        parameters:
            name (str): Name of the model to fetch

        returns:
            Any: A previously stored model object.

        Retrieve a stored model object.
        """
        return self._model_object_store.read(name)

    def fetch_metadata(self, name: str) -> JsonSerializable :
        """
        retrieve
        ========

        parameters:
            name (str): Name of the model to fetch metadata for

        returns:
            JsonSerializable: Metadata for model 

        Retrieve previously stored metadata for a model.
        """
        return self._metadata_store.read(name)

    def list(self) -> List[str]:
        """
        list
        ====

        returns:
            List[str]

        Return a list of available model names.
        """
        return self._model_object_store.list()

def store(name: str, model: Any, metadata: Any):
    warnings.warn("This function is deprecated. Use the views_runs.Storage and its methods instead")
    storage = Storage()
    storage.store(name, model, metadata)

def retrieve(name: str):
    warnings.warn("This function is deprecated. Use the views_runs.Storage and its methods instead")
    storage = Storage()
    return storage.retrieve(name)

def fetch_metadata(name: str):
    warnings.warn("This function is deprecated. Use the views_runs.Storage and its methods instead")
    storage = Storage()
    return storage.fetch_metadata(name)

def list():
    warnings.warn("This function is deprecated. Use the views_runs.Storage and its methods instead")
    storage = Storage()
    return storage.list()
