
import copy
import warnings
from typing import Any, List, Optional
from views_schema import models as schema
from views_storage.types import JsonSerializable
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

    def store(self, name: str, model: Any, metadata: Optional[schema.ModelMetadata] = None, overwrite: bool = False) -> None:
        """
        store
        =====

        parameters:
            name (str): Unique name associated with the model
            model (Any): A trained model object
            metadata (views_schema.ModelMetadata): An object containing metadata for the model.
            overwrite (bool) = False: Whether to overwrite the model if it already exists

        Store a model object and its associated metadata under a unique name.

        """

        self._model_object_store.write(name, model, overwrite = overwrite)

        if metadata:
            self.store_metadata(name, metadata, overwrite)

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

    def store_metadata(self, name: str, metadata: schema.ModelMetadata, overwrite: bool = False) -> None:
        """
        store_metadata
        ==============

        parameters:
            name (str)
            metadata (views_schema.ModelMetadata)
            overwrite (bool) = False

        Stores metadata for a model object. Can be used to update metadata for
        a model if overwrite is True.
        """
        if self._model_object_store.exists(name):
            self._metadata_store.write(name, metadata, overwrite = overwrite)
        else:
            raise KeyError(f"No model named {name}. Store a model object before storing metadata")

    def has_metadata(self, name: str) -> bool:
        """
        has_metadata
        ============

        parameters:
            name (str)
        returns:
            bool: Whether model has metadata or not

        Checks whether a model has metadata defined for it.
        """
        return self._metadata_store.exists(name)

    def fetch_metadata(self, name: str) -> schema.ModelMetadata:
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

    def exists(self, name: str, metadata: Optional[schema.ModelMetadata] = None) -> bool:
        """
        exists
        ======

        parameters:
            name (str): Name of model
            metadata (Optional[JsonSerializable]):
                If provided, checks if existing metadata is equivalent to this.

        returns:
            bool: Whether object exists or not

        Checks for the existence of a named model. If metadata is provided,
        also checks that existing metadata is equivalent to the one provided.
        """
        if metadata is not None:
            warnings.warn("The metadata parameter is deprecated for this function. Use .exists_with_metadata instead (readability)")
            return self.exists_with_metadata(name, metadata)
        else:
            return self._model_object_store.exists(name)

    def exists_with_metadata(self, name: str, metadata: schema.ModelMetadata) -> bool:
        """
        exists_with_metadata
        ====================

        parameters:
            name (str)
            metadata (views_schema.ModelMetadata)

        returns:
            bool

        Checks whether a model exists with this exact metadata specified.
        """

        exists = self.exists(name)
        if exists and self.has_metadata(name):
            existing_metadata = self.fetch_metadata(name)

            # Ignore training_date value when checking equivalence
            equivalent = copy.copy(metadata)
            equivalent.training_date = existing_metadata.training_date
            exists &= equivalent == existing_metadata
        return exists 

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
