import re
import sys
import traceback
from typing import List

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config import SHARDS_NUM, RED, RESET, MAGENTA, CYAN, VECTOR_FIELD
from logs import LOGGER
from common import read_fvecs, get_collection_dim, get_index_metric_type

additional_types = {'text': range(1, 62401), 'varchar': range(1, 62401)}


def format_bytes(num_bytes):
    """
    Converts a number of bytes into a human-readable string in B, KB, MB, or GB.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < (1024 ** 2):
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < (1024 ** 3):
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"

def get_additional_type_field(field_type):
    valid_field = False
    for base_type, valid_range in additional_types.items():
        if field_type.startswith(base_type):
            valid_field = True
            size = field_type[len(base_type):]
            if not size.isdigit():
                raise ValueError(f"Invalid field type: {field_type}. Size must be a valid integer.\n")
            if int(size) not in valid_range:
                raise ValueError(f"Invalid varchar field size: {size}.")
            break
    if not valid_field:
        raise ValueError(f"Invalid field type: {field_type}.\n")
    return True


def generate_additional_fields(additional_fields: List[str], additional_fields_types: str, dim: int):
    # Dictionary to map field types to DataType and handle dynamic type support
    field_type_mapping = {
        "int": DataType.INT64,
        "vector": DataType.FLOAT_VECTOR,
        "varchar": DataType.VARCHAR,
        "text": DataType.TEXT  # New data type "text" can be added easily here
    }

    schema_fields = []
    # print('additional_fields: {}'.format(additional_fields))
    # print('additional_fields_types: {}'.format(additional_fields_types))
    if isinstance(additional_fields, str):
        additional_fields_list = additional_fields.split(",")
    else:
        additional_fields_list = additional_fields
    additional_types_list = additional_fields_types.split(",")
    for index, name in enumerate(additional_fields_list):
        # print('name: {}'.format(name))
        field_type = additional_types_list[index]
        dtype = DataType.INT64 if field_type == 'int' else DataType.VARCHAR if field_type.startswith(
            'varchar') else DataType.FLOAT_VECTOR

        # Match for types like varchar6400 or text100
        match = re.match(r"([a-zA-Z]+)(\d+)?", field_type)
        if match:
            base_type, max_length = match.groups()
            dtype = field_type_mapping.get(base_type, None)

            if dtype:
                if dtype in [DataType.VARCHAR, DataType.TEXT] and max_length:
                    # Use max_length for varchar or text
                    field = FieldSchema(name=name, dtype=dtype, max_length=int(max_length))
                elif dtype == DataType.FLOAT_VECTOR:
                    field = FieldSchema(name=name, dtype=dtype, dim=dim)
                else:
                    # If it's a simple type like int, no additional arguments
                    field = FieldSchema(name=name, dtype=dtype)
            else:
                raise ValueError(f"Unsupported field type: {base_type}")
        else:
            raise ValueError(f"Invalid field type format: {field_type}")

        schema_fields.append(field)

    return schema_fields


class MilvusHelper:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """

    def __init__(self, milvus_host, milvus_port):
        try:
            self.collection = None
            connections.connect(alias='default', host=milvus_host, port=milvus_port)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{milvus_host} and PORT:{milvus_port}")
            self.host = milvus_host
            self.port = milvus_port
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(f"There is no collection named:{collection_name}")
        except Exception as e:
            LOGGER.error(f"ERROR: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name, vector_dimension, shards_num, additional_fields=None,
                          additional_fields_types=None, is_clustering_key=True):
        # Create milvus collection if not exists
        try:
            if not self.has_collection(collection_name):
                field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True,
                                     auto_id=False)
                if is_clustering_key:
                    field2 = FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, descrition="float vector",
                                         dim=vector_dimension, is_primary=False, is_clustering_key=True)
                else:
                    field2 = FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, descrition="float vector",
                                         dim=vector_dimension, is_primary=False)
                schema_fields = [field1, field2]
                if additional_fields is not None:
                    additional_schema_fields = generate_additional_fields(additional_fields.split(","),
                                                                          additional_fields_types, vector_dimension)
                    schema_fields.extend(additional_schema_fields)
                schema = CollectionSchema(fields=schema_fields, description="collection description")
                self.collection = Collection(name=collection_name, schema=schema, shards_num=shards_num,
                                             index_file_size=81920)
                LOGGER.debug("Create Milvus collection: {}".format(collection_name))
                return True
            else:
                self.collection = Collection(collection_name)
                LOGGER.debug(f"collection {collection_name} exists")
                return True
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def insert_vectors(self, collection_name, vectors_file_path):
        vectors = []
        count = 0
        print(f"About to read vectors from the {vectors_file_path} file and insert to the collection\n")
        for vector in read_fvecs(vectors_file_path):
            vectors.append(vector)
            count += 1
            if len(vectors) >= 16000:
                self.insert(vectors)
                print(f"\r{(count):,} vectors inserted       ", end='', flush=True)
                vectors = []
        if len(vectors) > 0:
            self.insert(collection_name=collection_name, vectors=vectors)
            print(f"\r{(count):,} vectors inserted       ")
        return 0

    def insert_data(self, collection_name, data):
        # Batch insert vectors to milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.insert(data)
            print('Insert vectors to Milvus in collection: {} with {} rows'.format(collection_name, len(data[0])))
            LOGGER.debug(f"Insert vectors to Milvus in collection: {collection_name} with {len(data[0])} rows")
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            traceback.print_exc()
            print(e)
            sys.exit(1)

    def insert(self, collection_name, vectors, ids=None, shards_num=SHARDS_NUM):
        # Batch insert vectors to milvus collection
        try:
            if not self.has_collection(collection_name):
                dim = len(vectors[0])
                self.create_collection(collection_name, dim, shards_num)
            else:
                self.validate_and_get_vector_field(collection_name)
            vlen = -1
            cnt = 0
            for v in vectors:
                if vlen == -1 or len(v) == vlen:
                    cnt += 1
                else:
                    print(f"{cnt} vectors of {vlen} elements")
                    cnt = 1
                vlen = len(v)
            print(f"{cnt} vectors of {vlen} elements")
            cdim = get_collection_dim(self.collection)
            if cdim:
                if cdim != vlen:
                    print(f"{RED}ERROR: {RESET}Input vector dim: {MAGENTA}{vlen}{RESET}, "
                          f"but the {CYAN}{collection_name}{RESET} collection dim is {MAGENTA}{cdim}{RESET}!")
                    sys.exit(0)
            else:
                print(f"Warning: Failed to get dimension for collection '{self.collection.name}'. "
                      f"Apparently, the collection does not exist. New collection will be created.")
            self.collection.insert([ids, vectors])
            LOGGER.debug(
                f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            traceback.print_exc()
            print(e)
            sys.exit(1)

    def create_index(self, collection_name, index_params):
        v_field = self.validate_and_get_vector_field(collection_name)
        try:
            self.set_collection(collection_name)
            status = self.collection.create_index(field_name=v_field, index_params=index_params)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{index_params}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        # Delete Milvus collection
        try:
            utility.drop_collection(collection_name)
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error("Failed to drop collection: {}".format(e))
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k, search_params):
        # Search vector in milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.flush()
            v_field = self.validate_and_get_vector_field(collection_name)
            if self.collection.is_empty:
                print(f"Collection {collection_name} is empty")
                sys.exit(0)
            if len(vectors[0]) != get_collection_dim(self.collection):
                print(f"Collection dimension ({get_collection_dim(self.collection)}) and "
                      f"query vectors dimension ({len(vectors[0])}) must equal")
                sys.exit(0)
            search_params["metric_type"] = get_index_metric_type(self.collection)
            # res = self.collection.search(vectors[:2], anns_field=v_field, param=search_params, limit=top_k)
            res = self.collection.search(vectors, anns_field=v_field, param=search_params, limit=top_k)
            LOGGER.debug("Successfully search in collection: {}".format(res))
            return res
        except Exception as e:
            traceback.print_exc()
            LOGGER.error("Failed to search vectors in Milvus: {}".format(e))
            sys.exit(1)

    def count(self, collection_name):
        # Get the number of milvus collection
        try:
            self.set_collection(collection_name)
            # Retrieve ALL fields (vector + metadata)
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["count(*)"],
                consistency_level="Strong"
            )

            self.collection.flush()
            num = results[0]
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)

    def get_index_params(self, collection_name):
        # get index info
        self.set_collection(collection_name)
        return [index.params for index in self.collection.indexes]

    def create_partition(self, collection_name, partition_name):
        # create a partition for Milvus
        self.set_collection(collection_name)
        if self.collection.has_partition(partition_name):
            return f"This partition {partition_name} exists"
        else:
            partition = self.collection.create_partition(partition_name)
            return partition

    def delete_index(self, collection_name):
        self.set_collection(collection_name)
        print(f"==> HAS INDEX 1: {self.collection.has_index()}")
        self.collection.drop_index()
        print(f"==> HAS INDEX 2: {self.collection.has_index()}")

    def load_data(self, collection_name, exclude_fields=""):
        # load data from disk to
        try:
            self.set_collection(collection_name)
            if exclude_fields:
                exclude = exclude_fields.split(",")
                fields = [field.name for field in self.collection.schema.fields if field.name not in exclude]
                print(f"ABOUT TO LOAD FIELDS: {fields}")
                self.collection.load(load_fields=fields)
            else:
                self.collection.load()
        except Exception as e:
            LOGGER.error(f"Failed load data: {e}")
            sys.exit(1)

    def list_collection(self):
        # List all collections.
        return utility.list_collections()

    def get_loading_progress(self, collection_name):
        # Query the progress of loading.
        return utility.loading_progress(collection_name)

    def get_index_progress(self, collection_name):
        # Query the progress of index building.
        return utility.index_building_progress(collection_name)

    def release_data(self, collection_name):
        # release collection data from memory
        try:
            self.set_collection(collection_name)
            self.collection.release()
        except Exception as e:
            LOGGER.error(f"Failed release data: {e}")
            sys.exit(1)

    def calculate_distance(self, vectors_left, vectors_right):
        # Calculate distance between two vector arrays.
        return utility.calc_distance(vectors_left, vectors_right)

    def compact(self, collection_name):
        try:
            self.set_collection(collection_name)
            self.collection.compact()
        except Exception as e:
            LOGGER.error(f"Failed compact data: {e}")
            sys.exit(1)

    def validate_and_get_vector_field(self, collection_name):
        if not self.collection:
            self.set_collection(collection_name)
        for field in self.collection.schema.fields:
            if 'dim' in field.params:
                return field.name
        print(f"{RED}Error:{RESET} Failed to find a field designated to hold vector data")
        exit(1)

    def get_collection_fields(self, collection_name):
        self.set_collection(collection_name)
        fields_names = []
        for field in self.collection.schema.fields:
            fields_names.append(field.name)
        return fields_names

    def print_collection_metadata(self, collection_name):
        self.set_collection(collection_name)
        # Print collection description
        print(f"Collection name: {CYAN}{self.collection.name}{RESET}")
        print(f"Collection description: {CYAN}{self.collection.description}{RESET}")
        print(f"Collection is_empty: {CYAN}{self.collection.is_empty}{RESET}")
        print(f"Collection num_entities: {CYAN}{self.collection.num_entities}{RESET}")

        # Print fields
        print("\nFields:")
        for field in self.collection.schema.fields:
            print(f"  - Field name: {CYAN}{field.name}{RESET}")
            print(f"    Data type: {CYAN}{field.dtype}{RESET}")
            print(f"    Description: {CYAN}{field.description}{RESET}")
            print(f"    Is primary key: {CYAN}{field.is_primary}{RESET}")
            print(f"    Is auto id: {CYAN}{field.is_auto_id}{RESET}")
            print(f"    Index params: {CYAN}{field.index_params}{RESET}")
            print(f"    Dimension: {CYAN}{field.dim}{RESET}")
            print(f"    Is clustering key: {CYAN}{field.is_clustering_key}{RESET}")
        if self.collection.indexes:
            print(self.get_index_params(collection_name))
        else:
            print("  No indexes found for this collection.")

    def rename_collection(self, collection_name, new_name):
        self.set_collection(collection_name)
        utility.rename_collection(collection_name, new_name)

    def print_segments(self, collection_name, sum_by):
        segment_info = utility.get_query_segment_info(collection_name)
        total_memory = 0
        total_count = 0
        nodes = {}
        num_segments = len(segment_info)
        for index, segment in enumerate(segment_info):
            curr_node_id = segment.nodeIds[0]
            total_memory += segment.mem_size
            total_count += segment.num_rows
            if curr_node_id in nodes:
                exists_node = nodes[curr_node_id]
                exists_node["Memory"] += segment.mem_size
                exists_node["Vectors"] += segment.num_rows
                exists_node["Segments"] += 1
            else:
                new_node = {"Node": curr_node_id, "Segments": 1, "Memory": segment.mem_size, "Vectors": segment.num_rows}
                nodes[curr_node_id] = new_node
            if sum_by is None:
                print(str(segment))

        if sum_by == 'collection':
            total_memory_str = format_bytes(total_memory)
            print(f"Collection Total Index Memory: {CYAN}{total_memory_str}{RESET}")
            print(f"Collection Total Vectors Count: {CYAN}{total_count}{RESET}")
            print(f"Collection Segments Count: {CYAN}{num_segments}{RESET}")

        if sum_by == 'node':
            print('Collection By Nodes:')
            for _, node in nodes.items():
                node_memory_str = format_bytes(node['Memory'])
                print(f"Node ID: {CYAN}{node['Node']}{RESET}, Segments: {CYAN}{node['Segments']}{RESET}, Memory: {CYAN}{node_memory_str}{RESET}, Vectors: {CYAN}{node['Vectors']}{RESET}")
