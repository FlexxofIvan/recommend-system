from pathlib import Path

import numpy as np
import polars as pl
import tritonclient.grpc as grpcclient
import yaml
from sentence_transformers import SentenceTransformer

from rec_sys.dataset_modules.cols_data.vector_data_utils import vectorize_df
from rec_sys.dataset_modules.graph_data.create_graphs import vec_data_to_graph
from rec_sys.dataset_modules.graph_data.graph_dataset_constants import (
    ASIN,
    EDGE_INDEX,
    FEAT_PRODUCT_INFO,
    FEAT_PRODUCT_NAME,
    PRODUCT_INFERENCE_NODE,
    USER_EMB,
    USER_NODE,
    USER_REL_PRODUCT,
)
from rec_sys.vector_database.vectorbase_utils import load_index, search_vector


class GraphInferenceClient:
    def __init__(
        self,
        data_config_path: str,
        vector_db_config_path: str,
        triton_url: str = "localhost:8001",
    ):
        # ---- load config ----
        with open(data_config_path, "r", encoding="utf-8") as f:
            self.data_cfg = yaml.safe_load(f)

        with open(vector_db_config_path, "r", encoding="utf-8") as f:
            self.db_cfg = yaml.safe_load(f)

        self.model_name = self.data_cfg["dataset"]["model_name"]
        self.words_fields = self.data_cfg["dataset"]["words_fields"]
        self.vector_size = self.db_cfg["dim"]

        self.sent_model = SentenceTransformer(self.model_name)
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)

        self.index_path = Path(self.db_cfg["index_path"])
        self.mapping_path = Path(self.db_cfg["mapping_path"])
        self.top_k = self.db_cfg["top_k"]

        self.index, self.prod_idxs = load_index(
            self.index_path, self.mapping_path, dim=self.vector_size
        )

    def preprocess(self, reviews: list) -> pl.DataFrame:
        """
        reviews: List[Review]
        """
        df = pl.DataFrame([r.dict() for r in reviews])

        vectorized_df = vectorize_df(
            df,
            self.sent_model.encode,
            self.words_fields,
            self.vector_size,
        )
        graphs = vec_data_to_graph(vectorized_df)
        return graphs

    @staticmethod
    def prepare_triton_inputs(graph: dict):
        """
        Подготавливает входы для Triton, используя константы.
        graph — HeteroData-like словарь
        """
        inputs = {
            "user_features": np.asarray(graph[USER_NODE][USER_EMB], dtype=np.float32),
            "product_info_features": np.asarray(
                graph[PRODUCT_INFERENCE_NODE][FEAT_PRODUCT_INFO], dtype=np.float32
            ),
            "product_name_features": np.asarray(
                graph[PRODUCT_INFERENCE_NODE][FEAT_PRODUCT_NAME], dtype=np.float32
            ),
            "edge_index": np.asarray(
                graph[USER_REL_PRODUCT][EDGE_INDEX], dtype=np.int64
            ),
            "edge_attr": np.asarray(
                graph[(PRODUCT_INFERENCE_NODE, USER_REL_PRODUCT, USER_NODE)][
                    USER_REL_PRODUCT
                ],
                dtype=np.float32,
            ),
        }
        return inputs

    def infer_graph(self, graph: dict):
        inputs_np = self.prepare_triton_inputs(graph)

        inputs = []
        for name, arr in inputs_np.items():
            dtype = "INT64" if arr.dtype == np.int64 else "FP32"
            inp = grpcclient.InferInput(name, arr.shape, dtype)
            inp.set_data_from_numpy(arr)
            inputs.append(inp)

        outputs = [grpcclient.InferRequestedOutput("user_emb")]

        response = self.triton_client.infer(
            model_name="graph_model",
            inputs=inputs,
            outputs=outputs,
        )

        return response.as_numpy("user_emb")

    def search_users(self, user_embs: np.ndarray) -> list[list[str]]:
        rec_products = []
        for query_vec in user_embs:
            results = search_vector(
                query_vec, self.index, self.prod_idxs, top_k=self.top_k
            )
            user_rec = [res[ASIN] for res in results]
            rec_products.append(user_rec)
        return rec_products

    def predict(self, reviews: list) -> list[list[str]]:
        graph = self.preprocess(reviews)
        user_embs = self.infer_graph(graph)
        return self.search_users(user_embs)
