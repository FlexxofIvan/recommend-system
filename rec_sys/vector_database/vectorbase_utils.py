import pickle

import numpy as np
from annoy import AnnoyIndex


def load_index(index_path, mapping_path, dim):
    index = AnnoyIndex(dim, "angular")
    index.load(str(index_path))

    with open(mapping_path, "rb") as f:
        prod_idxs = pickle.load(f)

    return index, prod_idxs


def search_vector(
    query_vec: np.ndarray,
    index: AnnoyIndex,
    prod_idxs: list,
    top_k: int = 10,
):
    query_vec = query_vec

    ids, distances = index.get_nns_by_vector(
        query_vec,
        top_k,
        include_distances=True,
    )

    results = [
        {
            "asin": prod_idxs[i],
            "distance": dist,
        }
        for i, dist in zip(ids, distances)
    ]

    return results
