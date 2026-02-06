import json
import logging
import os

import faiss
import numpy as np
from structures import Action, SearchResult, SkipDecision

# usage of 'faiss-cpu' and 'torch/numpy' results in OpenMP runtime conflicts.
# this setting allows the code to run, but may have performance implications.
# see known issues: https://github.com/faiss-wheels/faiss-wheels/issues/40
# and: https://github.com/peterwittek/somoclu/issues/135
# TODO: consider whether switching uv to conda is worthwhile and works around this
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class SkippingVectorDB:
    def __init__(self, n_checkpoints: int, vector_dim: int, device: str = "cpu"):
        self.n_checkpoints = n_checkpoints
        self.vector_dim = vector_dim

        if device == "cuda":
            logging.error("FAISS GPU support not yet implemented in this snippet.")
            pass

        # initialise FAISS Indices
        # TODO: experiment with index types
        # TODO: experiment with dimension reduction
        # TODO: experiment with other similarity metrics (currently cosine via IP)
        self.indexes = [faiss.IndexFlatIP(vector_dim) for _ in range(n_checkpoints)]

        # metadata storage
        # maps (checkpoint, vector_id) -> SkipDecision
        self.metadata: list[dict[int, SkipDecision]] = [
            {} for _ in range(n_checkpoints)
        ]

    def add_vector(
        self, checkpoint_idx: int, vector: np.ndarray, decision: SkipDecision
    ):
        """
        Adds a vector and its associated skip decision to the DB.
        """
        if checkpoint_idx >= self.n_checkpoints:
            raise ValueError(
                f"Checkpoint {checkpoint_idx} out of bounds "
                f"(Max {self.n_checkpoints - 1})"
            )

        # normalise vector for cosine similarity
        faiss.normalize_L2(vector)

        index = self.indexes[checkpoint_idx]
        current_id = index.ntotal

        # add to index
        index.add(vector)
        self.metadata[checkpoint_idx][current_id] = decision

    def search(
        self,
        checkpoint_idx: int,
        query_vector: np.ndarray,
    ) -> SearchResult | None:
        """
        Searches for a similar vector.
        Returns the similarity and associated SkipDecision if found.
        """
        index = self.indexes[checkpoint_idx]
        if index.ntotal == 0:
            return None

        # normalise query
        faiss.normalize_L2(query_vector)

        # search with k=1
        similarities, indices = index.search(query_vector, k=1)

        similarity = similarities[0][0]
        neighbor_id = indices[0][0]

        # retrieve the decision made for that neighbor
        decision = self.metadata[checkpoint_idx][neighbor_id]

        return SearchResult(similarity=similarity, decision=decision)

    def save(self, folder_path: str):
        """Saves raw indices and metadata to a specific folder using JSON."""
        assert not os.path.exists(folder_path), (
            f"Folder {folder_path} already exists. "
            "Choose a different path or remove it."
        )
        os.makedirs(folder_path)

        for i, (index, meta) in enumerate(
            zip(self.indexes, self.metadata, strict=False)
        ):
            # save index
            index_path = os.path.join(folder_path, f"ckpt_{i}.index")
            faiss.write_index(index, index_path)

            # save metadata
            # convert {int: SkipDecision} -> {str: dict} for JSON
            json_meta = {str(k): v.__dict__ for k, v in meta.items()}

            meta_path = os.path.join(folder_path, f"ckpt_{i}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(json_meta, f, indent=2)

        logging.info(f"SkippingVectorDB content saved to {folder_path}")

    @classmethod
    def load(cls, folder_path: str, n_checkpoints: int, vector_dim: int):
        """Loads indices and metadata from a folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No DB found at {folder_path}")

        db = cls(n_checkpoints, vector_dim)

        for i in range(n_checkpoints):
            index_path = os.path.join(folder_path, f"ckpt_{i}.index")
            meta_path = os.path.join(folder_path, f"ckpt_{i}_metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"Missing files for checkpoint {i} in {folder_path}"
                )

            db.indexes[i] = faiss.read_index(index_path)

            with open(meta_path) as f:
                raw_data = json.load(f)

            # reconstruct: {str: dict} -> {int: SkipDecision}
            db.metadata[i] = {}
            for k_str, v_dict in raw_data.items():
                v_dict["action"] = Action(v_dict["action"])  # Convert str -> Enum
                db.metadata[i][int(k_str)] = SkipDecision(**v_dict)

        logging.info(f"SkippingVectorDB loaded from {folder_path}")
        return db


# example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = SkippingVectorDB(n_checkpoints=24, vector_dim=768)

    # create a dummy vector and decision
    vec = np.random.rand(1, 768).astype("float32")
    decision = SkipDecision(action=Action.SKIP, skip_count=5)

    # add to DB
    db.add_vector(checkpoint_idx=0, vector=vec, decision=decision)

    # search for similar vector
    result = db.search(checkpoint_idx=0, query_vector=vec)
    if result:
        logging.info(f"Found decision: {result}")
    else:
        logging.info("No similar vector found.")

    # save and load DB
    db.save("test-results/db")
    loaded_db = SkippingVectorDB.load(
        "test-results/db", n_checkpoints=24, vector_dim=768
    )
    loaded_result = loaded_db.search(checkpoint_idx=0, query_vector=vec)
    logging.info(f"Found decision: {loaded_result}")
