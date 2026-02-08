import logging
import os
from dataclasses import dataclass

import faiss
import numpy as np
from structures import Action, SkipDecision

# usage of 'faiss-cpu' and 'torch/numpy' results in OpenMP runtime conflicts.
# this setting allows the code to run, but may have performance implications.
# see known issues: https://github.com/faiss-wheels/faiss-wheels/issues/40
# and: https://github.com/peterwittek/somoclu/issues/135
# TODO: consider whether switching uv to conda is worthwhile and works around this
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class SearchResult:
    similarity: float
    decision: SkipDecision

    def __str__(self):
        return (
            f"SearchResult(similarity={self.similarity:.2f}, decision={self.decision})"
        )


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
