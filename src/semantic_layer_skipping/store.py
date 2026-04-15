# we need to import torch before faiss to avoid OpenMP conflicts
# this arises as a problem when other scripts import store.py before torch
# see the KMP_DUPLICATE_LIB_OK setting below for workaround and links to issues
import torch  # noqa: I001, F401
import gc
import shutil
import json
import logging
import os
import random

import faiss
import numpy as np
from structures import Action, SearchResult, SkipDecision

# usage of 'faiss-cpu' and 'torch/numpy' results in OpenMP runtime conflicts.
# this setting allows the code to run, but may have performance implications.
# see known issues: https://github.com/faiss-wheels/faiss-wheels/issues/40
# and: https://github.com/peterwittek/somoclu/issues/135
# TODO: consider whether switching uv to conda is worthwhile and works around this
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

N_PROBE = 128  # Number of clusters to search in IVFPQ (if used)
MAX_NUM_TRAINING_VECTORS = 160_000


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

    def get_index_sizes(self) -> dict[int, int]:
        """
        Returns a dictionary mapping checkpoint indices to the total number of
        vectors stored in their respective FAISS indexes.
        """
        return {ckpt_idx: index.ntotal for ckpt_idx, index in enumerate(self.indexes)}

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
        Returns the similarity, associated SkipDecision, and neighbour ID if found.
        """
        index = self.indexes[checkpoint_idx]
        if index.ntotal == 0:
            return None

        # normalise query
        faiss.normalize_L2(query_vector)

        # search with k=1
        similarities, indices = index.search(query_vector, k=1)

        similarity = similarities[0][0]
        neighbour_id = indices[0][0]

        if neighbour_id == -1:
            logging.debug(
                f"FAISS returned -1 for ckpt {checkpoint_idx}. "
                f"{similarity=}. {neighbour_id=}"
                f"Possibly a NaN vector?"
            )
            return None

        # retrieve the decision made for that neighbour
        decision = self.metadata[checkpoint_idx][neighbour_id]

        return SearchResult(
            similarity=similarity,
            decision=decision,
            neighbour_id=int(neighbour_id),  # convert numpy to python int
        )

    def save(self, folder_path: str):
        """Saves raw indices and metadata to a specific folder using JSON."""
        assert not os.path.exists(folder_path), (
            f"Folder {folder_path} already exists. "
            "Choose a different path or remove it."
        )
        os.makedirs(folder_path)

        for i, (index, meta) in enumerate(
            zip(self.indexes, self.metadata, strict=True)
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

            logging.info(f"Saved index {i} with {index.ntotal} vectors.")

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

            if hasattr(db.indexes[i], "nprobe"):
                logging.info(f"Setting nprobe={N_PROBE} for index {i}")
                db.indexes[i].nprobe = N_PROBE

            with open(meta_path) as f:
                raw_data = json.load(f)

            # reconstruct: {str: dict} -> {int: SkipDecision}
            db.metadata[i] = {}
            for k_str, v_dict in raw_data.items():
                v_dict["action"] = Action(v_dict["action"])  # Convert str -> Enum
                db.metadata[i][int(k_str)] = SkipDecision(**v_dict)

        logging.info(f"SkippingVectorDB loaded from {folder_path}")
        return db

    @staticmethod
    def create_merged_subsampled_db_from_chunks(
        base_dir: str,
        output_dir: str,
        n_checkpoints: int,
        vector_dim: int,
        keep_fraction: float = 0.10,
    ):
        logging.info(
            f"Creating merged subsampled DB (keeping {keep_fraction * 100}% of vectors)"
        )

        # initialise new DB
        merged_db = SkippingVectorDB(n_checkpoints, vector_dim)

        chunk_dirs = sorted(
            [
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if d.startswith("db_chunk_")
            ]
        )
        assert chunk_dirs, f"No chunk directories found in {base_dir}"

        for chunk_dir in chunk_dirs:
            logging.info(f"Processing {chunk_dir}...")
            chunk_db = SkippingVectorDB.load(chunk_dir, n_checkpoints, vector_dim)

            for ckpt in range(n_checkpoints):
                index = chunk_db.indexes[ckpt]
                checkpoint_metadata = chunk_db.metadata[ckpt]

                n_vectors = index.ntotal
                if n_vectors == 0:
                    continue

                # select a subset of indices to keep
                n_keep = int(n_vectors * keep_fraction)
                keep_indices = random.sample(range(n_vectors), n_keep)

                # reconstruct vectors (FAISS allows this for Flat indices)
                for idx in keep_indices:
                    vec = index.reconstruct(idx).reshape(1, -1)
                    decision = checkpoint_metadata[idx]

                    # add to the new DB
                    merged_db.add_vector(ckpt, vec, decision)

            # force garbage collection of the large chunk
            del chunk_db

        # save the new compact DB
        merged_db.save(output_dir)
        logging.info(f"Successfully saved merged subsampled DB to {output_dir}")

    @staticmethod
    def create_ivfpq_db_from_exact(
        source_dir: str,
        output_dir: str,
        n_checkpoints: int,
        vector_dim: int,
        nlist: int = 4096,  # number of clusters (Voronoi cells)
        m: int = 64,  # subquantisers (vector_dim 1536 must be divisible by m)
        nbits: int = 8,  # bits per subquantiser (compresses to 1 byte)
    ):
        """
        Converts an exact DB into an IVFPQ DB.
        This is done checkpoint-by-checkpoint to save memory.
        """
        if vector_dim % m != 0:
            raise ValueError(
                f"vector_dim ({vector_dim}) must be divisible by m ({m}) for IVFPQ."
            )

        logging.info(
            f"Starting memory-efficient IVFPQ Conversion. Reading from {source_dir}"
        )
        # ensure output directory does not exist
        assert not os.path.exists(output_dir), (
            f"Folder {output_dir} already exists. Choose a different path or remove it."
        )
        os.makedirs(output_dir)

        for ckpt in range(n_checkpoints):
            exact_index_path = os.path.join(source_dir, f"ckpt_{ckpt}.index")
            exact_meta_path = os.path.join(source_dir, f"ckpt_{ckpt}_metadata.json")

            if not os.path.exists(exact_index_path) or not os.path.exists(
                exact_meta_path
            ):
                logging.warning(f"Missing files for checkpoint {ckpt}. Skipping.")
                continue

            logging.info(f"Processing Checkpoint {ckpt}")

            # load only this specific checkpoint's index into memory
            exact_index = faiss.read_index(exact_index_path)
            n_vectors = exact_index.ntotal

            if n_vectors == 0:
                logging.info(f"Checkpoint {ckpt} is empty. Saving empty files.")
                faiss.write_index(
                    faiss.IndexFlatIP(vector_dim),
                    os.path.join(output_dir, f"ckpt_{ckpt}.index"),
                )
                shutil.copy2(
                    exact_meta_path,
                    os.path.join(output_dir, f"ckpt_{ckpt}_metadata.json"),
                )
                del exact_index
                continue

            logging.info(f"Checkpoint {ckpt}: Extracting {n_vectors} vectors...")
            all_vectors = exact_index.reconstruct_n(0, n_vectors)

            # free memory: we have the raw vectors,
            # so we can delete the exact index immediately
            del exact_index
            gc.collect()

            # FAISS requires at least 39 points per centroid.
            MIN_POINTS_PER_CENTROID = 39
            current_nlist = nlist
            if n_vectors < current_nlist * MIN_POINTS_PER_CENTROID:
                current_nlist = max(1, n_vectors // MIN_POINTS_PER_CENTROID)
                logging.warning(
                    f"Checkpoint {ckpt}: Reduced nlist to {current_nlist} "
                    f"to satisfy FAISS constraints."
                )

            logging.info(
                f"Checkpoint {ckpt}: "
                f"Training IVFPQ with nlist={current_nlist}, m={m}..."
            )
            quantizer = faiss.IndexFlatIP(vector_dim)
            ivfpq_index = faiss.IndexIVFPQ(
                quantizer, vector_dim, current_nlist, m, nbits
            )
            ivfpq_index.metric_type = faiss.METRIC_INNER_PRODUCT

            # train on a maximum number of vectors to save time without losing accuracy
            train_subset = all_vectors
            if n_vectors > MAX_NUM_TRAINING_VECTORS:
                idx = np.random.choice(
                    n_vectors, MAX_NUM_TRAINING_VECTORS, replace=False
                )
                train_subset = all_vectors[idx]

            ivfpq_index.train(train_subset)
            del train_subset

            logging.info(f"Checkpoint {ckpt}: Adding encoded vectors to IVFPQ index...")
            ivfpq_index.add(all_vectors)

            # save the newly created IVFPQ index directly to disk
            out_index_path = os.path.join(output_dir, f"ckpt_{ckpt}.index")
            faiss.write_index(ivfpq_index, out_index_path)

            # save metadata: since IDs are preserved 1:1, we just copy the file
            out_meta_path = os.path.join(output_dir, f"ckpt_{ckpt}_metadata.json")
            shutil.copy2(exact_meta_path, out_meta_path)
            logging.info(f"Checkpoint {ckpt}: Saved IVFPQ index and copied metadata.")

            # garbace collection
            del all_vectors
            del ivfpq_index
            del quantizer
            gc.collect()

        logging.info(
            f"Successfully saved memory-efficient IVFPQ conversion to {output_dir}"
        )


def verify_and_set_faiss_threads():
    # check how many allocated CPUs
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    logging.info(f"Hardware Check: There are {cpus} available CPU cores.")

    # faiss threads
    current_faiss_threads = faiss.omp_get_max_threads()
    logging.info(
        f"Hardware Check: "
        f"FAISS OpenMP is defaulting to {current_faiss_threads} threads."
    )

    # set FAISS to use all available cores
    if current_faiss_threads < cpus // 2:
        num_threads = max(1, cpus // 2)
        faiss.omp_set_num_threads(num_threads)
        logging.info(
            f"Hardware Check: "
            f"Set FAISS OpenMP to use {faiss.omp_get_max_threads()} threads."
        )


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
