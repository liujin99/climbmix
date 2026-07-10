"""
Create a small test dataset for CLIMB demo.
"""

import argparse
import numpy as np
import pandas as pd


def create_test_data(output_path: str, num_docs: int = 500, num_clusters: int = 10):
    rng = np.random.default_rng(42)

    texts = []
    clusters = []
    for i in range(num_docs):
        cluster_id = rng.integers(0, num_clusters)
        text = f"Document {i} from cluster {cluster_id}. This is sample text for testing. "
        text += f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * rng.integers(5, 50)
        texts.append(text)
        clusters.append(cluster_id)

    quality_scores = rng.uniform(2.0, 5.0, size=(num_docs, 4))
    quality_columns = ["qs_quality", "qs_educational", "qs_informational", "qs_advertisement"]

    char_counts = np.array([len(t) for t in texts], dtype=np.int64)

    df = pd.DataFrame({
        "text": texts,
        "cluster": clusters,
        "row_in_shard": np.arange(num_docs),
    })
    for i, col in enumerate(quality_columns):
        df[col] = quality_scores[:, i]
    df["doc_char_count"] = char_counts

    df.to_parquet(output_path, index=False)
    print(f"[CreateTestData] Saved {num_docs} docs, {num_clusters} clusters → {output_path}")

    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        n = sum(1 for x in clusters if x == c)
        print(f"  Cluster {c}: {n} docs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-docs", type=int, default=500)
    parser.add_argument("--num-clusters", type=int, default=10)
    args = parser.parse_args()
    create_test_data(args.output, args.num_docs, args.num_clusters)


if __name__ == "__main__":
    main()
