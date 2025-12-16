import json
import numpy as np
import warnings
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from config import BenchmarkConfig

warnings.filterwarnings("ignore")


class RAGMetricsBenchmark:
    def __init__(self, config=BenchmarkConfig):
        self.embeddings = {}
        self.sources = []
        self.core_embedding_path = config.core_embedding_path
        self.embeddings_under_research = config.embeddings_under_research
        self.base_embeddings_path = config.base_embeddings_path

    # ============================================================
    # LOAD
    # ============================================================
    def load_embeddings(self):
        with open(self.core_embedding_path, "r", encoding="utf-8") as f:
            core = json.load(f)
        self.embeddings["parent"] = np.array(core[0]["text_embedding"]).reshape(1, -1)
        self.sources.append("parent")

        with open(self.embeddings_under_research, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_embs = [np.array(x["text_embedding"]) for x in test_data]
        self.embeddings["test"] = np.array(test_embs)
        self.sources.extend(["test"] * len(test_embs))

        with open(self.base_embeddings_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base_embs = [np.array(x["text_embedding"]) for x in base_data]
        self.embeddings["base"] = np.array(base_embs)
        self.sources.extend(["base"] * len(base_embs))

    # ============================================================
    # ALIGN
    # ============================================================
    def align_dimensions(self):
        dims = [
            self.embeddings["parent"].shape[1],
            self.embeddings["test"].shape[1],
            self.embeddings["base"].shape[1],
        ]
        min_dim = min(dims)

        for k in self.embeddings:
            self.embeddings[k] = self.embeddings[k][:, :min_dim]

    # ============================================================
    # COMBINE
    # ============================================================
    def combine(self):
        return np.vstack(
            [
                self.embeddings["parent"],
                self.embeddings["test"],
                self.embeddings["base"],
            ]
        )

    # ============================================================
    # RETRIEVAL (PARENT → CHILDREN)
    # ============================================================
    def calculate_retrieval_metrics(self, embeddings, k_values=(1, 3, 5, 10)):
        parent = embeddings[0]
        children = embeddings[1:]
        child_sources = self.sources[1:]

        distances = np.array([cosine(parent, x) for x in children])
        ranking = np.argsort(distances)

        ranked_sources = [child_sources[i] for i in ranking]
        ranked_distances = distances[ranking]

        metrics = {}

        for method in ("test", "base"):
            idxs = [i for i, s in enumerate(ranked_sources) if s == method]
            ranks = [i + 1 for i in idxs]
            dists = [float(ranked_distances[i]) for i in idxs]

            metrics[method] = {
                "mean_distance": float(np.mean(dists)),
                "median_distance": float(np.median(dists)),
                "mean_rank": float(np.mean(ranks)),
                "mrr": float(np.mean([1 / r for r in ranks])),
                "recall_at_k": {},
                "precision_at_k": {},
            }

            for k in k_values:
                top_k = ranked_sources[:k]
                hits = sum(1 for x in top_k if x == method)
                metrics[method]["recall_at_k"][k] = hits / len(idxs)
                metrics[method]["precision_at_k"][k] = hits / k

        metrics["distance_gap"] = (
            metrics["base"]["mean_distance"]
            - metrics["test"]["mean_distance"]
        )

        return metrics

    # ============================================================
    # CLUSTERING (DIAGNOSTIC ONLY)
    # ============================================================
    def calculate_clustering_metrics(self, embeddings):
        children = embeddings[1:]
        X = StandardScaler().fit_transform(children)

        sil, cal, dav = [], [], []
        cluster_range = range(2, min(10, len(X)))

        for k in cluster_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            sil.append(silhouette_score(X, labels))
            cal.append(calinski_harabasz_score(X, labels))
            dav.append(davies_bouldin_score(X, labels))

        best_k = list(cluster_range)[int(np.argmax(sil))]

        db = DBSCAN(eps=0.6, min_samples=5).fit_predict(X)
        noise_ratio = float(np.mean(db == -1))

        return {
            "kmeans": {
                "k": int(best_k),
                "silhouette": float(max(sil)),
                "calinski": float(max(cal)),
                "davies": float(min(dav)),
            },
            "dbscan": {
                "noise_ratio": noise_ratio,
            },
        }

    # ============================================================
    # REPORT
    # ============================================================
    def create_report(self, retrieval, clustering):
        def row(name, t, b, fmt="{:.4f}"):
            return f"""
            <tr>
                <td>{name}</td>
                <td>{fmt.format(t)}</td>
                <td>{fmt.format(b)}</td>
            </tr>
            """

        test = retrieval["test"]
        base = retrieval["base"]

        verdict = "TEST BETTER" if retrieval["distance_gap"] > 0 else "BASE BETTER"

        rows = ""
        rows += row("Mean cosine distance", test["mean_distance"], base["mean_distance"])
        rows += row("Median distance", test["median_distance"], base["median_distance"])
        rows += row("Mean rank", test["mean_rank"], base["mean_rank"], "{:.2f}")
        rows += row("MRR", test["mrr"], base["mrr"])

        for k in sorted(test["recall_at_k"]):
            rows += row(
                f"Recall@{k}",
                test["recall_at_k"][k],
                base["recall_at_k"][k],
            )
            rows += row(
                f"Precision@{k}",
                test["precision_at_k"][k],
                base["precision_at_k"][k],
            )

        with open("rag_metrics_report.html", "w", encoding="utf-8") as f:
            f.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>RAG Embedding Benchmark</title>
    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 40px;
    }}
    table {{
        border-collapse: collapse;
        width: 70%;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 8px 12px;
        text-align: center;
    }}
    th {{
        background-color: #f4f4f4;
    }}
    tr:nth-child(even) {{
        background-color: #fafafa;
    }}
    .verdict {{
        font-size: 28px;
        margin-top: 30px;
        font-weight: bold;
    }}
    </style>
    </head>
    <body>

    <h1>RAG Embedding Benchmark</h1>

    <h2>Retrieval Metrics</h2>
    <table>
    <tr>
        <th>Metric</th>
        <th>Test</th>
        <th>Base</th>
    </tr>
    {rows}
    </table>

    <h2>Clustering Diagnostics</h2>
    <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>KMeans clusters (k)</td><td>{clustering["kmeans"]["k"]}</td></tr>
    <tr><td>Silhouette</td><td>{clustering["kmeans"]["silhouette"]:.4f}</td></tr>
    <tr><td>Calinski–Harabasz</td><td>{clustering["kmeans"]["calinski"]:.2f}</td></tr>
    <tr><td>Davies–Bouldin</td><td>{clustering["kmeans"]["davies"]:.4f}</td></tr>
    <tr><td>DBSCAN noise ratio</td><td>{clustering["dbscan"]["noise_ratio"]:.4f}</td></tr>
    </table>

    <div class="verdict">{verdict}</div>

    </body>
    </html>
    """)

    # ============================================================
    # RUN
    # ============================================================
    def run(self):
        self.load_embeddings()
        self.align_dimensions()
        all_embs = self.combine()

        retrieval = self.calculate_retrieval_metrics(all_embs)
        clustering = self.calculate_clustering_metrics(all_embs)

        self.create_report(retrieval, clustering)

        print("RAG benchmark finished")
        print("Distance gap:", retrieval["distance_gap"])
        print("Verdict:", "TEST BETTER" if retrieval["distance_gap"] > 0 else "BASE BETTER")


if __name__ == "__main__":
    RAGMetricsBenchmark().run()
