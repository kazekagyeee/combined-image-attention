import json
import numpy as np
import warnings
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from config import BenchmarkConfig

warnings.filterwarnings("ignore")


class RAGDiagnosticBenchmark:
    def __init__(self, config=BenchmarkConfig):
        self.embeddings = {}
        self.sources = []
        self.parent_info = []
        self.core_embedding_path = config.core_embeddings_path
        self.embeddings_under_research = config.embeddings_under_research
        self.base_embeddings_path = config.base_embeddings_path

    # ================= LOAD =================
    def load_embeddings(self):
        with open(self.core_embedding_path, "r", encoding="utf-8") as f:
            parents = json.load(f)

        self.embeddings["parents"] = np.array([x["text_embedding"] for x in parents])
        self.parent_info = [
            x.get("question", f"Parent_{i+1}") for i, x in enumerate(parents)
        ]

        with open(self.embeddings_under_research, "r", encoding="utf-8") as f:
            test = json.load(f)
        self.embeddings["test"] = np.array([x["text_embedding"] for x in test])

        with open(self.base_embeddings_path, "r", encoding="utf-8") as f:
            base = json.load(f)
        self.embeddings["base"] = np.array([x["text_embedding"] for x in base])

    # ================= ALIGN =================
    def align_dimensions(self):
        min_dim = min(
            self.embeddings["parents"].shape[1],
            self.embeddings["test"].shape[1],
            self.embeddings["base"].shape[1],
        )
        for k in self.embeddings:
            self.embeddings[k] = self.embeddings[k][:, :min_dim]

    # ================= DIAGNOSTIC METRICS =================
    def diagnostic_metrics(self):
        results = []

        for p_idx, parent in enumerate(self.embeddings["parents"]):
            test_dist = np.array([cosine(parent, c) for c in self.embeddings["test"]])
            base_dist = np.array([cosine(parent, c) for c in self.embeddings["base"]])

            test_sim = 1 - test_dist
            base_sim = 1 - base_dist

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(test_dist) + np.var(base_dist)) / 2
            )
            cohens_d = (
                (np.mean(base_dist) - np.mean(test_dist)) / pooled_std
                if pooled_std > 0 else 0
            )

            results.append({
                "parent": self.parent_info[p_idx],
                "test_mean_distance": float(np.mean(test_dist)),
                "base_mean_distance": float(np.mean(base_dist)),
                "test_median_distance": float(np.median(test_dist)),
                "base_median_distance": float(np.median(base_dist)),
                "test_mean_similarity": float(np.mean(test_sim)),
                "base_mean_similarity": float(np.mean(base_sim)),
                "distance_gap": float(np.mean(base_dist) - np.mean(test_dist)),
                "cohens_d": float(cohens_d),
                "wasserstein": float(wasserstein_distance(test_dist, base_dist)),
            })

        return results

    # ================= AGGREGATE =================
    def aggregate(self, per_parent):
        keys = per_parent[0].keys()
        agg = {}

        for k in keys:
            if k == "parent":
                continue
            agg[k] = float(np.mean([p[k] for p in per_parent]))

        verdict = "TEST BETTER" if agg["distance_gap"] > 0 else "BASE BETTER"

        return agg, verdict

    # ================= HTML =================
    def save_html(self, per_parent, agg, verdict):
        rows = ""
        for p in per_parent:
            rows += f"""
            <tr>
                <td>{p['parent'][:40]}</td>
                <td>{p['test_mean_distance']:.4f}</td>
                <td>{p['base_mean_distance']:.4f}</td>
                <td>{p['distance_gap']:.4f}</td>
                <td>{p['cohens_d']:.2f}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RAG Diagnostic Benchmark</title>
<style>
body {{ font-family: Arial; margin: 40px; background:#f8f9fa; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
th {{ background: #343a40; color: white; }}
.verdict {{ font-size: 24px; margin: 30px 0; }}
.good {{ color: green; }}
.bad {{ color: red; }}
</style>
</head>
<body>

<h1>RAG Diagnostic Benchmark</h1>

<table>
<tr>
<th>Parent</th>
<th>Test Mean Dist</th>
<th>Base Mean Dist</th>
<th>Distance Gap</th>
<th>Cohen’s d</th>
</tr>
{rows}
</table>

<h2 class="verdict {'good' if verdict=='TEST BETTER' else 'bad'}">
Verdict: {verdict}
</h2>

<h3>Aggregated</h3>
<pre>{json.dumps(agg, indent=2)}</pre>

</body>
</html>
"""

        with open("rag_diagnostic_report.html", "w", encoding="utf-8") as f:
            f.write(html)

    # ================= RUN =================
    def run(self):
        self.load_embeddings()
        self.align_dimensions()
        per_parent = self.diagnostic_metrics()
        agg, verdict = self.aggregate(per_parent)
        self.save_html(per_parent, agg, verdict)
        print("Saved: rag_diagnostic_report.html")


if __name__ == "__main__":
    RAGDiagnosticBenchmark().run()
