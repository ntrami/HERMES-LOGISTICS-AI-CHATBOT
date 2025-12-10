import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from backend.app import process_query  # noqa: E402


def build_test_queries():
    base = [
        ("Which route had the most delays last week?", "delays_by_route"),
        ("Show delays per route", "delays_by_route"),
        ("Routes with highest delay", "delays_by_route"),
        ("Top 3 warehouses by processing time", "top_warehouses"),
        ("Fastest warehouses", "top_warehouses"),
        ("Which warehouse is best", "top_warehouses"),
        ("Show delay reasons", "delay_reasons"),
        ("Breakdown of delay causes", "delay_reasons"),
        ("Why are shipments delayed", "delay_reasons"),
        ("Predict delay next week", "predict_delay"),
        ("Forecast next week delays", "predict_delay"),
        ("Give me delay prediction", "predict_delay"),
    ]
    queries = base * 4 + random.sample(base, 2)  # ~50 queries
    random.shuffle(queries)
    return queries


def f1_score(y_true, y_pred):
    labels = set(y_true) | set(y_pred)
    f1s = []
    for label in labels:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def evaluate_method(method: int, queries):
    y_true, y_pred = [], []
    start = time.perf_counter()
    for q, label in queries:
        result = process_query(q, method)
        y_true.append(label)
        y_pred.append(result["intent"])
    elapsed = time.perf_counter() - start
    return {
        "method": method,
        "f1": round(f1_score(y_true, y_pred), 3),
        "avg_time_s": round(elapsed / len(queries), 4),
    }


def main():
    queries = build_test_queries()
    print("Evaluating 3 methods on", len(queries), "queries...")
    results = [evaluate_method(m, queries) for m in [1, 2, 3]]
    for res in results:
        name = {1: "Rule-based", 2: "Similarity", 3: "ML Classifier"}[res["method"]]
        print(f"Method {res['method']} ({name}): F1 {res['f1']}, Avg Time {res['avg_time_s']}s")


if __name__ == "__main__":
    main()


