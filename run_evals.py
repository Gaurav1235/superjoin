import json
import requests

API_URL = "http://localhost:8000"

def contains_any(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)

def eval_case(case):
    q = case["query"]
    res = requests.get(f"{API_URL}/search", params={"query": q, "k": 3}).json()

    if not res["results"]:
        return {
            "query": q,
            "category": case.get("category"),
            "passed": False,
            "reason": "no results",
            "top_text": "",
            "expected_keywords": case["expected_keywords"],
        }

    top = res["results"][0]["text"]
    keywords = case["expected_keywords"]

    passed = contains_any(top, keywords)

    return {
        "query": q,
        "category": case.get("category"),
        "passed": passed,
        "top_text": top,
        "expected_keywords": keywords,
    }

def run_all():
    with open("evals.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = [eval_case(c) for c in cases]

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print(f"\nOverall: {passed}/{total} passed\n")

    by_cat = {}
    for r in results:
        cat = r["category"] or "uncategorized"
        by_cat.setdefault(cat, []).append(r)

    for cat, rs in by_cat.items():
        p = sum(1 for r in rs if r["passed"])
        print(f"[{cat}] {p}/{len(rs)} passed")

    print("\nDetailed results:")
    for r in results:
        print("----")
        print("Category:", r["category"])
        print("Query   :", r["query"])
        print("Passed  :", r["passed"])
        print("Expected:", r["expected_keywords"])
        print("Top Text:", r["top_text"][:400])

if __name__ == "__main__":
    run_all()
