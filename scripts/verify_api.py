"""Automaticke testy"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    import urllib.request
    import urllib.error

    checks = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        checks.append((name, status))
        print(f"  [{status}] {name}")

    print("\nOvereni REST API\n")

    for f in ["api/__init__.py", "api/main.py", "api/constants.py", "api/services.py",
              "api/schemas/__init__.py", "api/schemas/request.py", "api/schemas/response.py"]:
        check(f"{f} existuje", Path(f).exists())
    check("Dockerfile existuje", Path("Dockerfile").exists())

    print("\n  Spoustim server...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(_PROJECT_ROOT),
    )

    ready = False
    for _ in range(30):
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2)
            if req.status == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    check("Server nastartoval", ready)

    if ready:
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8765/health")
            data = json.loads(resp.read())
            check(f"GET /health -> 200 (status={data.get('status')})", data.get("status") == "ok")
        except Exception as e:
            check(f"GET /health -> 200 (error: {e})", False)

        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8765/categories")
            data = json.loads(resp.read())
            check(f"GET /categories -> {len(data.get('categories', []))} kategorii", len(data.get("categories", [])) == 13)
        except Exception as e:
            check(f"GET /categories (error: {e})", False)

        try:
            payload = json.dumps({
                "product_weight_g": 500,
                "product_length_cm": 20,
                "product_height_cm": 10,
                "product_width_cm": 15,
                "category_group": "electronics",
                "avg_price": 150.0,
                "daily_turnover": 0.0,
                "cold_start": True,
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            check("POST /predict valid -> 200", resp.status == 200)
            check(f"Response ma recommended_zone ({data.get('recommended_zone')})", "recommended_zone" in data)
            check(f"Response ma confidence ({data.get('confidence')})", 0 <= data.get("confidence", -1) <= 1)
            check(f"Response ma similar_products ({len(data.get('similar_products', []))})", len(data.get("similar_products", [])) == 5)
            check(f"cold_start_mode={data.get('cold_start_mode')}", data.get("cold_start_mode") == True)
        except Exception as e:
            check(f"POST /predict valid (error: {e})", False)
            for name in ["recommended_zone", "confidence", "similar_products", "cold_start_mode"]:
                check(name, False)

        try:
            payload = json.dumps({"product_length_cm": 20}).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict", data=payload, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req)
                check("POST /predict invalid -> 422", False)
            except urllib.error.HTTPError as he:
                check(f"POST /predict invalid -> {he.code}", he.code == 422)
        except Exception as e:
            check(f"POST /predict invalid (error: {e})", False)

        try:
            payload = json.dumps({
                "product_weight_g": 500,
                "product_length_cm": 20,
                "product_height_cm": 10,
                "product_width_cm": 15,
                "category_group": "electronics",
                "avg_price": 150.0,
                "daily_turnover": 2.5,
                "cold_start": False,
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            check(f"POST /predict full model -> cold_start_mode=False", data.get("cold_start_mode") == False)
        except Exception as e:
            check(f"POST /predict full model (error: {e})", False)

        try:
            payload = json.dumps({
                "products": [
                    {
                        "product_id": "BATCH-A",
                        "product_weight_g": 500,
                        "product_length_cm": 20,
                        "product_height_cm": 10,
                        "product_width_cm": 15,
                        "category_group": "electronics",
                        "avg_price": 150.0,
                        "cold_start": True,
                    },
                    {
                        "product_id": "BATCH-B",
                        "product_weight_g": 12000,
                        "product_length_cm": 60,
                        "product_height_cm": 40,
                        "product_width_cm": 50,
                        "category_group": "furniture",
                        "avg_price": 800.0,
                        "daily_turnover": 0.3,
                        "cold_start": False,
                    },
                    {
                        "product_id": "BATCH-C",
                        "product_weight_g": 200,
                        "product_length_cm": 15,
                        "product_height_cm": 5,
                        "product_width_cm": 10,
                        "category_group": "books_media",
                        "avg_price": 50.0,
                        "cold_start": True,
                    },
                ]
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            check("POST /predict/batch valid 3 produkty -> 200", resp.status == 200)
            check(f"Batch summary.total={data.get('summary', {}).get('total')}", data.get("summary", {}).get("total") == 3)
            check(f"Batch summary.ok_count={data.get('summary', {}).get('ok_count')}", data.get("summary", {}).get("ok_count") == 3)
            check(f"Batch summary.error_count={data.get('summary', {}).get('error_count')}", data.get("summary", {}).get("error_count") == 0)
            results = data.get("results", [])
            check(f"Batch results pocet={len(results)}", len(results) == 3)
            check("Batch all results status=ok", all(r.get("status") == "ok" for r in results))
        except Exception as e:
            check(f"POST /predict/batch valid (error: {e})", False)

        try:
            payload = json.dumps({"products": []}).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req)
                check("POST /predict/batch prazdny -> 422", False)
            except urllib.error.HTTPError as he:
                check(f"POST /predict/batch prazdny -> {he.code}", he.code == 422)
        except Exception as e:
            check(f"POST /predict/batch prazdny (error: {e})", False)

        try:
            over_limit_products = [
                {
                    "product_weight_g": 500,
                    "product_length_cm": 20,
                    "product_height_cm": 10,
                    "product_width_cm": 15,
                    "category_group": "electronics",
                    "avg_price": 150.0,
                    "cold_start": True,
                }
                for _ in range(101)
            ]
            payload = json.dumps({"products": over_limit_products}).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req)
                check("POST /predict/batch 101 produktu -> 422", False)
            except urllib.error.HTTPError as he:
                check(f"POST /predict/batch 101 produktu -> {he.code}", he.code == 422)
        except Exception as e:
            check(f"POST /predict/batch over limit (error: {e})", False)

        try:
            payload = json.dumps({
                "products": [
                    {
                        "product_id": "OK-1",
                        "product_weight_g": 500,
                        "product_length_cm": 20,
                        "product_height_cm": 10,
                        "product_width_cm": 15,
                        "category_group": "electronics",
                        "avg_price": 150.0,
                        "cold_start": True,
                    },
                    {
                        "product_id": "OK-2",
                        "product_weight_g": 800,
                        "product_length_cm": 25,
                        "product_height_cm": 12,
                        "product_width_cm": 18,
                        "category_group": "home_garden",
                        "avg_price": 220.0,
                        "cold_start": True,
                    },
                    {
                        "product_id": "BAD-1",
                        "product_weight_g": 999999,
                        "product_length_cm": 20,
                        "product_height_cm": 10,
                        "product_width_cm": 15,
                        "category_group": "electronics",
                        "avg_price": 150.0,
                        "cold_start": True,
                    },
                ]
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            check("POST /predict/batch partial -> 200", resp.status == 200)
            check(f"Batch partial ok_count={data.get('summary', {}).get('ok_count')}", data.get("summary", {}).get("ok_count") == 2)
            check(f"Batch partial error_count={data.get('summary', {}).get('error_count')}", data.get("summary", {}).get("error_count") == 1)
            results = data.get("results", [])
            check("Batch partial result[2].status=error", len(results) >= 3 and results[2].get("status") == "error")
            check("Batch partial result[2].product_id=BAD-1", len(results) >= 3 and results[2].get("product_id") == "BAD-1")
        except Exception as e:
            check(f"POST /predict/batch partial (error: {e})", False)

        try:
            payload = json.dumps({
                "products": [
                    {
                        "product_id": "SKU-TEST",
                        "product_weight_g": 500,
                        "product_length_cm": 20,
                        "product_height_cm": 10,
                        "product_width_cm": 15,
                        "category_group": "electronics",
                        "avg_price": 150.0,
                        "cold_start": True,
                    }
                ]
            }).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            results = data.get("results", [])
            check("Batch product_id echo: results[0].product_id=SKU-TEST", len(results) >= 1 and results[0].get("product_id") == "SKU-TEST")
        except Exception as e:
            check(f"POST /predict/batch product_id echo (error: {e})", False)

        try:
            payload = json.dumps({"products": "not a list"}).encode()
            req = urllib.request.Request("http://127.0.0.1:8765/predict/batch", data=payload, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req)
                check("POST /predict/batch wrong shape -> 422", False)
            except urllib.error.HTTPError as he:
                check(f"POST /predict/batch wrong shape -> {he.code}", he.code == 422)
        except Exception as e:
            check(f"POST /predict/batch wrong shape (error: {e})", False)

        log_path = Path("logs/api_audit.log")
        check("Audit log existuje po requestech", log_path.exists())

    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()

    n_pass = sum(1 for _, s in checks if s == "PASS")
    n_fail = sum(1 for _, s in checks if s == "FAIL")
    print(f"\n{'=' * 40}")
    print(f"Celkem: {n_pass} PASS, {n_fail} FAIL")

    if n_fail > 0:
        print("VYSLEDEK: FAIL")
        sys.exit(1)
    else:
        print("VYSLEDEK: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
