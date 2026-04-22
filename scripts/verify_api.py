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
