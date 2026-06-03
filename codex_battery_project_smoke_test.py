import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
RESULT_JSON = ROOT / "codex_battery_project_smoke_result.json"
RESULT_XLSX = ROOT / "codex_battery_project_smoke_result.xlsx"


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


packages = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "matplotlib",
    "openpyxl",
    "docx",
]

result = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "packages": {name: package_available(name) for name in packages},
    "minimal_calculation": sum(range(1, 11)),
    "json_created": True,
    "xlsx_created": False,
    "xlsx_error": None,
}

try:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "smoke_test"
    ws.append(["key", "value"])
    for key, value in result.items():
        if key == "packages":
            ws.append([key, json.dumps(value, ensure_ascii=False)])
        else:
            ws.append([key, str(value)])
    ws.append(["status", "battery_project environment can run Python and write Excel"])
    wb.save(RESULT_XLSX)
    result["xlsx_created"] = True
except Exception as exc:
    result["xlsx_error"] = f"{type(exc).__name__}: {exc}"

RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(result, ensure_ascii=False, indent=2))
