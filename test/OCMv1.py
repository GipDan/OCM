"""
OCM v1 入口：将项目根加入 PYTHONPATH 后，可启动 Streamlit 或调用 Python API。

用法：
  # 在项目根 ops_cost_model_database 下：
  streamlit run app.py

  # 或在本目录用 Python 调用：
  PYTHONPATH=.. python -c "from ocm import init_db, get_connection, insert_record, fit_and_store_model; ..."
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_streamlit() -> None:
    import subprocess

    app = _ROOT / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"],
        check=False,
        cwd=str(_ROOT),
    )


if __name__ == "__main__":
    run_streamlit()
