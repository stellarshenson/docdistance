"""The guardrail ``baseline.py verify`` with both memos of every DocDistance switched on.

Run from the repository root:

    CI=true .venv/bin/python reports/E13-campaign/checks/batch-memos/guardrail_memo_on.py
"""

import runpy
import sys

if __name__ == "__main__":
    from docdistance.pipeline import MEMO_SIZE, DocDistance

    init = DocDistance.__init__

    def memo_on_init(self, *args, **kwargs):
        init(self, *args, **{**kwargs, "memo_size": MEMO_SIZE})

    DocDistance.__init__ = memo_on_init
    sys.argv = ["baseline.py", "verify", "--allow", "moved,n_moved", "--grounded"]
    runpy.run_path("reports/E13-campaign/baseline.py", run_name="__main__")
