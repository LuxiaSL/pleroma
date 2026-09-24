"""``python -m pleroma.serve [loom_serve args...]`` — the same server as
``python -m pleroma.serve.legacy``, with the same thread pins and logging
setup applied before anything heavy is imported."""

from __future__ import annotations

import logging
import os
import sys

# Same pins and same logging format as ``pleroma.serve.legacy`` applies at
# import; explicit here, before pleroma.serve.app (and so before torch).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)

from pleroma.serve.app import main  # noqa: E402

sys.exit(main())
