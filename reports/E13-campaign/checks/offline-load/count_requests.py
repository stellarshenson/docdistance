"""DEF-DEPS-14: count the connections that an offline load of the cached models opens.

The parent process listens on a free local port and starts a child process whose ``http_proxy`` and
``https_proxy`` name that port and whose environment holds no ``HF_HUB_OFFLINE``. Every request the
child sends reaches the listener, which counts the connection and closes it. The child imports
``huggingface_hub`` first (the case of a host process that imported the library before docdistance),
loads the segmenter and the three OpenVINO models with ``offline=True``, embeds one text and scores
one source-conditioned triple, both with the plain reader named (``input_format="plain"``). The
parent prints the lines of the child and the connection count.

Run from the repository root, which holds ``docdistance.json``:
``CI=true .venv/bin/python reports/E13-campaign/checks/offline-load/count_requests.py``
"""

import os
import socket
import subprocess
import sys
import threading

TEXT_A = "The cat sat still. The dog ran fast."
TEXT_B = "The cat sat still. Birds fly high."
SOURCE = "The cat sat still. The dog ran fast. Birds fly high. Fish swim deep."


def child() -> None:
    import huggingface_hub

    from docdistance import DocDistance, encoders

    loads = {
        "segmenter": encoders.Segmenter,
        "encoder": encoders.load_encoder,
        "reranker": encoders.load_reranker,
        "nli": encoders.load_nli,
    }
    failed = 0
    for name, load in loads.items():
        try:
            load()
            print(f"{name}: loaded", flush=True)
        except encoders.ModelsNotInstalled:
            failed += 1
            print(f"{name}: ModelsNotInstalled", flush=True)
    print("offline flag after the loads:", huggingface_hub.constants.HF_HUB_OFFLINE, flush=True)
    if failed:
        return
    dd = DocDistance()
    print("embed shape:", dd.embed(TEXT_A, input_format="plain").shape, flush=True)
    result = dd.distance_wrt_source(TEXT_A, TEXT_B, SOURCE, input_format="plain")
    print("source-conditioned d_sel:", result.d_sel, "grd_a:", result.grd_a, flush=True)


def count_connections(listener: socket.socket, accepted: list) -> None:
    while True:
        try:
            connection, _ = listener.accept()
        except OSError:  # the listener was closed: the child has ended
            return
        accepted.append(1)
        connection.close()


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
        sys.exit(0)
    listener = socket.create_server(("127.0.0.1", 0))
    proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
    accepted: list = []
    threading.Thread(target=count_connections, args=(listener, accepted), daemon=True).start()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env["http_proxy"] = env["https_proxy"] = env["HTTP_PROXY"] = env["HTTPS_PROXY"] = proxy
    res = subprocess.run(
        [sys.executable, __file__, "--child"], env=env, capture_output=True, text=True, check=False
    )
    listener.close()
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
    print(f"connections opened by the child: {len(accepted)}")
    sys.exit(res.returncode)
