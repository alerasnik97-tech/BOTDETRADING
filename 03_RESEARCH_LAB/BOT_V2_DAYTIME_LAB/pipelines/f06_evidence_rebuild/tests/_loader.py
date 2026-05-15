# Shared loader for tests. No real data. No strategy. stdlib only.
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # f06_evidence_rebuild/
PIPE = os.path.join(ROOT, "scripts", "f06_rebuild_pipeline.py")
FIXTURES = os.path.join(ROOT, "fixtures")


def load_pipeline():
    spec = importlib.util.spec_from_file_location("f06_rebuild_pipeline", PIPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fx(name):
    return os.path.join(FIXTURES, name)
