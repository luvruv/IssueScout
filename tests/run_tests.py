"""Runs every test_* function without needing pytest:  python tests/run_tests.py   (pytest also works)"""
import importlib, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = total = 0
for mod in sorted(f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.startswith("test_") and f.endswith(".py")):
    m = importlib.import_module(f"tests.{mod}")
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        total += 1
        try:
            getattr(m, name)(); print(f"PASS {mod}.{name}")
        except Exception:
            fails += 1; print(f"FAIL {mod}.{name}"); traceback.print_exc()
print(f"\n{total - fails}/{total} passed"); sys.exit(1 if fails else 0)
