import sys, platform, os

print("=== PYTHON ENVIRONMENT INFO ===")
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Python implementation:", platform.python_implementation())
print("Machine / Architecture:", platform.machine())
print("Platform:", platform.platform())
print("System:", platform.system())
print("Release:", platform.release())
print("LibC:", platform.libc_ver())
print("Environment variables (partial):")
for k in ["PYTHONHOME", "PYTHONPATH", "PATH", "VIRTUAL_ENV"]:
    print(f"  {k} =", os.environ.get(k))
print("================================")