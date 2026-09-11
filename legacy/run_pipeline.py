import subprocess

steps = [
    #("Daily Data", "run_daily.py"),
    ("Signal Graph", "run_signalgraph.py"),
    ("Features", "run_features.py"),
    ("Model Training", "run_model.py"),
]

for name, script in steps:
    print(f"\n=== Running {name} ({script}) ===\n")
    result = subprocess.run(["poetry", "run", "python", script])
    if result.returncode != 0:
        print(f"❌ {name} failed. Stopping.")
        break
