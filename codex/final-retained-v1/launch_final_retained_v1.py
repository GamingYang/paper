import subprocess
import os
from pathlib import Path


run_dir = Path(r"C:\Users\ROG\Desktop\paper\codex\final-retained-v1")
python_exe = Path(r"D:\Anaconda\envs\battery_project\python.exe")
script = run_dir / "train_final_retained_scheme_a.py"
stdout_path = run_dir / "final-retained-v1_stdout.log"
stderr_path = run_dir / "final-retained-v1_stderr.log"
pid_path = run_dir / "final-retained-v1.pid"

with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
    env = os.environ.copy()
    env_root = python_exe.parent
    path_prefix = [
        str(env_root),
        str(env_root / "Library" / "bin"),
        str(env_root / "DLLs"),
        str(env_root / "Scripts"),
    ]
    env["PATH"] = ";".join(path_prefix + [env.get("PATH", "")])
    env["CONDA_PREFIX"] = str(env_root)
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [str(python_exe), str(script)],
        cwd=r"C:\Users\ROG\Desktop\paper",
        stdout=stdout,
        stderr=stderr,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

pid_path.write_text(str(process.pid), encoding="utf-8")
print(process.pid)
