import subprocess
import tempfile
import os
import uuid


def execution_agent(code: str) -> dict:
    """
    Compiles and executes C code safely.
    Returns compile errors or program output.
    """

    unique_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()

    c_file = os.path.join(temp_dir, f"{unique_id}.c")
    exe_file = os.path.join(temp_dir, f"{unique_id}.exe")

    try:
        # Write code to temp file
        with open(c_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Compile
        compile_process = subprocess.run(
            ["gcc", c_file, "-o", exe_file],
            capture_output=True,
            text=True,
            timeout=5
        )

        if compile_process.returncode != 0:
            return {
                "status": "compile_error",
                "error": compile_process.stderr.strip()
            }

        # Run executable
        run_process = subprocess.run(
            [exe_file],
            capture_output=True,
            text=True,
            timeout=5
        )

        return {
            "status": "success",
            "output": run_process.stdout.strip(),
            "runtime_error": run_process.stderr.strip()
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Execution timed out."
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

    finally:
        # Cleanup
        if os.path.exists(c_file):
            os.remove(c_file)
        if os.path.exists(exe_file):
            os.remove(exe_file)