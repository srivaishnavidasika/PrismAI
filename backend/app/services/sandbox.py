import os
import subprocess
import tempfile
import uuid
import shutil
import time
from typing import Dict


# ----------------------------
# Configuration
# ----------------------------

EXEC_TIMEOUT = 3          # seconds
MAX_OUTPUT_SIZE = 5000    # characters

COMPILERS = {
    "c": "gcc",
    "cpp": "g++",
    "csharp": "csc",
}


# ----------------------------
# Public Router Function
# ----------------------------

def run_code(code: str, language: str) -> Dict:
    language = language.lower()

    if language == "c":
        return _run_c_cpp(code, "c")

    elif language == "cpp":
        return _run_c_cpp(code, "cpp")

    elif language == "python":
        return _run_python(code)

    elif language == "java":
        return _run_java(code)

    elif language == "csharp":
        return _run_csharp(code)

    elif language == "javascript":
        return _run_javascript(code)

    else:
        return _error_response(f"Unsupported language: {language}")


# ----------------------------
# C / C++
# ----------------------------

def _run_c_cpp(code: str, lang: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())

    extension = ".c" if lang == "c" else ".cpp"
    compiler = COMPILERS[lang]

    source_file = os.path.join(temp_dir, unique_id + extension)
    executable = os.path.join(temp_dir, unique_id)

    try:
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(code)

        compile_process = subprocess.run(
            [compiler, source_file, "-o", executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if compile_process.returncode != 0:
            return {
                "compiled": False,
                "compile_error": truncate_output(compile_process.stderr),
                "runtime_output": None,
                "runtime_error": None,
                "execution_time": None
            }

        start_time = time.time()

        run_process = subprocess.run(
            [executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=EXEC_TIMEOUT,
        )

        execution_time = round(time.time() - start_time, 4)

        return {
            "compiled": True,
            "compile_error": None,
            "runtime_output": truncate_output(run_process.stdout),
            "runtime_error": truncate_output(run_process.stderr) if run_process.stderr else None,
            "execution_time": execution_time
        }

    except subprocess.TimeoutExpired:
        return _timeout_response()

    except Exception as e:
        return _error_response(str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# Python
# ----------------------------

def _run_python(code: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "script.py")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        start_time = time.time()

        run_process = subprocess.run(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=EXEC_TIMEOUT,
        )

        execution_time = round(time.time() - start_time, 4)

        return {
            "compiled": True,
            "compile_error": None,
            "runtime_output": truncate_output(run_process.stdout),
            "runtime_error": truncate_output(run_process.stderr) if run_process.stderr else None,
            "execution_time": execution_time
        }

    except subprocess.TimeoutExpired:
        return _timeout_response()

    except Exception as e:
        return _error_response(str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# Java
# ----------------------------

def _run_java(code: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "Main.java")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        compile_process = subprocess.run(
            ["javac", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if compile_process.returncode != 0:
            return {
                "compiled": False,
                "compile_error": truncate_output(compile_process.stderr),
                "runtime_output": None,
                "runtime_error": None,
                "execution_time": None
            }

        start_time = time.time()

        run_process = subprocess.run(
            ["java", "-cp", temp_dir, "Main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=EXEC_TIMEOUT,
        )

        execution_time = round(time.time() - start_time, 4)

        return {
            "compiled": True,
            "compile_error": None,
            "runtime_output": truncate_output(run_process.stdout),
            "runtime_error": truncate_output(run_process.stderr) if run_process.stderr else None,
            "execution_time": execution_time
        }

    except subprocess.TimeoutExpired:
        return _timeout_response()

    except Exception as e:
        return _error_response(str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# C#
# ----------------------------

def _run_csharp(code: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "Program.cs")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        compile_process = subprocess.run(
            ["csc", file_path],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if compile_process.returncode != 0:
            return {
                "compiled": False,
                "compile_error": truncate_output(compile_process.stderr),
                "runtime_output": None,
                "runtime_error": None,
                "execution_time": None
            }

        start_time = time.time()

        run_process = subprocess.run(
            [os.path.join(temp_dir, "Program.exe")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=EXEC_TIMEOUT,
        )

        execution_time = round(time.time() - start_time, 4)

        return {
            "compiled": True,
            "compile_error": None,
            "runtime_output": truncate_output(run_process.stdout),
            "runtime_error": truncate_output(run_process.stderr) if run_process.stderr else None,
            "execution_time": execution_time
        }

    except subprocess.TimeoutExpired:
        return _timeout_response()

    except Exception as e:
        return _error_response(str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# JavaScript
# ----------------------------

def _run_javascript(code: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "script.js")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        start_time = time.time()

        run_process = subprocess.run(
            ["node", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=EXEC_TIMEOUT,
        )

        execution_time = round(time.time() - start_time, 4)

        return {
            "compiled": True,
            "compile_error": None,
            "runtime_output": truncate_output(run_process.stdout),
            "runtime_error": truncate_output(run_process.stderr) if run_process.stderr else None,
            "execution_time": execution_time
        }

    except subprocess.TimeoutExpired:
        return _timeout_response()

    except Exception as e:
        return _error_response(str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# Helpers
# ----------------------------

def truncate_output(output: str) -> str:
    if not output:
        return output

    if len(output) > MAX_OUTPUT_SIZE:
        return output[:MAX_OUTPUT_SIZE] + "\n\n[Output truncated]"

    return output


def _timeout_response() -> Dict:
    return {
        "compiled": True,
        "compile_error": None,
        "runtime_output": None,
        "runtime_error": "Execution timed out (possible infinite loop).",
        "execution_time": None
    }


def _error_response(message: str) -> Dict:
    return {
        "compiled": False,
        "compile_error": f"Sandbox failure: {message}",
        "runtime_output": None,
        "runtime_error": None,
        "execution_time": None
    }