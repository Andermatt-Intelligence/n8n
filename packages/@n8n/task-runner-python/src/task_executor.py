import multiprocessing
import traceback
import textwrap
import json
import io
import os
import re
import subprocess
import sys
import logging
import tempfile
import shutil

from src.errors import (
    TaskCancelledError,
    TaskKilledError,
    TaskResultMissingError,
    TaskResultReadError,
    TaskRuntimeError,
    TaskTimeoutError,
    TaskSubprocessFailedError,
    SecurityViolationError,
)
from src.import_validation import validate_module_import
from src.config.security_config import SecurityConfig

from src.message_types.broker import NodeMode, Items, Query
from src.message_types.pipe import (
    PipeResultMessage,
    PipeErrorMessage,
    TaskErrorInfo,
    PrintArgs,
)
from src.pipe_reader import PipeReader
from src.constants import (
    EXECUTOR_CIRCULAR_REFERENCE_KEY,
    EXECUTOR_USER_OUTPUT_KEY,
    EXECUTOR_ALL_ITEMS_FILENAME,
    EXECUTOR_PER_ITEM_FILENAME,
    SIGTERM_EXIT_CODE,
    SIGKILL_EXIT_CODE,
    PIPE_MSG_PREFIX_LENGTH,
    LOG_PIPE_READER_TIMEOUT_TRIGGERED,
)

from multiprocessing.context import ForkServerProcess
from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)

MULTIPROCESSING_CONTEXT = multiprocessing.get_context("forkserver")
MAX_PRINT_ARGS_ALLOWED = 100

# PEP 723 inline script metadata pattern (official reference implementation)
# See: https://peps.python.org/pep-0723/#reference-implementation
PEP723_REGEX = (
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
PEP723_PATTERN = re.compile(PEP723_REGEX)

type PipeConnection = Connection


def has_pep723_metadata(code: str) -> bool:
    """Check if code contains PEP 723 inline script metadata."""
    return any(m.group("type") == "script" for m in PEP723_PATTERN.finditer(code))


def is_uv_available() -> bool:
    """Check if UV is available on the system."""
    return shutil.which("uv") is not None


def extract_pep723_block(code: str) -> str:
    """
    Extract the raw PEP 723 script block from code.

    Returns the full block including delimiters, or empty string if not found.
    """
    matches = [
        m for m in re.finditer(PEP723_PATTERN, code) if m.group("type") == "script"
    ]
    if matches:
        return matches[0].group(0)
    return ""


def generate_uv_wrapper_script(
    user_code: str,
    node_mode: str,
    items_json: str,
    query_json: str,
) -> str:
    """
    Generate a wrapper script for UV execution that preserves PEP 723 metadata.

    The wrapper:
    1. Preserves the user's PEP 723 metadata block (for dependencies)
    2. Loads items from an environment variable (JSON)
    3. Captures stdout/stderr from user code
    4. Returns results as JSON on stdout
    """
    # Extract PEP 723 metadata from user code (preserve it at the top)
    pep723_block = extract_pep723_block(user_code)

    # Remove PEP 723 block from user code (we'll put it at the top of wrapper)
    user_code_without_metadata = PEP723_PATTERN.sub("", user_code).strip()

    # Indent user code for the function body
    indented_user_code = textwrap.indent(user_code_without_metadata, "    ")

    wrapper = f'''{pep723_block}
# Auto-generated wrapper for n8n Python task execution
import json
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr

# Load items from environment variable
_items_json = os.environ.get("N8N_ITEMS", "[]")
_query_json = os.environ.get("N8N_QUERY", "null")

_items = json.loads(_items_json)
_item = _items[0] if _items else {{}}
_query = json.loads(_query_json)

_user_stdout = io.StringIO()
_user_stderr = io.StringIO()


def _user_function():
{indented_user_code}


def _format_result(result, node_mode):
    """Format the result based on node mode."""
    if result is None:
        return []

    if node_mode == "per_item":
        # Per-item mode expects results to be a list of items
        if isinstance(result, list):
            return result
        return [result]

    # all_items mode - wrap result appropriately
    if isinstance(result, list):
        return result

    return [{{"json": result}}]


try:
    with redirect_stdout(_user_stdout), redirect_stderr(_user_stderr):
        _result = _user_function()

    _formatted_result = _format_result(_result, "{node_mode}")

    print(json.dumps({{
        "ok": True,
        "result": _formatted_result,
        "print_args": _user_stdout.getvalue().splitlines(),
        "stderr": _user_stderr.getvalue(),
    }}, default=str))

except Exception as _e:
    import traceback
    print(json.dumps({{
        "ok": False,
        "error": {{
            "message": str(_e),
            "description": "",
            "stack": traceback.format_exc(),
            "stderr": _user_stderr.getvalue(),
        }},
        "print_args": _user_stdout.getvalue().splitlines(),
    }}, default=str))
    sys.exit(1)
'''
    return wrapper


class UvExecutor:
    """Handles execution of Python code via UV with PEP 723 support."""

    @staticmethod
    def create_process(
        code: str,
        node_mode: str,
        items: list,
        query,
        default_deps: list[str],
        task_timeout: int,
    ) -> tuple[subprocess.Popen, str]:
        """
        Create a UV subprocess for executing a Python code task.

        Returns the process and the path to the temp script (for cleanup).
        """
        items_json = json.dumps(items, default=str, ensure_ascii=False)
        query_json = json.dumps(query, default=str, ensure_ascii=False)

        wrapper_script = generate_uv_wrapper_script(
            user_code=code,
            node_mode=node_mode,
            items_json=items_json,
            query_json=query_json,
        )

        # Write wrapper script to a temp file
        fd, script_path = tempfile.mkstemp(suffix=".py", prefix="n8n_uv_task_")
        try:
            os.write(fd, wrapper_script.encode("utf-8"))
        finally:
            os.close(fd)

        # Build UV command
        cmd = [
            "uv",
            "-qq",  # Silent mode
            "--no-progress",
            "run",
            "--no-project",  # Don't use any pyproject.toml
            "--no-config",  # Don't use any uv.toml
        ]

        # Add default dependencies via --with flag
        for dep in default_deps:
            cmd.extend(["--with", dep])

        cmd.append(script_path)

        # Set up environment with items data
        env = os.environ.copy()
        env["N8N_ITEMS"] = items_json
        env["N8N_QUERY"] = query_json

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        return process, script_path

    @staticmethod
    def execute_process(
        process: subprocess.Popen,
        script_path: str,
        task_timeout: int,
    ) -> tuple[list, list[list[str]], int]:
        """
        Execute a UV subprocess and return the results.

        Returns: (result_items, print_args, result_size_bytes)
        """
        try:
            stdout, stderr = process.communicate(timeout=task_timeout)

            if process.returncode == -15:  # SIGTERM
                raise TaskCancelledError()

            if process.returncode != 0:
                # Try to parse error from stdout (our wrapper outputs JSON even on error)
                try:
                    response = json.loads(stdout)
                    if not response.get("ok"):
                        error = response.get("error", {})
                        raise TaskRuntimeError(error)
                except json.JSONDecodeError:
                    # Fallback to stderr
                    raise TaskRuntimeError(
                        {
                            "message": f"UV execution failed with exit code {process.returncode}",
                            "description": stderr,
                            "stack": "",
                            "stderr": stderr,
                        }
                    )

            # Parse successful response
            response = json.loads(stdout)

            if not response.get("ok"):
                error = response.get("error", {})
                raise TaskRuntimeError(error)

            result = response.get("result", [])
            print_args = [[line] for line in response.get("print_args", [])]
            result_size_bytes = len(stdout.encode("utf-8"))

            return result, print_args, result_size_bytes

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TaskTimeoutError(task_timeout)

        finally:
            # Clean up temp script
            try:
                os.unlink(script_path)
            except OSError:
                pass

    @staticmethod
    def stop_process(process: subprocess.Popen | None):
        """Stop a running UV subprocess."""
        if process is None:
            return

        try:
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except (ProcessLookupError, OSError):
            pass


class TaskExecutor:
    """Responsible for executing Python code tasks in isolated subprocesses."""

    @staticmethod
    def create_process(
        code: str,
        node_mode: NodeMode,
        items: Items,
        security_config: SecurityConfig,
        query: Query = None,
    ) -> tuple[ForkServerProcess, PipeConnection, PipeConnection]:
        """Create a subprocess for executing a Python code task and a pipe for communication."""

        fn = (
            TaskExecutor._all_items
            if node_mode == "all_items"
            else TaskExecutor._per_item
        )

        # thread in runner process reads, subprocess writes
        read_conn, write_conn = MULTIPROCESSING_CONTEXT.Pipe(duplex=False)

        process = MULTIPROCESSING_CONTEXT.Process(
            target=fn,
            args=(
                code,
                items,
                write_conn,
                security_config,
                query,
            ),
        )

        return process, read_conn, write_conn

    @staticmethod
    def execute_process(
        process: ForkServerProcess,
        read_conn: PipeConnection,
        write_conn: PipeConnection,
        task_timeout: int,
        pipe_reader_timeout: float,
        continue_on_fail: bool,
    ) -> tuple[Items, PrintArgs, int]:
        """Execute a subprocess for a Python code task."""

        print_args: PrintArgs = []

        pipe_reader = PipeReader(read_conn.fileno(), read_conn)
        pipe_reader.start()

        try:
            try:
                process.start()
            except Exception as e:
                raise TaskSubprocessFailedError(-1, e)
            finally:
                write_conn.close()

            process.join(timeout=task_timeout)

            if process.is_alive():
                TaskExecutor.stop_process(process)
                raise TaskTimeoutError(task_timeout)

            if process.exitcode == SIGTERM_EXIT_CODE:
                raise TaskCancelledError()

            if process.exitcode == SIGKILL_EXIT_CODE:
                raise TaskKilledError()

            if process.exitcode != 0:
                assert process.exitcode is not None
                raise TaskSubprocessFailedError(process.exitcode)

            pipe_reader.join(timeout=pipe_reader_timeout)

            if pipe_reader.is_alive():
                logger.warning(
                    LOG_PIPE_READER_TIMEOUT_TRIGGERED.format(
                        timeout=pipe_reader_timeout
                    )
                )
                try:
                    read_conn.close()
                except Exception:
                    pass

            if pipe_reader.error:
                raise TaskResultReadError(pipe_reader.error)

            if pipe_reader.pipe_message is None:
                raise TaskResultMissingError()

            returned = pipe_reader.pipe_message

            if "error" in returned:
                raise TaskRuntimeError(returned["error"])

            if "result" not in returned:
                raise TaskResultMissingError()

            result = returned["result"]
            print_args = returned.get("print_args", [])
            assert pipe_reader.message_size is not None
            result_size_bytes = pipe_reader.message_size

            return result, print_args, result_size_bytes

        except Exception as e:
            if continue_on_fail:
                return [{"json": {"error": str(e)}}], print_args, 0
            raise

    @staticmethod
    def stop_process(process: ForkServerProcess | None):
        """Stop a running subprocess, gracefully else force-killing."""

        if process is None or not process.is_alive():
            return

        try:
            process.terminate()
            process.join(timeout=1)  # 1s grace period

            if process.is_alive():
                process.kill()
                process.join()
        except (ProcessLookupError, ConnectionError, BrokenPipeError):
            # subprocess is dead or unreachable
            pass

    @staticmethod
    def _all_items(
        raw_code: str,
        items: Items,
        write_conn,
        security_config: SecurityConfig,
        query: Query = None,
    ):
        """Execute a Python code task in all-items mode."""

        if security_config.runner_env_deny:
            os.environ.clear()

        TaskExecutor._sanitize_sys_modules(security_config)

        print_args: PrintArgs = []
        sys.stderr = stderr_capture = io.StringIO()

        try:
            wrapped_code = TaskExecutor._wrap_code(raw_code)
            compiled_code = compile(wrapped_code, EXECUTOR_ALL_ITEMS_FILENAME, "exec")

            globals = {
                "__builtins__": TaskExecutor._filter_builtins(security_config),
                "_items": items,
                "_query": query,
                "print": TaskExecutor._create_custom_print(print_args),
            }

            exec(compiled_code, globals)

            result = globals[EXECUTOR_USER_OUTPUT_KEY]
            TaskExecutor._put_result(write_conn.fileno(), result, print_args)

        except BaseException as e:
            TaskExecutor._put_error(
                write_conn.fileno(), e, stderr_capture.getvalue(), print_args
            )

    @staticmethod
    def _per_item(
        raw_code: str,
        items: Items,
        write_conn,
        security_config: SecurityConfig,
        _query: Query = None,  # unused, only to keep signatures consistent across modes
    ):
        """Execute a Python code task in per-item mode."""

        if security_config.runner_env_deny:
            os.environ.clear()

        TaskExecutor._sanitize_sys_modules(security_config)

        print_args: PrintArgs = []
        sys.stderr = stderr_capture = io.StringIO()

        try:
            wrapped_code = TaskExecutor._wrap_code(raw_code)
            compiled_code = compile(wrapped_code, EXECUTOR_PER_ITEM_FILENAME, "exec")

            filtered_builtins = TaskExecutor._filter_builtins(security_config)
            custom_print = TaskExecutor._create_custom_print(print_args)

            result: Items = []
            for index, item in enumerate(items):
                globals = {
                    "__builtins__": filtered_builtins,
                    "_item": item,
                    "print": custom_print,
                }

                exec(compiled_code, globals)

                user_output = globals[EXECUTOR_USER_OUTPUT_KEY]

                if user_output is None:
                    continue

                json_data = TaskExecutor._extract_json_data_per_item(user_output)

                output_item = {"json": json_data, "pairedItem": {"item": index}}

                if isinstance(user_output, dict) and "binary" in user_output:
                    output_item["binary"] = user_output["binary"]

                result.append(output_item)

            TaskExecutor._put_result(write_conn.fileno(), result, print_args)

        except BaseException as e:
            TaskExecutor._put_error(
                write_conn.fileno(), e, stderr_capture.getvalue(), print_args
            )

    @staticmethod
    def _wrap_code(raw_code: str) -> str:
        indented_code = textwrap.indent(raw_code, "    ")
        return f"def _user_function():\n{indented_code}\n\n{EXECUTOR_USER_OUTPUT_KEY} = _user_function()"

    @staticmethod
    def _extract_json_data_per_item(user_output):
        if not isinstance(user_output, dict):
            return user_output

        if "json" in user_output:
            return user_output["json"]

        if "binary" in user_output:
            return {k: v for k, v in user_output.items() if k != "binary"}

        return user_output

    @staticmethod
    def _put_result(write_fd: int, result: Items, print_args: PrintArgs):
        message: PipeResultMessage = {
            "result": result,
            "print_args": TaskExecutor._truncate_print_args(print_args),
        }

        data = json.dumps(message, default=str, ensure_ascii=False).encode("utf-8")
        length_bytes = len(data).to_bytes(PIPE_MSG_PREFIX_LENGTH, "big")

        try:
            TaskExecutor._write_bytes(write_fd, length_bytes)
            TaskExecutor._write_bytes(write_fd, data)
        finally:
            try:
                os.close(write_fd)
            except Exception:
                pass

    @staticmethod
    def _put_error(
        write_fd: int,
        e: BaseException,
        stderr: str = "",
        print_args: PrintArgs | None = None,
    ):
        if print_args is None:
            print_args = []

        task_error_info: TaskErrorInfo = {
            "message": f"Process exited with code {e.code}"
            if isinstance(e, SystemExit)
            else str(e),
            "description": getattr(e, "description", ""),
            "stack": traceback.format_exc(),
            "stderr": stderr,
        }

        message: PipeErrorMessage = {
            "error": task_error_info,
            "print_args": TaskExecutor._truncate_print_args(print_args),
        }

        data = json.dumps(message, default=str, ensure_ascii=False).encode("utf-8")
        length_bytes = len(data).to_bytes(PIPE_MSG_PREFIX_LENGTH, "big")

        try:
            TaskExecutor._write_bytes(write_fd, length_bytes)
            TaskExecutor._write_bytes(write_fd, data)
        finally:
            try:
                os.close(write_fd)
            except Exception:
                pass

    # ========== print() ==========

    @staticmethod
    def _create_custom_print(print_args: PrintArgs):
        def custom_print(*args):
            serializable_args = []

            for arg in args:
                try:
                    json.dumps(arg, default=str, ensure_ascii=False)
                    serializable_args.append(arg)
                except Exception as _:
                    # Ensure args are serializable so they are transmissible
                    # through the multiprocessing queue and via websockets.
                    serializable_args.append(
                        {
                            EXECUTOR_CIRCULAR_REFERENCE_KEY: repr(arg),
                            "__type__": type(arg).__name__,
                        }
                    )

            formatted = TaskExecutor._format_print_args(*serializable_args)
            print_args.append(formatted)
            print("[user code]", *args)

        return custom_print

    @staticmethod
    def _format_print_args(*args) -> list[str]:
        """
        Takes the args passed to a `print()` call in user code and converts them
        to string representations suitable for display in a browser console.

        Expects all args to be serializable.
        """

        formatted = []

        for arg in args:
            if isinstance(arg, str):
                formatted.append(f"'{arg}'")

            elif arg is None or isinstance(arg, (int, float, bool)):
                formatted.append(str(arg))

            elif isinstance(arg, dict) and EXECUTOR_CIRCULAR_REFERENCE_KEY in arg:
                formatted.append(f"[Circular {arg.get('__type__', 'Object')}]")

            else:
                formatted.append(json.dumps(arg, default=str, ensure_ascii=False))

        return formatted

    @staticmethod
    def _truncate_print_args(print_args: PrintArgs) -> PrintArgs:
        """Truncate print_args to prevent pipe buffer overflow."""

        if not print_args or len(print_args) <= MAX_PRINT_ARGS_ALLOWED:
            return print_args

        truncated = print_args[:MAX_PRINT_ARGS_ALLOWED]
        truncated.append(
            [
                f"[Output truncated - {len(print_args) - MAX_PRINT_ARGS_ALLOWED} more print statements]"
            ]
        )

        return truncated

    # ========== security ==========

    @staticmethod
    def _filter_builtins(security_config: SecurityConfig):
        """Get __builtins__ with denied ones removed."""

        if len(security_config.builtins_deny) == 0:
            filtered = dict(__builtins__)
        else:
            filtered = {
                k: v
                for k, v in __builtins__.items()
                if k not in security_config.builtins_deny
            }

        filtered["__import__"] = TaskExecutor._create_safe_import(security_config)

        return filtered

    @staticmethod
    def _sanitize_sys_modules(security_config: SecurityConfig):
        safe_modules = {
            "builtins",
            "__main__",
            "sys",
            "traceback",
            "linecache",
            "importlib",
            "importlib.machinery",
        }

        if "*" in security_config.stdlib_allow:
            safe_modules.update(sys.stdlib_module_names)
        else:
            safe_modules.update(security_config.stdlib_allow)

        if "*" in security_config.external_allow:
            safe_modules.update(
                name
                for name in sys.modules.keys()
                if name not in sys.stdlib_module_names
            )
        else:
            safe_modules.update(security_config.external_allow)

        # keep modules marked as safe and submodules of those
        safe_prefixes = [safe + "." for safe in safe_modules]
        modules_to_remove = [
            name
            for name in sys.modules.keys()
            if name not in safe_modules
            and not any(name.startswith(prefix) for prefix in safe_prefixes)
        ]

        for module_name in modules_to_remove:
            del sys.modules[module_name]

    @staticmethod
    def _create_safe_import(security_config: SecurityConfig):
        original_import = __builtins__["__import__"]

        def safe_import(name, *args, **kwargs):
            is_allowed, error_msg = validate_module_import(name, security_config)

            if not is_allowed:
                assert error_msg is not None
                raise SecurityViolationError(
                    message="Security violation detected",
                    description=error_msg,
                )

            return original_import(name, *args, **kwargs)

        return safe_import

    # ========== pipe I/O ==========

    @staticmethod
    def _write_bytes(fd: int, data: bytes):
        total_written = 0
        while total_written < len(data):
            written = os.write(fd, data[total_written:])
            if written == 0:
                raise OSError("Write failed")
            total_written += written
