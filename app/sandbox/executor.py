"""代码沙箱：抽象基类 + 受限子进程实现 + Docker 执行器占位。

SubprocessExecutor 安全模型（macOS 无 Docker 的降级方案）：
- 在临时目录内执行，数据文件以绝对路径白名单方式只读可用
- 静态 import 黑名单（os.remove / shutil / socket / subprocess 等）
- resource 模块设置 CPU 时间 / 内存 / 文件大小软限制
- 硬超时后 kill 整个进程组

DockerExecutor 预留接口，未来有 Docker 环境后实现隔离执行。
"""
from __future__ import annotations

import ast
import base64
import logging
import os
import resource
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    images: list[str] = field(default_factory=list)  # base64 编码的 PNG
    success: bool = False
    timed_out: bool = False
    error: str = ""


class CodeExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        code: str,
        files: list[str] | None = None,
        timeout: int = 60,
        cpu_seconds: int = 30,
        memory_mb: int = 1024,
    ) -> ExecutionResult:
        """执行代码。files 为允许只读访问的数据文件绝对路径白名单。"""
        raise NotImplementedError


# ---------- 静态安全检查 ----------

_FORBIDDEN_MODULES = {
    "os", "sys", "shutil", "subprocess", "socket", "requests", "urllib",
    "http", "ftplib", "ctypes", "signal", "multiprocessing", "pathlib",
    "importlib", "pickle", "marshal", "tempfile",
}
# os 被整体禁用，但数据分析脚本通常只需要 pandas/matplotlib/sklearn 等科学计算库
_ALLOWED_IMPORT_ROOTS = {
    "pandas", "numpy", "matplotlib", "openpyxl", "sklearn", "scipy",
    "json", "math", "statistics",
    "datetime", "collections", "itertools", "re", "csv", "io", "functools",
    "data_files",  # 沙箱注入的数据清单模块（DATA_FILES / FILES / OUT_DIR）
}


def static_check(code: str) -> list[str]:
    """静态检查 import，返回违规列表（空列表 = 通过）。"""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES or root not in _ALLOWED_IMPORT_ROOTS:
                    violations.append(f"禁止 import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_MODULES or root not in _ALLOWED_IMPORT_ROOTS:
                    violations.append(f"禁止 from ... import: {node.module}")
    return violations


# ---------- 子进程实现 ----------


class SubprocessExecutor(CodeExecutor):
    """受限子进程沙箱。data 文件以绝对路径传入，脚本内直接用该路径读取。"""

    def execute(
        self,
        code: str,
        files: list[str] | None = None,
        timeout: int = 60,
        cpu_seconds: int = 30,
        memory_mb: int = 1024,
    ) -> ExecutionResult:
        violations = static_check(code)
        if violations:
            return ExecutionResult(success=False, error="; ".join(violations))

        files = [str(Path(f).resolve()) for f in (files or [])]
        for f in files:
            if not Path(f).is_file():
                return ExecutionResult(success=False, error=f"数据文件不存在: {f}")

        workdir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        outdir = workdir / "out"
        outdir.mkdir()
        script_path = workdir / "analysis.py"
        script_path.write_text(code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
            "SANDBOX_OUT_DIR": str(outdir),
            # 数据文件路径通过环境变量白名单注入，脚本从 os.environ 读不到 os 模块——
            # 所以这里改用生成一个 data_files.py 供脚本 import。
        }
        # 生成数据清单模块：DATA_FILES 以文件名（basename）为键，杜绝模型猜 FILES 下标
        manifest = workdir / "data_files.py"
        manifest.write_text(
            "FILES = " + repr(list(files)) + "\n"
            "DATA_FILES = " + repr({Path(f).name: f for f in files}) + "\n"
            "OUT_DIR = " + repr(str(outdir)) + "\n",
            encoding="utf-8",
        )

        def _limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
            mem = memory_mb * 1024 * 1024
            try:
                # macOS 不允许把 RLIMIT_AS 降到低于父进程当前 soft limit(INF)，
                # 失败时跳过内存限制，其余限制仍然生效。
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # 便于整组 kill
                preexec_fn=_limits,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
                return ExecutionResult(
                    stdout=proc.stdout or "",
                    stderr="",
                    success=False,
                    timed_out=True,
                    error=f"执行超过 {timeout}s，已终止",
                )
        except Exception as exc:
            return ExecutionResult(success=False, error=f"沙箱启动失败: {exc}")

        images: list[str] = []
        for png in sorted(outdir.glob("*.png")):
            images.append(base64.b64encode(png.read_bytes()).decode())

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            images=images,
            success=proc.returncode == 0,
            error="" if proc.returncode == 0 else f"脚本退出码 {proc.returncode}",
        )


class DockerExecutor(CodeExecutor):
    """Docker 容器沙箱（占位）。

    未来实现要点：
    - 使用 gVisor/Firecracker 或至少 --network none --read-only --cap-drop ALL
    - 数据目录以只读 volume 挂载到 /data
    - 资源限制用 --memory / --cpus，超时由调用方控制
    """

    def execute(self, code, files=None, timeout=60, cpu_seconds=30, memory_mb=1024) -> ExecutionResult:
        raise NotImplementedError(
            "DockerExecutor 尚未实现：本机无 Docker 环境。"
            "请使用 SubprocessExecutor，或在有 Docker 的环境实现此类。"
        )


_default_executor: CodeExecutor | None = None


def get_executor() -> CodeExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = SubprocessExecutor()
    return _default_executor
