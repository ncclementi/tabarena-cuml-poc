"""Utility functions for collecting benchmark environment metadata."""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import git
import psutil


def get_system_info() -> dict:
    """Collect OS, architecture, CPU count, total RAM, user, hostname."""
    try:
        user = getpass.getuser()
    except Exception:
        try:
            user = os.getlogin()
        except OSError:
            user = None

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "user": user,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
    }


def get_cuda_info() -> dict:
    """Collect CUDA device count, device names, CUDA version."""
    try:
        from numba import cuda

        device_count = len(cuda.gpus)
        device_names = [cuda.gpus[i].name.decode() for i in range(device_count)]
        cuda_version = ".".join(map(str, cuda.runtime.get_version()))

        return {
            "cuda_available": True,
            "cuda_device_count": device_count,
            "cuda_device_names": json.dumps(device_names),
            "cuda_version": cuda_version,
        }
    except Exception as e:
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_names": json.dumps([]),
            "cuda_version": None,
            "cuda_error": str(e),
        }


def get_python_info() -> dict:
    """Collect Python version and pip freeze snapshot."""
    # Get pip freeze output
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        pip_freeze = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        pip_freeze = None

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "pip_freeze": pip_freeze,
    }


def get_git_info(search_path: Path | None = None) -> dict:
    """Collect commit SHA and dirty status of the current repository."""
    try:
        repo = git.Repo(search_path, search_parent_directories=True)
        return {
            "git_commit": str(repo.head.commit),
            "git_branch": str(repo.active_branch) if not repo.head.is_detached else None,
            "git_dirty": repo.is_dirty(),
            "git_working_dir": str(repo.working_tree_dir),
        }
    except Exception as e:
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "git_working_dir": None,
            "git_error": str(e),
        }


def collect_all_metadata(search_path: Path | None = None) -> dict:
    """Aggregate all metadata with execution timestamp.

    Parameters
    ----------
    search_path : Path | None
        Optional path to search for git repository. If None, uses current directory.

    Returns
    -------
    dict
        Dictionary containing all collected metadata with prefixed keys.
    """
    metadata = {
        "execution_datetime": datetime.now().isoformat(),
        "execution_timestamp": datetime.now().timestamp(),
    }

    # Add system info with prefix
    for key, value in get_system_info().items():
        metadata[f"system.{key}"] = value

    # Add CUDA info with prefix
    for key, value in get_cuda_info().items():
        metadata[f"cuda.{key}"] = value

    # Add Python info with prefix
    for key, value in get_python_info().items():
        metadata[f"python.{key}"] = value

    # Add git info with prefix
    for key, value in get_git_info(search_path).items():
        metadata[f"git.{key}"] = value

    return metadata
