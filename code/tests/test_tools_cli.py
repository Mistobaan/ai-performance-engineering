from __future__ import annotations

import os
import sys
from types import SimpleNamespace


def test_tool_launch_injects_repo_root_pythonpath(monkeypatch):
    import core.tools.tools_commands as tools_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools_commands.subprocess, "run", fake_run)

    exit_code = tools_commands._run_tool("kv-cache", ["--help"])

    assert exit_code == 0
    assert recorded["cmd"] == [
        sys.executable,
        str(tools_commands.TOOLS["kv-cache"].script_path.resolve()),
        "--help",
    ]
    assert str(tools_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)


def test_tool_launch_prefers_module_name_when_available(monkeypatch):
    import core.tools.tools_commands as tools_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools_commands.subprocess, "run", fake_run)

    exit_code = tools_commands._run_tool("kernel-verification", ["--help"])

    assert exit_code == 0
    assert recorded["cmd"] == [
        sys.executable,
        "-m",
        "ch20.kernel_verification_tool",
        "--help",
    ]
    assert str(tools_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)
