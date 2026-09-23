# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import setup_env


def test_activate_prefers_installed_runtime_over_toolchain(tmp_path: Path) -> None:
    setup = object.__new__(setup_env.Setup)
    setup.workspace = tmp_path
    setup.venv_dir = tmp_path / ".venv"
    (setup.venv_dir / "bin").mkdir(parents=True)
    (setup.venv_dir / "bin" / "activate").write_text("", encoding="utf-8")

    setup.write_activate_local("/install", ("/install/lib", "/toolchain/lib"))

    lines = (
        (setup.venv_dir / "bin" / "activate.local")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    exports = [line for line in lines if "export LD_LIBRARY_PATH=" in line]
    assert "/toolchain/lib" in exports[0]
    assert "/install/lib" in exports[1]
    assert exports[1].startswith("    *) export LD_LIBRARY_PATH=/install/lib")
