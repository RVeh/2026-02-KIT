from pathlib import Path
import sys


def setup_project_path():
    """
    Sorgt dafür, dass scr/Python im Python-Suchpfad liegt.
    Funktioniert lokal und in Binder.
    """
    project_root = Path.cwd().resolve()
    while project_root != project_root.parent and not (project_root / "scr").exists():
        project_root = project_root.parent

    if not (project_root / "scr").exists():
        raise RuntimeError("Projektroot mit 'scr/' nicht gefunden.")

    python_path = project_root / "scr" / "Python"
    sys.path.insert(0, str(python_path))
