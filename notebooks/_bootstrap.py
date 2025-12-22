from pathlib import Path
import sys

def setup_project_path():
    """Macht scr/Python als Importpfad verfügbar (Binder + lokal)."""
    project_root = Path.cwd().resolve()
    while project_root != project_root.parent and not (project_root / "scr").exists():
        project_root = project_root.parent

    if not (project_root / "scr").exists():
        raise RuntimeError("Projektroot mit 'scr/' nicht gefunden.")

    sys.path.insert(0, str(project_root / "scr" / "Python"))
