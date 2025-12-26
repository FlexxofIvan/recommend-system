from pathlib import Path

from dvc.api import DVCFileSystem
from hydra.utils import get_original_cwd


def ensure_data_available(rel_path: Path):
    project_root = Path(get_original_cwd())
    fs = DVCFileSystem(project_root)

    abs_path = project_root / rel_path
    if abs_path.exists() and any(abs_path.iterdir()):
        return

    fs.get(str(rel_path), str(abs_path), recursive=True)
