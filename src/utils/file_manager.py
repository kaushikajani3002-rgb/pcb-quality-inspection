import os
from pathlib import Path
from typing import List

class FileManager:
    """
    Utility class to manage directories and file creation.
    Gives assurance that folder footprints exist before files are written.
    """
    @staticmethod
    def ensure_directories_exist(base_dir: Path, folders: List[str]) -> None:
        """
        Creates directory paths if they do not exist.
        """
        for folder in folders:
            dir_path = base_dir / folder
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def initialize_system_paths() -> None:
        """
        Gives a baseline initialization of standard directories.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        required_dirs = [
            "configs",
            "templates",
            "outputs/reports",
            "outputs/predictions",
            "outputs/visualizations",
            "outputs/exported_reports",
            "logs",
            "assets",
            "docs",
            "src/mock",
            "src/inspection"
        ]
        FileManager.ensure_directories_exist(project_root, required_dirs)
