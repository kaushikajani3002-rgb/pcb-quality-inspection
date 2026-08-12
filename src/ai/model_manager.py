"""
ModelManager — Cached Dual-Model Inference Manager for PCB AOI
==============================================================
Provides lazy, on-demand loading and caching of YOLO models.

Two model roles:
    A. Component model   — common to ALL PCB profiles
    B. Defect model      — resolved per PCB profile via configs/model.yaml

Usage:
    manager = ModelManager()
    comp_model = manager.get_component_model()
    defect_model, defect_name = manager.get_defect_model("arduino_uno")
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils.logger import logger


class ModelManager:
    """
    Centralized model loader with an internal cache.

    Models are loaded from disk the first time they are requested
    and reused on subsequent calls.  No model is loaded at __init__
    time (lazy / on-demand).
    """

    def __init__(self):
        from src.utils.config_loader import ConfigLoader
        self._config = ConfigLoader()
        self._cache: Dict[str, Any] = {}          # model_name -> YOLO object
        self._resolved_paths: Dict[str, str] = {}  # model_name -> absolute path

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_component_model(self) -> Any:
        """
        Returns the common component detection YOLO model.
        Loads from disk on first call; returns cached instance afterwards.
        """
        comp_cfg = self._config.get("models.component_model")
        if not comp_cfg or not isinstance(comp_cfg, dict):
            logger.error("Component model not configured in models.component_model")
            return None

        name = comp_cfg.get("name", "Component")
        path_rel = comp_cfg.get("path", "")
        return self._load_cached(name, path_rel)

    def get_defect_model(self, profile_key: str) -> Tuple[Any, Optional[str]]:
        """
        Returns (yolo_model, model_name) for the given PCB profile key.

        Parameters
        ----------
        profile_key : str
            Internal template key, e.g. "arduino_uno", "esp32_devkit".

        Returns
        -------
        (model, name) : Tuple
            model is a YOLO object or None if weights are absent.
            name is the human-readable model name (e.g. "DeepPCB").
        """
        mapping = self._config.get(f"models.defect_mapping.{profile_key}")
        if mapping is None:
            logger.warning(f"No defect model mapping for profile '{profile_key}'.")
            return None, None

        # Support both old (string) and new (dict) config formats
        if isinstance(mapping, dict):
            name = mapping.get("name", profile_key)
            path_rel = mapping.get("path", "")
        else:
            # Backward compat: flat string value (e.g. "DeepPCB")
            name = str(mapping)
            path_rel = self._config.get(f"models.trained.{name}", "")

        model = self._load_cached(name, path_rel)
        return model, name

    def get_component_model_info(self) -> Dict[str, str]:
        """Returns name and path of the component model (without loading it)."""
        comp_cfg = self._config.get("models.component_model") or {}
        name = comp_cfg.get("name", "Component") if isinstance(comp_cfg, dict) else "Component"
        path_rel = comp_cfg.get("path", "") if isinstance(comp_cfg, dict) else ""
        abs_path = str(self._config.project_root / path_rel) if path_rel else ""
        return {"name": name, "path": abs_path}

    def get_defect_model_info(self, profile_key: str) -> Dict[str, str]:
        """Returns name and path of the defect model for a profile (without loading it)."""
        mapping = self._config.get(f"models.defect_mapping.{profile_key}")
        if mapping is None:
            return {"name": None, "path": ""}
        if isinstance(mapping, dict):
            name = mapping.get("name", profile_key)
            path_rel = mapping.get("path", "")
        else:
            name = str(mapping)
            path_rel = self._config.get(f"models.trained.{name}", "")
        abs_path = str(self._config.project_root / path_rel) if path_rel else ""
        return {"name": name, "path": abs_path}

    def is_cached(self, model_name: str) -> bool:
        """Check whether a model is already loaded in the cache."""
        return model_name in self._cache

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _load_cached(self, name: str, path_rel: str) -> Any:
        """Load a YOLO model from disk or return the cached instance."""
        # Return cached model if available
        if name in self._cache:
            logger.debug(f"Model '{name}' served from cache.")
            return self._cache[name]

        if not path_rel:
            logger.error(f"No path configured for model '{name}'.")
            return None

        abs_path = str(self._config.project_root / path_rel)

        if not os.path.exists(abs_path):
            logger.error(f"Model file not found for '{name}' at: {abs_path}")
            return None

        try:
            from ultralytics import YOLO
            model = YOLO(abs_path)
            self._cache[name] = model
            self._resolved_paths[name] = abs_path
            logger.info(f"Loaded and cached '{name}' YOLO model from {abs_path}")
            return model
        except Exception as e:
            logger.error(f"Error initializing YOLO model '{name}' from {abs_path}: {e}")
            return None
