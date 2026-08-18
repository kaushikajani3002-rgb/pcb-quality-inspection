import os
import sys
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO
from src.utils.logger import logger
from src.utils.config_loader import ConfigLoader

# Fallback module-level cache for non-streamlit environments
_global_model_cache: Dict[str, YOLO] = {}

class ModelManager:
    """
    Centralized Model Manager to handle lazy loading and caching of YOLO models.
    Supports Streamlit session state and standalone scripts.
    """
    def __init__(self):
        self.config = ConfigLoader()

    def _get_cache(self) -> Dict[str, YOLO]:
        """
        Retrieves the appropriate cache depending on runtime environment.
        """
        if "streamlit" in sys.modules:
            import streamlit as st
            if "_model_cache" not in st.session_state:
                st.session_state["_model_cache"] = {}
            return st.session_state["_model_cache"]
        return _global_model_cache

    def get_component_model(self) -> YOLO:
        """
        Loads and caches the common component detection model.
        """
        comp_config = self.config.get("models.component_model")
        if not isinstance(comp_config, dict):
            # Fallback to old flat config structure if any
            path_rel = self.config.get("models.trained.Component", "models/trained/component_detector_best.pt")
            name = "Component"
        else:
            name = comp_config.get("name", "Component")
            path_rel = comp_config.get("path", "models/trained/component_detector_best.pt")

        return self._load_and_cache(name, path_rel, "Common Component Model")

    def get_defect_model(self, profile: str) -> YOLO:
        """
        Loads and caches the profile-specific defect inspection model.
        """
        defect_mapping = self.config.get("models.defect_mapping")
        if not defect_mapping or not isinstance(defect_mapping, dict):
            raise KeyError("models.defect_mapping configuration is missing or invalid.")

        profile_config = defect_mapping.get(profile)
        if not profile_config:
            raise KeyError(f"No defect model mapping found for profile: '{profile}'")

        if isinstance(profile_config, dict):
            name = profile_config.get("name", "UnknownDefectModel")
            path_rel = profile_config.get("path")
        else:
            # Fallback for flat string config
            name = profile_config
            path_rel = self.config.get(f"models.trained.{name}")

        if not name or not path_rel:
            raise ValueError(f"Incomplete defect model configuration for profile '{profile}'")

        return self._load_and_cache(name, path_rel, f"Profile '{profile}' defect detection")

    def _load_and_cache(self, name: str, path_rel: str, context: str) -> YOLO:
        """
        Internal helper to load a YOLO model or return its cached instance.
        """
        cache = self._get_cache()
        if name in cache and cache[name] is not None:
            logger.info(f"Reusing cached YOLO model '{name}' for {context}")
            return cache[name]

        # Resolve path (support relative project path or absolute fallback path)
        resolved_path = self.config.project_root / path_rel
        if not resolved_path.exists():
            abs_path = Path(path_rel)
            fallback_dataset_path = Path(r"D:\PCB\Dataset\Component_best.pt")
            if abs_path.exists():
                resolved_path = abs_path
            elif fallback_dataset_path.exists() and name.lower() == "component":
                resolved_path = fallback_dataset_path
            else:
                error_msg = f"Model '{name}' not found for context '{context}'. Expected weights file at: {resolved_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        try:
            import time
            import torch
            logger.info(f"MODEL LOAD START: Loading YOLO model '{name}' from disk: {resolved_path}")
            start_time = time.time()
            model = YOLO(str(resolved_path))
            end_time = time.time()
            load_duration = end_time - start_time
            
            # Check device and CUDA availability
            cuda_available = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
            model_device = getattr(model, "device", "Unknown")
            
            logger.info(f"MODEL LOAD END: Successfully loaded '{name}' in {load_duration:.4f} seconds.")
            logger.info(f"  - Device: {model_device}")
            logger.info(f"  - CUDA Available: {cuda_available} (Device Name: {device_name})")
            model_classes_count = len(model.names) if hasattr(model, "names") else 0
            logger.info(f"  - Model Classes Count: {model_classes_count}")
            
            cache[name] = model
            return model
        except Exception as e:
            error_msg = f"Failed to initialize YOLO model '{name}' from {resolved_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
