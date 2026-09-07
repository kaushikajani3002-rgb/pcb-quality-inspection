import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# Setup Python sys.path so we can import modules from project root
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
for mod in [
    "src.utils.json_loader", "src.utils.template_manager", "src.utils.logger",
    "src.mock.mock_results", "src.inspection.inspection_engine", "src.inspection.position_checker",
    "src.ai.detection_engine", "src.ai.model_manager"
]:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

from src.utils.config_loader import ConfigLoader
from src.utils.template_manager import TemplateManager
from src.utils.logger import logger
from src.utils.report_exporter import ReportExporterFactory
from src.utils.validators import ImageValidator, ConfigurationValidator
from src.utils.helper import Helper
from src.utils.constants import (
    STATUS_PASS, STATUS_FAIL, 
    STATE_IDLE, STATE_PROCESSING, STATE_COMPLETED, STATE_ERROR,
    COLOR_PASS, COLOR_FAIL
)
from src.mock.mock_results import MockInspectionService
from src.inspection.inspection_engine import InspectionEngine
from src.ai.detection_engine import (
    load_model, run_component_inspection, run_circuit_inspection,
    build_inventory_table, compute_dashboard_metrics
)
from src.ai.model_manager import ModelManager

# -----------------------------------------------------------------------------
# PAGE SETUP & STYLING (AOI Optical Inspection Suite Industrial Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AOI Optical Inspection Suite - SMT LINE 02",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injector for High-Precision AOI Theme
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">

<style>
    /* Industrial Theme Global Palette */
    :root {
        --bg-bright: #faf8ff;
        --surface-lowest: #ffffff;
        --surface-low: #f2f3ff;
        --surface-container: #eaedff;
        --surface-high: #e2e7ff;
        --on-surface: #131b2e;
        --on-surface-variant: #3f4850;
        --outline: #707881;
        --outline-variant: #bfc7d2;
        --primary: #006194;
        --primary-container: #cce5ff;
        --on-primary: #ffffff;
        --secondary: #006c49;
        --secondary-container: #6cf8bb;
        --error: #ba1a1a;
        --error-container: #ffdad6;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #faf8ff !important;
        color: #131b2e !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #bfc7d2 !important;
    }

    h1, h2, h3, h4, h5, h6, .headline-font {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    .telemetry-font, code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Industrial Strip */
    .top-header-strip {
        background-color: #ffffff;
        border-bottom: 1px solid #bfc7d2;
        padding: 12px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(19, 27, 46, 0.05);
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #bfc7d2;
        background-color: #f2f3ff;
        color: #131b2e;
    }

    .pill-primary {
        background-color: #cce5ff;
        color: #004b73;
        border-color: #93ccff;
    }

    .pill-secondary {
        background-color: #6cf8bb;
        color: #004d33;
        border-color: #4edea3;
    }

    .pill-error {
        background-color: #ffdad6;
        color: #93000a;
        border-color: #ba1a1a;
    }

    /* Industrial Card Containers */
    .aoi-card {
        background-color: #ffffff;
        border: 1px solid #bfc7d2;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(19, 27, 46, 0.05);
        margin-bottom: 16px;
    }

    .aoi-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        color: #131b2e;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #eaedff;
        padding-bottom: 8px;
    }

    /* Status Badges & Pills */
    .badge-ready {
        background-color: #6ffbbe;
        color: #002113;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }

    .badge-armed {
        background-color: #cce5ff;
        color: #004b73;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #93ccff;
    }

    .badge-error {
        background-color: #ffdad6;
        color: #93000a;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #ba1a1a;
    }

    /* Metric KPI Cards */
    .metric-kpi-card {
        background-color: #ffffff;
        border: 1px solid #bfc7d2;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(19, 27, 46, 0.04);
    }
    
    .metric-kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
        color: #006194;
    }

    .metric-kpi-lbl {
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #707881;
        letter-spacing: 0.05em;
    }

    /* Streamlit Widget Custom Styling */
    .stButton>button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase !important;
        transition: all 0.15s ease !important;
    }

    .stButton>button[kind="primary"] {
        background-color: #006c49 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0, 108, 73, 0.2) !important;
    }

    .stButton>button[kind="primary"]:hover {
        background-color: #005236 !important;
        transform: translateY(-1px);
    }

    /* Station Routing Links */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 6px;
        color: #3f4850;
        font-weight: 500;
        font-size: 13px;
        margin-bottom: 4px;
        text-decoration: none;
    }

    .nav-item.active {
        background-color: #cce5ff;
        color: #004b73;
        font-weight: 700;
        border-left: 4px solid #006194;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE STATE MACHINE
# -----------------------------------------------------------------------------
if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = STATE_IDLE
if "current_pcb_template" not in st.session_state:
    st.session_state.current_pcb_template = {}
if "comp_results" not in st.session_state:
    st.session_state.comp_results = None
if "circ_results" not in st.session_state:
    st.session_state.circ_results = None
if "error_message" not in st.session_state:
    st.session_state.error_message = ""
if "selected_template" not in st.session_state:
    st.session_state.selected_template = "Arduino Uno"
if "last_comp_file_name" not in st.session_state:
    st.session_state.last_comp_file_name = None
if "last_circ_file_name" not in st.session_state:
    st.session_state.last_circ_file_name = None

# Load configurations
try:
    config = ConfigLoader()
    template_manager = TemplateManager()
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

def on_template_change():
    new_template_lbl = st.session_state.temp_select_key
    st.session_state.selected_template = new_template_lbl
    # Clear old inspection results
    st.session_state.comp_results = None
    st.session_state.circ_results = None
    st.session_state.workflow_status = STATE_IDLE
    logger.info(f"PCB Profile Template changed to: {new_template_lbl}. Old results cleared.")

# Resolve directories
report_dir = config.get_resolved_path("report_folder")
output_dir = config.get_resolved_path("output_folder")

# Ensure required folders exist
report_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & STATION ROUTING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background-color: #006194; border-radius: 8px; padding: 14px; text-align: center; margin-bottom: 16px; color: #ffffff;">
        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; letter-spacing: 0.05em;">🔬 AOI OPTICAL INSPECTION</span><br>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; opacity: 0.85;">SMT LINE 02 // STATION 04</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Station Routing Navigation
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #707881; text-transform: uppercase; margin-bottom: 8px;">STATION ROUTING</div>
    <div class="nav-item active"><span class="material-symbols-outlined" style="font-size:18px;">view_in_ar</span> Inspection Console</div>
    <div class="nav-item"><span class="material-symbols-outlined" style="font-size:18px;">insights</span> Defect Analytics & Pareto</div>
    <div class="nav-item"><span class="material-symbols-outlined" style="font-size:18px;">videocam</span> Live Telemetry & Cameras</div>
    <div class="nav-item"><span class="material-symbols-outlined" style="font-size:18px;">tune</span> Recipe & Threshold Params</div>
    <div class="nav-item"><span class="material-symbols-outlined" style="font-size:18px;">center_focus_strong</span> Calibration & Optics</div>
    <div class="nav-item"><span class="material-symbols-outlined" style="font-size:18px;">history_edu</span> Audit Trail & Export</div>
    <hr style="border: 0; border-top: 1px solid #bfc7d2; margin: 12px 0;">
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;'>OPERATOR CONTROLS</h4>", unsafe_allow_html=True)
    
    # 1. Device Selection Dropdown
    device_options = {
        "Arduino Uno": "arduino_uno",
        "ESP32 DevKit": "esp32_devkit",
        "STM32 Blue Pill": "stm32_blue_pill",
        "Generic PCB": "generic_pcb"
    }
    
    try:
        template_idx = list(device_options.keys()).index(st.session_state.selected_template)
    except ValueError:
        template_idx = 0

    selected_device_lbl = st.selectbox(
        "PCB Profile Template",
        options=list(device_options.keys()),
        index=template_idx,
        key="temp_select_key",
        on_change=on_template_change,
        help="Loads physical measurements and expected component positions."
    )
    selected_template_stem = device_options[selected_device_lbl]
    st.session_state.selected_template = selected_device_lbl
    
    # 2. Parameters Sliders
    conf_threshold = st.slider(
        "Confidence Gate", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.confidence", 0.50)),
        step=0.05,
        help="YOLO model classification score cut-off."
    )
    
    iou_threshold = st.slider(
        "IoU Overlap Gate", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.iou", 0.45)),
        step=0.05,
        help="Non-Maximum Suppression (NMS) bounding boxes intersection slider."
    )

    position_tolerance = st.slider(
        "Positional Tolerance (mm)",
        min_value=0.2,
        max_value=5.0,
        value=float(config.get("inspection.position_tolerance", 1.5)),
        step=0.1,
        help="Spatial tolerance threshold for PCB placement validation."
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: JetBrains Mono; font-size: 10px; font-weight: 700; color: #707881; text-transform: uppercase;'>INFERENCE ENGINES STATUS</div>", unsafe_allow_html=True)
    
    # Lazy loading models through ModelManager
    model_manager = ModelManager()
    
    comp_model = None
    defect_model = None
    comp_ready = False
    def_ready = False
    comp_err_msg = ""
    def_err_msg = ""
    
    # Resolve defect model mapping name
    defect_mapping = config.get("models.defect_mapping") or {}
    mapping_entry = defect_mapping.get(selected_template_stem)
    if isinstance(mapping_entry, dict):
        selected_defect_model_name = mapping_entry.get("name", "Defect Detector")
    else:
        selected_defect_model_name = mapping_entry or "Defect Detector"

    try:
        comp_model = model_manager.get_component_model()
        comp_ready = True
    except Exception as ex:
        comp_err_msg = str(ex)
        logger.error(f"Failed to load Component Model: {ex}")

    try:
        defect_model = model_manager.get_defect_model(selected_template_stem)
        def_ready = True
    except Exception as ex:
        def_err_msg = str(ex)
        logger.error(f"Failed to load Defect Model for template {selected_template_stem}: {ex}")

    # Display status badges
    if comp_ready:
        st.markdown("<div style='margin-top:6px;'><span class='badge-ready'>YOLOv11-Edge [Parts] READY</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='margin-top:6px;'><span class='badge-error'>Component Model Error</span><br><small style='color:#ba1a1a;'>{comp_err_msg}</small></div>", unsafe_allow_html=True)

    if def_ready:
        st.markdown(f"<div style='margin-top:6px;'><span class='badge-armed'>{selected_defect_model_name} [Defects] ARMED</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='margin-top:6px;'><span class='badge-error'>Defect Model Error</span><br><small style='color:#ba1a1a;'>{def_err_msg}</small></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: JetBrains Mono; font-size: 10px; font-weight: 700; color: #707881; text-transform: uppercase;'>SIMULATION & FLAGS</div>", unsafe_allow_html=True)

    solder_bridge_flag = st.checkbox("Force Solder Bridge Flagging", value=True)
    submicron_reticle_flag = st.checkbox("Render Sub-Micron Precision Reticle", value=True)

    defect_mode = st.checkbox(
        "Force Anomaly/Defect Mode", 
        value=True,
        help="Toggles simulated inspection errors (missing, misaligned, cracks)."
    )

    debug_mode = st.checkbox(
        "Enable Inference Debug Mode", 
        value=True,
        help="Enables display of raw YOLO outputs, matching matrices, and hardware specs."
    )

    operator_name = st.text_input(
        "Operator ID", 
        value=config.get("dashboard.default_operator", "Operator_AOI_04")
    )

    # PLC Conveyor indicator
    st.markdown("""
    <div style="background-color: #f2f3ff; border: 1px solid #bfc7d2; border-radius: 6px; padding: 8px 12px; margin-top: 12px; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;">PLC CONVEYOR</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #006c49;">SYNCED 100%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_run, col_reset = st.columns(2)
    with col_run:
        run_clicked = st.button("▶ RUN AOI", use_container_width=True, type="primary")
    with col_reset:
        reset_clicked = st.button("🔄 RESET", use_container_width=True)

# Handle Reset Click
if reset_clicked:
    st.session_state.workflow_status = STATE_IDLE
    st.session_state.comp_results = None
    st.session_state.circ_results = None
    st.session_state.error_message = ""
    st.session_state.last_comp_file_name = None
    st.session_state.last_circ_file_name = None
    logger.info("Inspection system reset.")
    st.rerun()

# Load Selected Template Profile
template = template_manager.load_template(selected_template_stem)
if not template:
    st.error(f"Error loading template config for {selected_device_lbl}")
    st.stop()
st.session_state.current_pcb_template = template

# -----------------------------------------------------------------------------
# TOP INDUSTRIAL HEADER STRIP & METADATA PILLS
# -----------------------------------------------------------------------------
board_dims = template.get("board_dimensions", {})
w_mm = board_dims.get("width_mm", "68.6")
h_mm = board_dims.get("height_mm", "53.4")
critical_comps = [c["id"] for c in template.get("components", [])]

st.markdown(f"""
<div class="top-header-strip flex flex-wrap items-center justify-between gap-4">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 10px; height: 10px; border-radius: 50%; background-color: #006c49; display: inline-block;"></span>
            <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; margin: 0; text-transform: uppercase; color: #131b2e;">
                AOI Optical Inspection Suite <span style="color: #707881; font-weight: 400; font-size: 15px;">// SMT LINE 02 // BAY 4</span>
            </h2>
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="pill-badge pill-secondary">CUDA GPU: 60 FPS (12.8ms)</span>
            <span class="pill-badge pill-primary">DEVICE: {selected_device_lbl}</span>
            <span class="pill-badge">DIM: {w_mm} × {h_mm} mm</span>
            <span class="pill-badge">PARTS: {len(critical_comps)} Nom</span>
            <span class="pill-badge pill-primary">IPC-A-610G CLASS 3</span>
            <span class="pill-badge">LOT: #B84-9021</span>
            <span class="pill-badge pill-error">E-STOP: ARMED</span>
            <span class="pill-badge">SHIFT A</span>
            <span class="pill-badge pill-secondary">OPERATOR: {operator_name}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DUAL OPTICAL CHANNELS (IMAGE ACQUISITION PANEL)
# -----------------------------------------------------------------------------
st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700; text-transform: uppercase; margin-bottom: 12px;'>📸 Image Acquisition & Dual Optical Channels</h3>", unsafe_allow_html=True)
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #bfc7d2; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 700; color: #006194; display: flex; justify-content: space-between; align-items: center;">
            <span>CH-01: RGB COAXIAL HIGH-RES FEED</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; background-color: #6cf8bb; color: #004d33; padding: 2px 6px; border-radius: 4px;">2048×1536 // 5500K</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    comp_file = st.file_uploader(
        "Upload Component Image (CH-01)", 
        type=["png", "jpg", "jpeg"],
        key="comp_uploader",
        help="Image optimized for component presence and position validation."
    )
    
    if comp_file:
        if st.session_state.last_comp_file_name != comp_file.name:
            st.session_state.comp_results = None
            st.session_state.last_comp_file_name = comp_file.name
            st.session_state.workflow_status = STATE_IDLE
        st.success("✓ CH-01 Optical Feed Captured")
        st.image(comp_file, width=220, caption="CH-01 Optical Frame")
    else:
        if st.session_state.last_comp_file_name is not None:
            st.session_state.comp_results = None
            st.session_state.last_comp_file_name = None
            st.session_state.workflow_status = STATE_IDLE
        st.info("ℹ CH-01 Standby: Awaiting Optical Capture Upload")

with col_upload2:
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #bfc7d2; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 700; color: #006c49; display: flex; justify-content: space-between; align-items: center;">
            <span>CH-02: TELECENTRIC SOLDER MASK FEED</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; background-color: #cce5ff; color: #004b73; padding: 2px 6px; border-radius: 4px;">IR RING 850nm // POLARIZED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    circ_file = st.file_uploader(
        "Upload Circuit/Defect Image (CH-02)", 
        type=["png", "jpg", "jpeg"],
        key="circ_uploader",
        help="Image optimized for solder joint defects and trace fracture inspection."
    )
    
    if circ_file:
        if st.session_state.last_circ_file_name != circ_file.name:
            st.session_state.circ_results = None
            st.session_state.last_circ_file_name = circ_file.name
            st.session_state.workflow_status = STATE_IDLE
        st.success("✓ CH-02 Optical Feed Captured")
        st.image(circ_file, width=220, caption="CH-02 Telecentric Frame")
    else:
        if st.session_state.last_circ_file_name is not None:
            st.session_state.circ_results = None
            st.session_state.last_circ_file_name = None
            st.session_state.workflow_status = STATE_IDLE
        st.info("ℹ CH-02 Standby: Awaiting Telecentric Capture Upload")

st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 16px 0;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INSPECTION EXECUTION TRIGGER
# -----------------------------------------------------------------------------
if run_clicked:
    if not comp_file and not circ_file:
        st.error("Please upload at least one image to run inspection.")
        st.stop()
        
    st.session_state.workflow_status = STATE_PROCESSING
    st.session_state.error_message = ""
    
    # Validate sliders
    val_ok, val_msg = ConfigurationValidator.validate_inspection_parameters(
        conf_threshold, iou_threshold, position_tolerance
    )
    if not val_ok:
        st.session_state.workflow_status = STATE_ERROR
        st.session_state.error_message = f"Slider Config Error: {val_msg}"
    else:
        # Process Component Image if uploaded
        if comp_file:
            logger.info("Executing component inspection pipeline...")
            meta_ok, meta_msg = ImageValidator.validate_file_metadata(comp_file.name, comp_file.size)
            if not meta_ok:
                st.session_state.workflow_status = STATE_ERROR
                st.session_state.error_message = f"Component Image Error: {meta_msg}"
            else:
                try:
                    comp_results = run_component_inspection(
                        uploaded_image=comp_file,
                        component_model=comp_model,
                        conf_slider=conf_threshold,
                        iou_slider=iou_threshold,
                        active_template=template,
                        position_tolerance_slider=position_tolerance
                    )
                    st.session_state.comp_results = comp_results
                except Exception as ex:
                    st.session_state.workflow_status = STATE_ERROR
                    st.session_state.error_message = f"Component Inspection Crash: {ex}"
                    logger.error(st.session_state.error_message, exc_info=True)
        else:
            st.session_state.comp_results = {
                "status": "NOT_INSPECTED",
                "reason": "Component image not uploaded"
            }
            
        # Process Circuit Image if uploaded
        if circ_file:
            logger.info("Executing circuit defect inspection pipeline...")
            meta_ok, meta_msg = ImageValidator.validate_file_metadata(circ_file.name, circ_file.size)
            if not meta_ok:
                st.session_state.workflow_status = STATE_ERROR
                st.session_state.error_message = f"Circuit Image Error: {meta_msg}"
            else:
                try:
                    temp_matched_dets = st.session_state.comp_results.get("detected_components", []) if st.session_state.comp_results else []
                    circ_results = run_circuit_inspection(
                        uploaded_image=circ_file,
                        defect_model=defect_model,
                        conf_slider=conf_threshold,
                        iou_slider=iou_threshold,
                        defect_mode=defect_mode,
                        matched_detections=temp_matched_dets
                    )
                    st.session_state.circ_results = circ_results
                except Exception as ex:
                    st.session_state.workflow_status = STATE_ERROR
                    st.session_state.error_message = f"Circuit Inspection Crash: {ex}"
                    logger.error(st.session_state.error_message, exc_info=True)
        else:
            st.session_state.circ_results = {
                "status": "NOT_INSPECTED",
                "reason": "Circuit image not uploaded"
            }
            
        if st.session_state.workflow_status != STATE_ERROR:
            st.session_state.workflow_status = STATE_COMPLETED

# -----------------------------------------------------------------------------
# FINAL RESULTS & AGGREGATOR VIEW
# -----------------------------------------------------------------------------
if st.session_state.workflow_status == STATE_ERROR:
    st.error(st.session_state.error_message)

if st.session_state.workflow_status == STATE_COMPLETED:
    comp_res = st.session_state.comp_results
    circ_res = st.session_state.circ_results
    
    comp_status = comp_res.get("status") if comp_res else "NOT_INSPECTED"
    circ_status = circ_res.get("status") if circ_res else "NOT_INSPECTED"
    
    # 1. FINAL RESULT AGGREGATOR BANNER
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700; text-transform: uppercase;'>⚡ Aggregated System Inspection Status</h3>", unsafe_allow_html=True)
    
    if comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "PASS":
        st.markdown("""
        <div style="background-color: #6cf8bb; border: 2px solid #006c49; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px;">
            <span style="color: #002113; font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 1px;">🟢 COMPONENT DETECTION COMPLETE & CIRCUIT PASSED</span><br>
            <span style="color: #004d33; font-family: 'JetBrains Mono', monospace; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>PASS</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "FAIL":
        st.markdown("""
        <div style="background-color: #ffdad6; border: 2px solid #ba1a1a; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px;">
            <span style="color: #93000a; font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 1px;">🔴 CIRCUIT DEFECT DETECTED</span><br>
            <span style="color: #ba1a1a; font-family: 'JetBrains Mono', monospace; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>FAIL</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "NOT_INSPECTED":
        st.markdown("""
        <div style="background-color: #6cf8bb; border: 2px solid #006c49; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px;">
            <span style="color: #002113; font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 1px;">🟢 COMPONENT DETECTION COMPLETE</span><br>
            <span style="color: #004d33; font-family: 'JetBrains Mono', monospace; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>NOT INSPECTED</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status == "NOT_INSPECTED" and circ_status == "PASS":
        st.markdown("""
        <div style="background-color: #6cf8bb; border: 2px solid #006c49; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px;">
            <span style="color: #002113; font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 1px;">🟢 CIRCUIT INSPECTION PASSED</span><br>
            <span style="color: #004d33; font-family: 'JetBrains Mono', monospace; font-size: 13px;">Component Detection: <b>NOT INSPECTED</b> | Circuit Inspection: <b>PASS</b></span>
        </div>
        """, unsafe_allow_html=True)

    elif comp_status == "NOT_INSPECTED" and circ_status == "FAIL":
        st.markdown("""
        <div style="background-color: #ffdad6; border: 2px solid #ba1a1a; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px;">
            <span style="color: #93000a; font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 1px;">🔴 CIRCUIT DEFECT DETECTED</span><br>
            <span style="color: #ba1a1a; font-family: 'JetBrains Mono', monospace; font-size: 13px;">Component Detection: <b>NOT INSPECTED</b> | Circuit Inspection: <b>FAIL</b></span>
        </div>
        """, unsafe_allow_html=True)

    # 2. METROLOGY KPI CARDS ROW
    if comp_status in ("PASS", "DETECTION_COMPLETE"):
        detected_comps = comp_res.get("detected_components", [])
        detected_counts = comp_res.get("detected_counts", {})
        total_detected = len(detected_comps)
        unique_types = len(detected_counts)
        total_conf_sum = sum(float(d.get("confidence", 0.0)) for d in detected_comps)
        avg_conf_pct = (total_conf_sum / total_detected * 100.0) if total_detected > 0 else 0.0
        verdict_str = "PASSED" if circ_status != "FAIL" else "DEFECT FLAGGED"
        verdict_color = "#006c49" if circ_status != "FAIL" else "#ba1a1a"

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-kpi-card">
                <div class="metric-kpi-lbl">TOTAL COMPONENTS</div>
                <div class="metric-kpi-val">{total_detected}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-kpi-card">
                <div class="metric-kpi-lbl">UNIQUE TYPES</div>
                <div class="metric-kpi-val" style="color:#006194;">{unique_types}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-kpi-card">
                <div class="metric-kpi-lbl">MEAN CONFIDENCE</div>
                <div class="metric-kpi-val" style="color:#006c49;">{avg_conf_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-kpi-card">
                <div class="metric-kpi-lbl">VERDICT STATUS</div>
                <div class="metric-kpi-val" style="color:{verdict_color}; font-size:20px;">{verdict_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # 3. SPATIAL RETICLE INSPECTION STAGE (VISUAL INSPECTION)
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #bfc7d2; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: #131b2e; display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined text-primary">filter_center_focus</span> SPATIAL RETICLE INSPECTION STAGE
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background-color: #e2e7ff; color: #006194; padding: 4px 8px; border-radius: 4px; font-weight: 700;">
                SCALE: 1.000px = 33.5μm // TELECENTRIC 1:1
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700; margin-bottom: 8px;'>COMPONENT OPTICAL RETICLE (CH-01)</h4>", unsafe_allow_html=True)
        if comp_status in ("PASS", "DETECTION_COMPLETE"):
            tab_comp_orig, tab_comp_box = st.tabs(["Original Feed", "Annotated Reticle Overlays"])
            with tab_comp_orig:
                st.image(comp_res["original_image"], use_container_width=True)
            with tab_comp_box:
                st.image(comp_res["annotated_image"], use_container_width=True)
        else:
            st.info("Component inspection feed offline.")
            
    with col_vis2:
        st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700; margin-bottom: 8px;'>CIRCUIT TELECENTRIC RETICLE (CH-02)</h4>", unsafe_allow_html=True)
        if circ_status in ("PASS", "FAIL"):
            tab_circ_orig, tab_circ_box = st.tabs(["Original Feed", "Defect Overlays"])
            with tab_circ_orig:
                st.image(circ_res["original_image"], use_container_width=True)
            with tab_circ_box:
                st.image(circ_res["annotated_image"], use_container_width=True)
        else:
            st.info("Circuit inspection feed offline.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. VERIFICATION REGISTERS
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700; text-transform: uppercase;'>📋 Metrology Verification Registers</h3>", unsafe_allow_html=True)
    tab_comp_reg, tab_circ_reg = st.tabs(["COMPONENT INVENTORY LEDGER", "CIRCUIT DEFECT LEDGER"])
    
    def format_class_name(raw_name: str) -> str:
        if not raw_name:
            return "Unknown"
        s = str(raw_name).strip()
        if s.lower() == "ic":
            return "IC"
        elif s.lower() == "led":
            return "LED"
        elif s.lower() == "pcb":
            return "PCB"
        else:
            return s.capitalize()

    with tab_comp_reg:
        if comp_status in ("PASS", "DETECTION_COMPLETE"):
            detected_comps = comp_res.get("detected_components", [])
            detected_counts = comp_res.get("detected_counts", {})
            
            total_detected = len(detected_comps)
            unique_types = len(detected_counts)
            
            col_inv_left, col_inv_right = st.columns([1, 1.3])
            
            with col_inv_left:
                st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700;'>TYPE BREAKDOWN SUMMARY</h4>", unsafe_allow_html=True)
                if detected_counts:
                    summary_rows = []
                    running_total = 0
                    for ctype, count in sorted(detected_counts.items(), key=lambda x: x[1], reverse=True):
                        summary_rows.append({
                            "Component Type": format_class_name(ctype),
                            "Count": count
                        })
                        running_total += count
                    
                    summary_rows.append({
                        "Component Type": "TOTAL DETECTED",
                        "Count": running_total
                    })
                    
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No component types detected.")
                    
            with col_inv_right:
                st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700;'>ANNOTATED SPATIAL OVERLAY</h4>", unsafe_allow_html=True)
                if "annotated_image" in comp_res:
                    st.image(comp_res["annotated_image"], use_container_width=True, caption=f"Annotated PCB Component Overlays ({total_detected} Bounding Boxes)")
                else:
                    st.info("Visual overlay not available.")
                    
            st.markdown("---")
            st.markdown("<h4 style='font-family: Space Grotesk; font-weight: 700;'>COMPONENT METROLOGY DETECTIONS TABLE</h4>", unsafe_allow_html=True)
            if detected_comps:
                sorted_dets = sorted(detected_comps, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                details_rows = []
                for i, d in enumerate(sorted_dets, start=1):
                    details_rows.append({
                        "#": i,
                        "Component": format_class_name(d.get("class_name", d.get("type", "Unknown"))),
                        "Class ID": d.get("class_id", "N/A"),
                        "Confidence": f"{float(d.get('confidence', 0.0))*100.0:.1f}%",
                        "X1": f"{float(d.get('x1', 0.0)):.1f}",
                        "Y1": f"{float(d.get('y1', 0.0)):.1f}",
                        "X2": f"{float(d.get('x2', 0.0)):.1f}",
                        "Y2": f"{float(d.get('y2', 0.0)):.1f}",
                        "Center (x%, y%)": f"({float(d.get('center_x_pct', 0.0)):.2f}%, {float(d.get('center_y_pct', 0.0)):.2f}%)"
                    })
                st.dataframe(pd.DataFrame(details_rows), use_container_width=True, hide_index=True)
            else:
                st.success("No components detected.")
        else:
            st.info("Please upload a PCB component image to generate component inventory.")
            
    with tab_circ_reg:
        if circ_status in ("PASS", "FAIL"):
            circ_rows = []
            for d in circ_res.get("defects", []):
                circ_rows.append({
                    "Defect ID": d["id"],
                    "Defect Type": d["class_name"],
                    "Confidence": f"{d['confidence']*100:.1f}%",
                    "Location (x%, y%)": f"({d['center_x_pct']:.2f}, {d['center_y_pct']:.2f})",
                    "Bounding Box Size (w% x h%)": f"{d['width_pct']:.2f}% x {d['height_pct']:.2f}%",
                    "Severity Status": d["severity"]
                })
                
            if circ_rows:
                st.dataframe(pd.DataFrame(circ_rows), use_container_width=True, hide_index=True)
            else:
                st.success("Zero defect markers flagged in the active verification register.")
        else:
            st.info("Circuit defect register not available. Image was not uploaded.")

    # 5. INDUSTRIAL FOOTER EXPORTER BAR
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700; text-transform: uppercase; margin-top: 24px;'>📤 Export Inspection Logs & QC Certificates</h3>", unsafe_allow_html=True)
    
    export_payload = {
        "status": "FAIL" if (comp_status == "FAIL" or circ_status == "FAIL") else ("PASS" if comp_status == "PASS" and circ_status == "PASS" else "INCOMPLETE"),
        "inspection_date": Helper.get_current_timestamp(),
        "operator": operator_name,
        "template_name": template.get("board_name", selected_device_lbl),
        "processing_time": comp_res.get("processing_time", "0.000 sec") if comp_status != "NOT_INSPECTED" else circ_res.get("processing_time", "0.000 sec"),
        "component_statistics": comp_res.get("component_statistics", {"total_expected": len(template.get("components", [])), "total_detected": 0, "by_type": {}}) if comp_status != "NOT_INSPECTED" else {},
        "missing": comp_res.get("missing", []) if comp_status != "NOT_INSPECTED" else [],
        "misaligned": comp_res.get("misaligned", []) if comp_status != "NOT_INSPECTED" else [],
        "cracks": circ_res.get("defects", []) if circ_status != "NOT_INSPECTED" else [],
        "extra": comp_res.get("extra", []) if comp_status != "NOT_INSPECTED" else []
    }
    
    col_pdf, col_csv, col_json = st.columns(3)
    log_stamp = Helper.get_log_timestamp()
    
    # PDF export
    pdf_filename = f"report_{selected_template_stem}_{log_stamp}.pdf"
    pdf_path = report_dir / pdf_filename
    pdf_exporter = ReportExporterFactory.get_exporter("pdf")
    pdf_success = pdf_exporter.export(export_payload, pdf_path)
    
    with col_pdf:
        if pdf_success and pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download QC Certificate (PDF)",
                    data=f.read(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("📄 PDF Exporter Offline", disabled=True, use_container_width=True)

    # CSV export
    csv_filename = f"report_{selected_template_stem}_{log_stamp}.csv"
    csv_path = report_dir / csv_filename
    csv_exporter = ReportExporterFactory.get_exporter("csv")
    csv_success = csv_exporter.export(export_payload, csv_path)
    
    with col_csv:
        if csv_success and csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📊 Download CSV Discrepancies Ledger",
                    data=f.read(),
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.button("📊 CSV Exporter Offline", disabled=True, use_container_width=True)

    # JSON export
    json_filename = f"report_{selected_template_stem}_{log_stamp}.json"
    json_path = report_dir / json_filename
    json_exporter = ReportExporterFactory.get_exporter("json")
    json_success = json_exporter.export(export_payload, json_path)
    
    with col_json:
        if json_success and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="💻 Download JSON Telemetry Log",
                    data=f.read(),
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.button("💻 JSON Exporter Offline", disabled=True, use_container_width=True)

    # 6. INFERENCE DEBUG CONSOLE
    if debug_mode:
        st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>🛠️ Inference Telemetry & Debug Console</h3>", unsafe_allow_html=True)
        with st.expander("Show Complete Backend & AI Model Trace Log", expanded=False):
            st.json({
                "component_inspection_debug": comp_res.get("debug_info") if comp_res else "NOT RUN",
                "circuit_inspection_debug": circ_res if circ_res else "NOT RUN"
            })
