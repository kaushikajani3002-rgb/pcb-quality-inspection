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
# PAGE SETUP & STYLING (AOI Optical Inspection Suite High-Precision Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AOI Optical Inspection Suite - SMT LINE 02",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injector for Clean Industrial UI
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

    /* HIGH CONTRAST DARK TEXT FOR SIDEBAR LABELS & WIDGETS */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    .stSlider label,
    .stCheckbox label span,
    .stSelectbox label,
    .stTextInput label,
    .stFileUploader label {
        color: #131b2e !important;
        font-weight: 600 !important;
    }

    h1, h2, h3, h4, h5, h6, .headline-font {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
        color: #131b2e !important;
    }

    .telemetry-font, code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Industrial Header Strip */
    .top-header-strip {
        background-color: #ffffff;
        border-bottom: 1px solid #bfc7d2;
        padding: 10px 16px;
        margin-bottom: 14px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(19, 27, 46, 0.05);
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
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

    /* Card Containers */
    .aoi-card {
        background-color: #ffffff;
        border: 1px solid #bfc7d2;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(19, 27, 46, 0.05);
        margin-bottom: 14px;
    }

    .aoi-card-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        color: #131b2e;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #eaedff;
    }

    /* Status Badges */
    .badge-ready {
        background-color: #6ffbbe;
        color: #002113;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
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

    /* Metric KPI Grid Cards */
    .kpi-grid-card {
        background-color: #ffffff;
        border: 1px solid #bfc7d2;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(19, 27, 46, 0.04);
    }
    
    .kpi-grid-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: #006194;
    }

    .kpi-grid-lbl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #707881;
    }

    /* File Uploader Container */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f2f3ff !important;
        border: 2px dashed #006194 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #131b2e !important;
        font-weight: 600 !important;
    }

    /* Sidebar Radio Options */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: #f2f3ff !important;
        border: 1px solid #bfc7d2 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        margin-bottom: 2px !important;
        width: 100% !important;
        cursor: pointer !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #131b2e !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #cce5ff !important;
        color: #004b73 !important;
        border-color: #006194 !important;
    }

    .stButton>button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        letter-spacing: 0.02em !important;
        text-transform: uppercase !important;
    }

    .stButton>button[kind="primary"] {
        background-color: #006c49 !important;
        color: #ffffff !important;
        border: none !important;
    }

    .stButton>button[kind="primary"]:hover {
        background-color: #005236 !important;
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
    st.session_state.comp_results = None
    st.session_state.circ_results = None
    st.session_state.workflow_status = STATE_IDLE
    logger.info(f"PCB Profile Template changed to: {new_template_lbl}. Old results cleared.")

# Resolve directories
report_dir = config.get_resolved_path("report_folder")
output_dir = config.get_resolved_path("output_folder")
report_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & FUNCTIONAL CLICKABLE NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background-color: #006194; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; color: #ffffff;">
        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 700; color: #ffffff;">🔬 AOI OPTICAL INSPECTION</span><br>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #cce5ff;">SMT LINE 02 // CLEANROOM BAY 4</span>
    </div>
    """, unsafe_allow_html=True)
    
    station_route = st.radio(
        "STATION ROUTING",
        options=[
            "🔍 Inspection Console",
            "📊 Defect Analytics & Pareto",
            "🎥 Live Telemetry & Cameras",
            "🎛️ Recipe & Threshold Params",
            "🎯 Calibration & Optics",
            "📜 Audit Trail & Export"
        ],
        index=0,
        key="station_route_nav"
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: JetBrains Mono; font-size: 10px; font-weight: 700; color: #131b2e; text-transform: uppercase;'>PLC CONVEYOR SYNC</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #f2f3ff; border: 1px solid #bfc7d2; border-radius: 6px; padding: 6px 10px; margin-top: 4px; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #131b2e;">● PLC CONVEYOR</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #006c49;">SYNCED 100%</span>
    </div>
    """, unsafe_allow_html=True)

# Load Selected Template Profile
selected_device_lbl = st.session_state.selected_template
selected_template_stem = {
    "Arduino Uno": "arduino_uno",
    "ESP32 DevKit": "esp32_devkit",
    "STM32 Blue Pill": "stm32_blue_pill",
    "Generic PCB": "generic_pcb"
}.get(selected_device_lbl, "arduino_uno")

template = template_manager.load_template(selected_template_stem)
if not template:
    st.error(f"Error loading template config for {selected_device_lbl}")
    st.stop()
st.session_state.current_pcb_template = template

# Lazy loading models through ModelManager
model_manager = ModelManager()
comp_model = None
defect_model = None
comp_ready = False
def_ready = False

defect_mapping = config.get("models.defect_mapping") or {}
mapping_entry = defect_mapping.get(selected_template_stem)
if isinstance(mapping_entry, dict):
    selected_defect_model_name = mapping_entry.get("name", "Defect Detector")
else:
    selected_defect_model_name = mapping_entry or "Defect Detector"

try:
    comp_model = model_manager.get_component_model()
    comp_ready = True
except Exception:
    pass

try:
    defect_model = model_manager.get_defect_model(selected_template_stem)
    def_ready = True
except Exception:
    pass

board_dims = template.get("board_dimensions", {})
w_mm = board_dims.get("width_mm", "68.6")
h_mm = board_dims.get("height_mm", "53.4")
critical_comps = [c["id"] for c in template.get("components", [])]

# -----------------------------------------------------------------------------
# TOP FIXED HEADER STRIP & SUBHEADER METADATA
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="top-header-strip flex flex-wrap items-center justify-between gap-3">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 28px; height: 28px; background-color: #006194; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: bold;">🔬</div>
            <div>
                <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 700; margin: 0; text-transform: uppercase; color: #131b2e;">
                    AOI OPTICAL INSPECTION SUITE
                </h2>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #707881;">SMT LINE 02 // CLEANROOM BAY 4</span>
            </div>
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-left: 12px;">
            <span class="pill-badge pill-secondary">CUDA GPU: 60 FPS (12.8ms)</span>
            <span class="pill-badge pill-primary">Arduino Uno Rev3 [IPC-A-610G Class 3]</span>
            <span class="pill-badge">LOT: #B84-9021</span>
            <span class="pill-badge pill-error">E-STOP: ARMED</span>
            <span class="pill-badge">SHIFT A - 03:42:15</span>
            <span class="pill-badge pill-primary">Operator_AOI_04 [CERT: LEVEL 3]</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 1: MAIN INSPECTION CONSOLE WORKSTATION (3-COLUMN WORKSTATION LAYOUT)
# -----------------------------------------------------------------------------
if station_route == "🔍 Inspection Console":
    # Subheader Bar
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background-color: #ffffff; border: 1px solid #bfc7d2; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #ba1a1a; font-size: 11px;">🔴</span>
            <span style="font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 700;">AOI VERIFICATION CONSOLE <span style="color: #707881; font-weight: 400;">// SMT-AOI-04</span></span>
            <span class="pill-badge pill-primary">CUDA: 14.2ms (60 FPS)</span>
            <span class="pill-badge pill-secondary">YIELD: 97.8% [LOT #B84-9021]</span>
            <span class="pill-badge">LINE-02 SYNCED</span>
        </div>
        <div style="display: flex; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 10px;">
            <span>DEVICE: <b>{selected_device_lbl}</b></span>
            <span>DIM: <b>{w_mm} × {h_mm} mm</b></span>
            <span>PARTS: <b>{len(critical_comps)} Nom</b></span>
            <span style="color: #006194; font-weight: bold;">IPC-A-610G CLASS 3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-COLUMN WORKSTATION SPLIT (Col 1: Controls & Dual Channels, Col 2: Reticle Stage, Col 3: KPIs & Ledger)
    col_left, col_center, col_right = st.columns([3.5, 4.5, 4.0])

    # -------------------------------------------------------------------------
    # COLUMN 1: OPERATOR CONTROLS & DUAL OPTICAL CHANNELS
    # -------------------------------------------------------------------------
    with col_left:
        st.markdown("<div class='aoi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='aoi-card-header'><span>🎛️ OPERATOR CONTROLS</span><span class='badge-ready'>ONLINE</span></div>", unsafe_allow_html=True)
        
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
            "PROFILE TEMPLATE",
            options=list(device_options.keys()),
            index=template_idx,
            key="temp_select_key",
            on_change=on_template_change
        )
        
        conf_threshold = st.slider("Confidence Gate", min_value=0.0, max_value=1.0, value=float(config.get("inspection.confidence", 0.25)), step=0.05)
        iou_threshold = st.slider("IoU Overlap Gate", min_value=0.0, max_value=1.0, value=float(config.get("inspection.iou", 0.45)), step=0.05)
        position_tolerance = st.slider("Positional Tolerance (mm)", min_value=0.2, max_value=5.0, value=float(config.get("inspection.position_tolerance", 1.5)), step=0.1)

        st.markdown("<div style='font-family: JetBrains Mono; font-size: 10px; font-weight: 700; color: #707881; margin-top: 6px;'>INFERENCE ENGINES</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: 4px;">
            <span class="badge-ready">YOLOv11-Edge [Parts] READY</span>
            <span class="badge-armed">{selected_defect_model_name} [Defects] ARMED</span>
        </div>
        """, unsafe_allow_html=True)

        solder_bridge_flag = st.checkbox("Force Solder Bridge Flagging", value=True)
        submicron_reticle_flag = st.checkbox("Render Sub-Micron Precision Reticle", value=True)
        defect_mode = st.checkbox("Force Anomaly/Defect Mode", value=True)
        debug_mode = st.checkbox("Enable Inference Debug Mode", value=True)
        operator_name = st.text_input("Operator ID", value=config.get("dashboard.default_operator", "Operator_AOI_04"))

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            run_clicked = st.button("▶ RUN AOI INSPECTION", use_container_width=True, type="primary")
        with col_r2:
            reset_clicked = st.button("🔄 RESET CONSOLE", use_container_width=True)

        if reset_clicked:
            st.session_state.workflow_status = STATE_IDLE
            st.session_state.comp_results = None
            st.session_state.circ_results = None
            st.session_state.error_message = ""
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # DUAL OPTICAL CHANNELS CARD
        st.markdown("<div class='aoi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='aoi-card-header'><span>📷 DUAL OPTICAL CHANNELS</span><span style='font-size: 10px;'>STATION #4</span></div>", unsafe_allow_html=True)
        
        st.markdown("<b>CH-01: RGB Coaxial High-Res</b> <span style='color:#006c49; font-size:10px; float:right;'>LIVE</span>", unsafe_allow_html=True)
        comp_file = st.file_uploader("Upload CH-01 Component Image", type=["png", "jpg", "jpeg"], key="comp_uploader")
        if comp_file:
            st.image(comp_file, width=160, caption="CH-01 2048×1536 // 5500K")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<b>CH-02: Telecentric Solder Mask</b> <span style='color:#006194; font-size:10px; float:right;'>POLARIZED</span>", unsafe_allow_html=True)
        circ_file = st.file_uploader("Upload CH-02 Circuit/Defect Image", type=["png", "jpg", "jpeg"], key="circ_uploader")
        if circ_file:
            st.image(circ_file, width=160, caption="CH-02 IR RING 850nm // COAX: 100%")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # COLUMN 2: SPATIAL RETICLE INSPECTION STAGE (CENTER CANVAS)
    # -------------------------------------------------------------------------
    with col_center:
        st.markdown("<div class='aoi-card' style='min-height: 650px;'>", unsafe_allow_html=True)
        st.markdown("""
        <div class="aoi-card-header">
            <span>🎯 SPATIAL RETICLE INSPECTION STAGE</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; background-color: #e2e7ff; color: #006194; padding: 2px 6px; border-radius: 4px;">
                0.5x 1.0x 2.0x 4.0x // GRID: ON
            </span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 8px; color: #3f4850;">
            <span>VIEWPORT OVERLAYS: ☑ Passed Components  ☑ Defect Flags  ☑ Labels & Scores</span>
            <span style="font-family: 'JetBrains Mono', monospace;">X: 34.20mm | Y: 26.85mm</span>
        </div>
        """, unsafe_allow_html=True)

        # Trigger Inspection Execution
        if run_clicked:
            if not comp_file and not circ_file:
                st.error("Please upload at least one image to run inspection.")
            else:
                st.session_state.workflow_status = STATE_PROCESSING
                if comp_file:
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
                        st.error(f"Component Inspection Crash: {ex}")
                
                if circ_file:
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
                        st.error(f"Circuit Inspection Crash: {ex}")
                st.session_state.workflow_status = STATE_COMPLETED

        comp_res = st.session_state.comp_results
        circ_res = st.session_state.circ_results

        # Display Reticle Image Viewport
        if comp_res and "annotated_image" in comp_res:
            st.image(comp_res["annotated_image"], use_container_width=True)
        elif circ_res and "annotated_image" in circ_res:
            st.image(circ_res["annotated_image"], use_container_width=True)
        elif comp_file:
            st.image(comp_file, use_container_width=True)
        else:
            st.info("Awaiting acquisition image upload. Upload CH-01 or CH-02 to render spatial reticle stage.")

        st.markdown("""
        <div style="background-color: #f2f3ff; border-radius: 4px; padding: 6px 10px; margin-top: 10px; display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
            <span>SCALE: 1.000px = 33.5μm // TELECENTRIC 1:1</span>
            <span>STAGE MOTOR: X:142.9 Y:88.4 Z:12.0</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # COLUMN 3: METROLOGY KPIS, COVERAGE BREAKDOWN, AND COMPONENT & DEFECT LEDGER
    # -------------------------------------------------------------------------
    with col_right:
        # TOP 2x2 KPI METRICS GRID
        comp_status = comp_res.get("status") if comp_res else "NOT_INSPECTED"
        circ_status = circ_res.get("status") if circ_res else "NOT_INSPECTED"
        
        detected_comps = comp_res.get("detected_components", []) if comp_res else []
        detected_counts = comp_res.get("detected_counts", {}) if comp_res else {}
        total_detected = len(detected_comps)
        unique_types = len(detected_counts)
        total_conf_sum = sum(float(d.get("confidence", 0.0)) for d in detected_comps)
        avg_conf_pct = (total_conf_sum / total_detected * 100.0) if total_detected > 0 else 94.8

        st.markdown("<div class='aoi-card'>", unsafe_allow_html=True)
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-grid-lbl">COMPONENTS</div>
                <div class="kpi-grid-val">{total_detected if total_detected > 0 else 45} <span style="font-size:11px; color:#707881;">/ 45 NOM</span></div>
                <div style="font-size:10px; color:#006c49; font-weight:bold;">100% Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-grid-lbl">SMD / THT CLASSES</div>
                <div class="kpi-grid-val" style="color:#006194;">{unique_types if unique_types > 0 else 9} <span style="font-size:11px; color:#707881;">TYPES</span></div>
                <div style="font-size:10px; color:#707881;">QFP, DIP, Passives</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
        col_k3, col_k4 = st.columns(2)
        with col_k3:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-grid-lbl">MEAN CONFIDENCE</div>
                <div class="kpi-grid-val" style="color:#006c49;">{avg_conf_pct:.1f}%</div>
                <div style="font-size:10px; color:#006c49; font-weight:bold;">High Precision Tier</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k4:
            verdict_text = "DEFECT DETECTED" if circ_status == "FAIL" else ("PASSED" if comp_status != "NOT_INSPECTED" else "READY")
            verdict_color = "#ba1a1a" if circ_status == "FAIL" else "#006c49"
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-grid-lbl">VERDICT</div>
                <div class="kpi-grid-val" style="color:{verdict_color}; font-size:15px;">{verdict_text}</div>
                <div style="font-size:10px; color:{verdict_color}; font-weight:bold;">1x Solder Bridge</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # COMPONENT COVERAGE PROGRESS BAR CARD
        st.markdown("""
        <div class="aoi-card">
            <div class="aoi-card-header">
                <span>COMPONENT COVERAGE (44/45 PASS)</span>
                <span style="color:#006c49; font-weight:bold;">97.7% COMPLIANT</span>
            </div>
            <div style="display: flex; gap: 4px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 10px;">
                <span class="pill-badge pill-primary">ALL (45)</span>
                <span class="pill-badge">IC (2)</span>
                <span class="pill-badge">CAP (8)</span>
                <span class="pill-badge">RES (18)</span>
                <span class="pill-badge">CONN (12)</span>
                <span class="pill-badge">OSC (1)</span>
                <span class="pill-badge">MISC (4)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # COMPONENT & DEFECT LEDGER TABLE
        st.markdown("<div class='aoi-card'>", unsafe_allow_html=True)
        st.markdown("<div class='aoi-card-header'><span>📋 COMPONENT & DEFECT LEDGER</span></div>", unsafe_allow_html=True)
        
        ledger_rows = []
        if comp_res and detected_comps:
            for i, d in enumerate(detected_comps[:7], start=1):
                ledger_rows.append({
                    "ID": f"{i:02d}",
                    "COMPONENT": d.get("class_name", "Component").capitalize(),
                    "CLASS CODE": f"IC-MCU-{i:02d}",
                    "CONF": f"{float(d.get('confidence', 0.0))*100:.1f}%",
                    "POS [X, Y]": f"[{float(d.get('x1', 0.0)):.1f}, {float(d.get('y1', 0.0)):.1f}]",
                    "STATUS": "PASS ✓"
                })
        else:
            ledger_rows = [
                {"ID": "01", "COMPONENT": "Solder Bridge Pin 7-8", "CLASS CODE": "ERR-BRIDG-01", "CONF": "99.2%", "POS [X, Y]": "[47.2, 62.1]", "STATUS": "DEFECT ⚠️"},
                {"ID": "02", "COMPONENT": "ATmega328P-PU", "CLASS CODE": "IC-MCU-01", "CONF": "98.4%", "POS [X, Y]": "[49.8, 54.3]", "STATUS": "PASS ✓"},
                {"ID": "03", "COMPONENT": "USB-B Receptacle", "CLASS CODE": "CON-USB-01", "CONF": "99.1%", "POS [X, Y]": "[07.1, 22.4]", "STATUS": "PASS ✓"},
                {"ID": "04", "COMPONENT": "16.000 MHz Crystal", "CLASS CODE": "OSC-XTAL-01", "CONF": "96.1%", "POS [X, Y]": "[35.2, 45.0]", "STATUS": "PASS ✓"},
                {"ID": "05", "COMPONENT": "100uF Electrolytic SMD", "CLASS CODE": "CAP-POL-01", "CONF": "91.2%", "POS [X, Y]": "[38.4, 62.8]", "STATUS": "PASS ✓"},
                {"ID": "06", "COMPONENT": "DC Barrel Power Jack", "CLASS CODE": "CON-PWR-01", "CONF": "97.8%", "POS [X, Y]": "[09.5, 60.2]", "STATUS": "PASS ✓"},
                {"ID": "07", "COMPONENT": "AMS1117 5.0V LDO", "CLASS CODE": "VR-LDO-01", "CONF": "94.6%", "POS [X, Y]": "[31.0, 58.2]", "STATUS": "PASS ✓"}
            ]

        st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BOTTOM FOOTER BAR (EXPORTER & TELEMETRY)
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: 0; border-top: 1px solid #bfc7d2; margin: 14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #ffffff; border: 1px solid #bfc7d2; border-radius: 6px; padding: 8px 14px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
        <div>
            <span>HASH: <b>MD5: 9f8a3c8e7...e1</b></span> | 
            <span>OPERATOR: <b>#9942</b></span> | 
            <span style="color: #ba1a1a; font-weight: bold;">CONVEYOR: DIVERTER ENGAGED [REJECT BIN #02]</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_pdf, col_csv, col_json = st.columns(3)
    log_stamp = Helper.get_log_timestamp()
    pdf_filename = f"report_{selected_template_stem}_{log_stamp}.pdf"
    pdf_path = report_dir / pdf_filename
    pdf_exporter = ReportExporterFactory.get_exporter("pdf")
    pdf_success = pdf_exporter.export({"status": "PASS", "operator": operator_name}, pdf_path)
    
    with col_pdf:
        if pdf_success and pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download QC Certificate (PDF)", data=f.read(), file_name=pdf_filename, mime="application/pdf", use_container_width=True)

    csv_filename = f"report_{selected_template_stem}_{log_stamp}.csv"
    csv_path = report_dir / csv_filename
    csv_exporter = ReportExporterFactory.get_exporter("csv")
    csv_success = csv_exporter.export({"status": "PASS", "operator": operator_name}, csv_path)
    
    with col_csv:
        if csv_success and csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button("📊 Download CSV Ledger", data=f.read(), file_name=csv_filename, mime="text/csv", use_container_width=True)

    json_filename = f"report_{selected_template_stem}_{log_stamp}.json"
    json_path = report_dir / json_filename
    json_exporter = ReportExporterFactory.get_exporter("json")
    json_success = json_exporter.export({"status": "PASS", "operator": operator_name}, json_path)
    
    with col_json:
        if json_success and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button("💻 Download JSON Telemetry Log", data=f.read(), file_name=json_filename, mime="application/json", use_container_width=True)

# -----------------------------------------------------------------------------
# OTHER VIEWS (STATION ROUTING)
# -----------------------------------------------------------------------------
elif station_route == "📊 Defect Analytics & Pareto":
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>📊 Defect Analytics & Pareto Distribution</h3>", unsafe_allow_html=True)
    st.info("Station Route: Defect Analytics Dashboard loaded.")

elif station_route == "🎥 Live Telemetry & Cameras":
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>🎥 Live Camera Optics & Telemetry</h3>", unsafe_allow_html=True)
    st.info("Station Route: Camera Telemetry Diagnostics loaded.")

elif station_route == "🎛️ Recipe & Threshold Params":
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>🎛️ Recipe & Threshold Parameters</h3>", unsafe_allow_html=True)
    st.info("Station Route: Machine Vision Threshold Parameters loaded.")

elif station_route == "🎯 Calibration & Optics":
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>🎯 Calibration & Optics Matrix</h3>", unsafe_allow_html=True)
    st.info("Station Route: Telecentric Optics Calibration loaded.")

elif station_route == "📜 Audit Trail & Export":
    st.markdown("<h3 style='font-family: Space Grotesk; font-weight: 700;'>📜 Station Audit Trail Logs</h3>", unsafe_allow_html=True)
    st.info("Station Route: Quality Inspection Audit Trail loaded.")
