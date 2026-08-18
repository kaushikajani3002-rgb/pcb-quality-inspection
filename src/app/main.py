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
    "src.ai.detection_engine"
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

# -----------------------------------------------------------------------------
# PAGE SETUP & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial PCB AOI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injector for industrial aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }
    .metric-val-pass {
        color: #00FF66;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-val-fail {
        color: #FF3333;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-val-neutral {
        color: #38bdf8;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        margin-top: 5px;
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
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 20px;">
        <span style="color: #38bdf8; font-size: 18px; font-weight: bold; letter-spacing: 1px;">🛡️ INDUSTRIAL AOI</span><br>
        <span style="color: #64748b; font-size: 10px; text-transform: uppercase;">PCB Quality Inspection</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Operator Controls")
    
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
        "Select PCB Template Profile",
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
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.confidence", 0.50)),
        step=0.05,
        help="YOLO model classification score cut-off."
    )
    
    iou_threshold = st.slider(
        "IoU Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.iou", 0.45)),
        step=0.05,
        help="Non-Maximum Suppression (NMS) bounding boxes intersection slider."
    )

    # Load Position Tolerance from configuration
    position_tolerance = float(config.get("inspection.position_tolerance", 1.5))

    st.markdown("---")
    st.subheader("Model Status")
    
    # Lazy loading models through ModelManager
    from src.ai.model_manager import ModelManager
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

    # Display status
    if comp_ready:
        st.markdown("<span style='color:#10b981; font-weight:bold;'>✓ Component Detector Ready</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#ef4444; font-weight:bold;'>✕ Component Detector Failed</span><br><small style='color:#ef4444;'>{comp_err_msg}</small>", unsafe_allow_html=True)

    if def_ready:
        st.markdown(f"<span style='color:#10b981; font-weight:bold;'>✓ Circuit Defect Detector Ready</span> (`{selected_defect_model_name}`)", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#ef4444; font-weight:bold;'>✕ Circuit Detector Failed</span> (`{selected_defect_model_name}`)<br><small style='color:#ef4444;'>{def_err_msg}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Simulation Options")

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

    # Action Triggers
    col_run, col_reset = st.columns(2)
    with col_run:
        run_clicked = st.button("▶ Run AOI", use_container_width=True)
    with col_reset:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)

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
# MAIN DASHBOARD VIEW
# -----------------------------------------------------------------------------
st.title("🏭 Automated Optical Inspection Assembly Verification Console")
st.caption(f"System: {config.get('dashboard.company_name')} | Status: CONNECTED")

# PCB specifications layout
with st.expander("🔍 Selected PCB Specifications & Component Footprint Map", expanded=True):
    board_dims = template.get("board_dimensions", {})
    w_mm = board_dims.get("width_mm", "N/A")
    h_mm = board_dims.get("height_mm", "N/A")
    critical_comps = [c["id"] for c in template.get("components", [])]
    
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown(f"**Board Name:** `{template.get('board_name', 'Unknown PCB')}`")
        st.markdown(f"**Physical Dimensions:** `{w_mm} mm × {h_mm} mm`")
    with col_meta2:
        st.markdown(f"**Critical Component Count:** `{len(critical_comps)}` expected")
        st.markdown(f"**Inspection Standard:** `Euclidean mm Misalignment`")
    with col_meta3:
        st.markdown("**Mapped Components:**")
        st.caption(", ".join(critical_comps))

st.markdown("---")

# -----------------------------------------------------------------------------
# TWO IMAGE UPLOAD CARDS (SIDE BY SIDE)
# -----------------------------------------------------------------------------
st.subheader("Image Acquisition Panel")
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.markdown("""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 10px; font-weight: bold; color: #38bdf8;">
        CARD 1: COMPONENT INSPECTION IMAGE
    </div>
    """, unsafe_allow_html=True)
    comp_file = st.file_uploader(
        "Upload component image", 
        type=["png", "jpg", "jpeg"],
        key="comp_uploader",
        help="Image optimized for component presence and position validation."
    )
    
    if comp_file:
        if st.session_state.last_comp_file_name != comp_file.name:
            st.session_state.comp_results = None
            st.session_state.last_comp_file_name = comp_file.name
            st.session_state.workflow_status = STATE_IDLE
        st.success("✓ Image uploaded")
        # Display small thumbnail
        st.image(comp_file, width=150)
    else:
        if st.session_state.last_comp_file_name is not None:
            st.session_state.comp_results = None
            st.session_state.last_comp_file_name = None
            st.session_state.workflow_status = STATE_IDLE
        st.warning("⚠ Image not uploaded")

with col_upload2:
    st.markdown("""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 10px; font-weight: bold; color: #38bdf8;">
        CARD 2: CIRCUIT / DEFECT INSPECTION IMAGE
    </div>
    """, unsafe_allow_html=True)
    circ_file = st.file_uploader(
        "Upload circuit image", 
        type=["png", "jpg", "jpeg"],
        key="circ_uploader",
        help="Image optimized for solder joint defects and trace fracture inspection."
    )
    
    if circ_file:
        if st.session_state.last_circ_file_name != circ_file.name:
            st.session_state.circ_results = None
            st.session_state.last_circ_file_name = circ_file.name
            st.session_state.workflow_status = STATE_IDLE
        st.success("✓ Image uploaded")
        # Display small thumbnail
        st.image(circ_file, width=150)
    else:
        if st.session_state.last_circ_file_name is not None:
            st.session_state.circ_results = None
            st.session_state.last_circ_file_name = None
            st.session_state.workflow_status = STATE_IDLE
        st.warning("⚠ Image not uploaded")

st.markdown("---")

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
                    # Provide matched detections to circuit inspection for simulated crack mappings
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
    
    # 1. FINAL RESULT AGGREGATOR
    st.subheader("Aggregated System Status")
    
    # 1. FINAL RESULT AGGREGATOR
    st.subheader("Aggregated System Status")
    
    if comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "PASS":
        st.markdown("""
        <div style="background-color: #1e3a27; border: 2px solid #00FF66; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #00ff6633; margin-bottom: 20px;">
            <span style="color: #00FF66; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🟢 COMPONENT DETECTION COMPLETE & CIRCUIT PASSED</span><br>
            <span style="color: #94a3b8; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>PASS</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "FAIL":
        st.markdown("""
        <div style="background-color: #3f1e1e; border: 2px solid #FF3333; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #ff333333; margin-bottom: 20px;">
            <span style="color: #FF3333; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🔴 CIRCUIT DEFECT DETECTED</span><br>
            <span style="color: #94a3b8; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>FAIL</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status in ("PASS", "DETECTION_COMPLETE") and circ_status == "NOT_INSPECTED":
        st.markdown("""
        <div style="background-color: #1e3a27; border: 2px solid #00FF66; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #00ff6633; margin-bottom: 20px;">
            <span style="color: #00FF66; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🟢 COMPONENT DETECTION COMPLETE</span><br>
            <span style="color: #94a3b8; font-size: 13px;">Component Detection: <b>COMPLETE</b> | Circuit Inspection: <b>NOT INSPECTED</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    elif comp_status == "NOT_INSPECTED" and circ_status == "PASS":
        st.markdown("""
        <div style="background-color: #1e3a27; border: 2px solid #00FF66; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #00ff6633; margin-bottom: 20px;">
            <span style="color: #00FF66; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🟢 CIRCUIT INSPECTION PASSED</span><br>
            <span style="color: #94a3b8; font-size: 13px;">Component Detection: <b>NOT INSPECTED</b> | Circuit Inspection: <b>PASS</b></span>
        </div>
        """, unsafe_allow_html=True)

    elif comp_status == "NOT_INSPECTED" and circ_status == "FAIL":
        st.markdown("""
        <div style="background-color: #3f1e1e; border: 2px solid #FF3333; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #ff333333; margin-bottom: 20px;">
            <span style="color: #FF3333; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🔴 CIRCUIT DEFECT DETECTED</span><br>
            <span style="color: #94a3b8; font-size: 13px;">Component Detection: <b>NOT INSPECTED</b> | Circuit Inspection: <b>FAIL</b></span>
        </div>
        """, unsafe_allow_html=True)
        
    else: # Neither inspected
        st.markdown("""
        <div style="background-color: #3b3a30; border: 2px solid #facc15; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #facc1533; margin-bottom: 20px;">
            <span style="color: #facc15; font-size: 28px; font-weight: bold; letter-spacing: 2px;">⚠️ NO INSPECTION EXECUTED</span><br>
            <span style="color: #e2e8f0; font-size: 14px; font-weight: bold;">Please upload an image to begin inspection.</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Inspection Status Cards
    st.subheader("Independent Inspection Results")
    col_card1, col_card2 = st.columns(2)
    
    with col_card1:
        st.markdown("<div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px;'>", unsafe_allow_html=True)
        st.markdown("### COMPONENT INSPECTION")
        if comp_status in ("PASS", "DETECTION_COMPLETE"):
            st.markdown("<h2 style='color:#00FF66; margin-top:0;'>🟢 DETECTION COMPLETE</h2>", unsafe_allow_html=True)
            stats = comp_res.get("component_statistics", {})
            total_detected = stats.get("total_detected", len(comp_res.get("detected_components", [])))
            st.markdown(f"**Total Components Detected:** `{total_detected}`")
            
            detected_counts = comp_res.get("detected_counts", {})
            if detected_counts:
                st.markdown("**Component Types:**")
                for cname, count in sorted(detected_counts.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"- **{cname}:** `{count}`")
        else:
            st.markdown("<h2 style='color:#e2e8f0; margin-top:0;'>⚪ NOT INSPECTED</h2>", unsafe_allow_html=True)
            st.caption(comp_res.get("reason", "Component image not uploaded"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_card2:
        st.markdown("<div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px;'>", unsafe_allow_html=True)
        st.markdown("### CIRCUIT INSPECTION")
        if circ_status == "PASS":
            st.markdown("<h2 style='color:#00FF66; margin-top:0;'>🟢 PASS</h2>", unsafe_allow_html=True)
        elif circ_status == "FAIL":
            st.markdown("<h2 style='color:#FF3333; margin-top:0;'>🔴 FAIL</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:#e2e8f0; margin-top:0;'>⚪ NOT INSPECTED</h2>", unsafe_allow_html=True)
            st.caption(circ_res.get("reason", "Circuit image not uploaded"))
            
        if circ_status in ("PASS", "FAIL"):
            defects = circ_res.get("defects", [])
            st.markdown(f"**Defects Detected:** `{len(defects)}`")
            
            solder_defects = len([d for d in defects if "crack" in d.get("class_name", "").lower() or "solder" in d.get("class_name", "").lower()])
            trace_defects = len(defects) - solder_defects
            st.markdown(f"**Solder Defects:** `{solder_defects}`")
            st.markdown(f"**Trace Defects:** `{trace_defects}`")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Visual Inspection Area
    st.markdown("### Visual Inspection Area")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("#### COMPONENT VISUALS")
        if comp_status in ("PASS", "DETECTION_COMPLETE"):
            tab_comp_orig, tab_comp_box = st.tabs(["Original Image", "Component Overlays"])
            with tab_comp_orig:
                st.image(comp_res["original_image"], use_container_width=True)
            with tab_comp_box:
                st.image(comp_res["annotated_image"], use_container_width=True)
        else:
            st.info("Component inspection not available.\nComponent image not uploaded.")
            
    with col_vis2:
        st.markdown("#### CIRCUIT VISUALS")
        if circ_status in ("PASS", "FAIL"):
            tab_circ_orig, tab_circ_box = st.tabs(["Original Image", "Defect Overlays"])
            with tab_circ_orig:
                st.image(circ_res["original_image"], use_container_width=True)
            with tab_circ_box:
                st.image(circ_res["annotated_image"], use_container_width=True)
        else:
            st.info("Circuit inspection not available.\nCircuit image not uploaded.")

    # 4. Verification Registers
    st.markdown("### Verification Registers")
    tab_comp_reg, tab_circ_reg = st.tabs(["COMPONENT INVENTORY", "CIRCUIT DEFECT REGISTER"])
    
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
            stats = comp_res.get("component_statistics", {})
            
            total_detected = len(detected_comps)
            unique_types = len(detected_counts)
            
            total_conf_sum = sum(float(d.get("confidence", 0.0)) for d in detected_comps)
            avg_conf_pct = (total_conf_sum / total_detected * 100.0) if total_detected > 0 else 0.0
            
            # --- TOP KPI METRICS ROW ---
            st.markdown("""
            <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <div style="color: #00FF66; font-size: 16px; font-weight: bold; margin-bottom: 12px; letter-spacing: 1px;">
                    🟢 COMPONENT DETECTION COMPLETE
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 150px; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Total Components</div>
                        <div style="color: #00FF66; font-size: 28px; font-weight: bold; margin-top: 4px;">{}</div>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Unique Types</div>
                        <div style="color: #38bdf8; font-size: 28px; font-weight: bold; margin-top: 4px;">{}</div>
                    </div>
                    <div style="flex: 1; min-width: 150px; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                        <div style="color: #94a3b8; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Avg Confidence</div>
                        <div style="color: #f87171; font-size: 28px; font-weight: bold; margin-top: 4px;">{:.1f}%</div>
                    </div>
                </div>
                <div style="color: #64748b; font-size: 12px; margin-top: 10px; font-weight: 500;">
                    {} components detected across {} component types (Avg. Conf: {:.1f}%)
                </div>
            </div>
            """.format(total_detected, unique_types, avg_conf_pct, total_detected, unique_types, avg_conf_pct), unsafe_allow_html=True)
            
            # --- TWO COLUMN SPLIT: SUMMARY vs VISUALIZATION ---
            col_inv_left, col_inv_right = st.columns([1, 1.3])
            
            with col_inv_left:
                st.markdown("#### COMPONENT TYPE SUMMARY")
                if detected_counts:
                    summary_rows = []
                    running_total = 0
                    for ctype, count in sorted(detected_counts.items(), key=lambda x: x[1], reverse=True):
                        summary_rows.append({
                            "Component Type": format_class_name(ctype),
                            "Count": count
                        })
                        running_total += count
                    
                    # Total row
                    summary_rows.append({
                        "Component Type": "TOTAL",
                        "Count": running_total
                    })
                    
                    # Consistency check verification
                    if running_total != total_detected:
                        st.warning(f"Discrepancy detected: Type sum ({running_total}) != Total detected ({total_detected})")
                        
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No component types detected.")
                    
            with col_inv_right:
                st.markdown("#### COMPONENT VISUALIZATION")
                if "annotated_image" in comp_res:
                    st.image(comp_res["annotated_image"], use_container_width=True, caption=f"Annotated PCB Component Overlays ({total_detected} Bounding Boxes)")
                else:
                    st.info("Visual overlay not available.")
                    
            # --- FULL WIDTH DETECTION DETAILS TABLE (SORTED BY CONFIDENCE DESCENDING) ---
            st.markdown("---")
            st.markdown("#### COMPONENT DETECTION DETAILS")
            if detected_comps:
                # Sort detections by confidence descending for display
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
            st.markdown("""
            <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 20px; text-align: center;">
                <h3 style="color: #94a3b8; margin-top: 0;">⚪ NO COMPONENTS DETECTED / NOT INSPECTED</h3>
                <p style="color: #64748b; font-size: 14px;">Please upload a PCB component image to generate component inventory.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with tab_circ_reg:
        if circ_status in ("PASS", "FAIL"):
            # Build Circuit Defect Register Table
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

    # 5. Report Exporters
    st.markdown("### Export Logs and Reports")
    
    # Construct combined export payload
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
                    label="📄 Download PDF Inspection Report",
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
                    label="📊 Download CSV Discrepancies Log",
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
                    label="💻 Download JSON Database Log",
                    data=f.read(),
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.button("💻 JSON Exporter Offline", disabled=True, use_container_width=True)

    # 6. Inference Debug Console
    if debug_mode:
        st.markdown("---")
        st.markdown("### 🛠️ Inference Debug Console")
        with st.expander("Show Complete Backend & AI Model Trace Log", expanded=True):
            st.json({
                "component_inspection_debug": comp_res.get("debug_info") if comp_res else "NOT RUN",
                "circuit_inspection_debug": circ_res if circ_res else "NOT RUN"
            })
