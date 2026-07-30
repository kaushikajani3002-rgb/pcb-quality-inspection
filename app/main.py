import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# Setup Python sys.path so we can import modules from project root
import sys
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
for mod in [
    "utils.json_loader", "utils.template_manager", "utils.logger",
    "mock.mock_results", "inspection.inspection_engine", "inspection.position_checker",
    "detection_engine"
]:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

from utils.config_loader import ConfigLoader
from utils.template_manager import TemplateManager
from utils.logger import logger
from utils.report_exporter import ReportExporterFactory
from utils.validators import ImageValidator, ConfigurationValidator
from utils.helper import Helper
from utils.constants import (
    STATUS_PASS, STATUS_FAIL, 
    STATE_IDLE, STATE_PROCESSING, STATE_COMPLETED, STATE_ERROR,
    COLOR_PASS, COLOR_FAIL
)
from mock.mock_results import MockInspectionService
from inspection.inspection_engine import InspectionEngine
from detection_engine import (
    load_model, run_component_counting, run_defect_detection,
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
if "active_inspection_results" not in st.session_state:
    st.session_state.active_inspection_results = {}
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "detection_image" not in st.session_state:
    st.session_state.detection_image = None
if "segmentation_image" not in st.session_state:
    st.session_state.segmentation_image = None
if "error_message" not in st.session_state:
    st.session_state.error_message = ""

# Load configurations
try:
    config = ConfigLoader()
    template_manager = TemplateManager()
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

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
    
    # 1. Device Selection Dropdown - Mapped strictly to the three requested options
    device_options = {
        "Arduino Uno": "arduino_uno",
        "ESP32 DevKit": "esp32_devkit",
        "STM32 Blue Pill": "stm32_blue_pill"
    }
    
    selected_device_lbl = st.selectbox(
        "Select PCB Template Profile",
        options=list(device_options.keys()),
        index=0,
        help="Loads physical measurements and expected component positions."
    )
    selected_template_stem = device_options[selected_device_lbl]
    
    # 2. Image File Uploader
    uploaded_file = st.file_uploader(
        "Upload PCB Image", 
        type=["png", "jpg", "jpeg"],
        help="Upload target board camera feed. (Max size: 15MB)"
    )

    # 3. Parameters Sliders (Confidence, IoU, and Position Tolerance in mm)
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.confidence", 0.50)),
        step=0.05
    )
    
    iou_threshold = st.slider(
        "IoU Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(config.get("inspection.iou", 0.45)),
        step=0.05
    )

    position_tolerance = st.slider(
        "Position Tolerance (mm)", 
        min_value=0.1, 
        max_value=5.0, 
        value=1.5,
        step=0.1,
        help="Euclidean physical displacement limit in millimeters."
    )

    st.markdown("---")
    st.subheader("Simulation Options")
    
    selected_defect_model_lbl = st.selectbox(
        "Select Defect Model",
        options=["None", "DeepPCB", "DsPCBSD+", "HRIPCB", "TDD-PCB"],
        index=0,
        help="Select defect model (untrained) to run solder/trace checks."
    )
    
    if selected_defect_model_lbl != "None":
        st.sidebar.error(f"Selected model '{selected_defect_model_lbl}' weights do not exist yet. Using graceful fallback.")

    defect_mode = st.checkbox(
        "Force Anomaly/Defect Mode", 
        value=True,
        help="Toggles simulated inspection errors (missing, misaligned, cracks)."
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
    st.session_state.active_inspection_results = {}
    st.session_state.original_image = None
    st.session_state.detection_image = None
    st.session_state.segmentation_image = None
    st.session_state.error_message = ""
    logger.info("Inspection system reset.")
    st.rerun()

# Load Selected Template Profile
template = template_manager.load_template(selected_template_stem)
if template:
    st.session_state.current_pcb_template = template
else:
    st.error(f"Error loading template config for {selected_device_lbl}")
    st.stop()

# Handle Run Click
if run_clicked:
    st.session_state.workflow_status = STATE_PROCESSING
    st.session_state.error_message = ""
    
    # Validate sliders
    val_ok, val_msg = ConfigurationValidator.validate_inspection_parameters(
        conf_threshold, iou_threshold, position_tolerance
    )
    if not val_ok:
        st.session_state.workflow_status = STATE_ERROR
        st.session_state.error_message = f"Config Error: {val_msg}"
        logger.error(st.session_state.error_message)
    else:
        # Validate uploaded image if present
        image_valid = True
        if uploaded_file is not None:
            meta_ok, meta_msg = ImageValidator.validate_file_metadata(
                uploaded_file.name, uploaded_file.size
            )
            if not meta_ok:
                image_valid = False
                st.session_state.workflow_status = STATE_ERROR
                st.session_state.error_message = f"Image Metadata Error: {meta_msg}"
            else:
                image_bytes = uploaded_file.read()
                sig_ok, sig_msg = ImageValidator.validate_image_bytes(image_bytes)
                if not sig_ok:
                    image_valid = False
                    st.session_state.workflow_status = STATE_ERROR
                    st.session_state.error_message = f"Image Header Error: {sig_msg}"
        
        if image_valid:
            if uploaded_file is None:
                st.session_state.workflow_status = STATE_ERROR
                st.session_state.error_message = "Please upload a PCB image before running AOI."
                logger.error(st.session_state.error_message)
            else:
                logger.info("Executing real component YOLO inference pipeline...")
                progress_text = "Verifying physical alignments..."
                my_bar = st.progress(0, text=progress_text)
                for pct in range(100):
                    time.sleep(0.005)
                    my_bar.progress(pct + 1, text=progress_text)
                my_bar.empty()

                try:
                    # Load real YOLO component model
                    component_model = load_model("Component")
                    
                    # Run actual component detection and checker aggregation
                    results = run_component_counting(
                        uploaded_image=uploaded_file,
                        component_model=component_model,
                        conf_slider=conf_threshold,
                        iou_slider=iou_threshold,
                        active_template=template,
                        position_tolerance_slider=position_tolerance,
                        defect_mode=defect_mode
                    )
                    
                    st.session_state.active_inspection_results = results
                    st.session_state.original_image = results["original_image"]
                    st.session_state.detection_image = results["annotated_image"]
                    st.session_state.segmentation_image = results["segmentation_image"]
                    st.session_state.workflow_status = STATE_COMPLETED
                    
                    logger.info(f"Inference complete. Status: {results['status']}")
                except Exception as ex:
                    st.session_state.workflow_status = STATE_ERROR
                    st.session_state.error_message = f"Pipeline Crash: {ex}"
                    logger.error(st.session_state.error_message, exc_info=True)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD VIEW
# -----------------------------------------------------------------------------
st.title("🏭 Automated Optical Inspection Assembly Verification Console")
st.caption(f"System: {config.get('dashboard.company_name')} | Status: CONNECTED")

# Error message panel
if st.session_state.workflow_status == STATE_ERROR:
    st.error(st.session_state.error_message)

# Device Metadata Summary Panel
with st.expander("🔍 Selected PCB Specifications & Component Footprint Map", expanded=True):
    board_dims = template.get("board_dimensions", {})
    w_mm = board_dims.get("width_mm", "N/A")
    h_mm = board_dims.get("height_mm", "N/A")
    critical_comps = [c["id"] for c in template.get("components", [])]
    
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown(f"**Board Name:** `{template.get('board_name', template.get('template_name', 'Unknown PCB'))}`")
        st.markdown(f"**Physical Dimensions:** `{w_mm} mm × {h_mm} mm`")
    with col_meta2:
        st.markdown(f"**Critical Component Count:** `{len(critical_comps)}` expected")
        st.markdown(f"**Inspection Standard:** `Euclidean mm Misalignment`")
    with col_meta3:
        st.markdown("**Mapped Components:**")
        st.caption(", ".join(critical_comps))

if st.session_state.workflow_status == STATE_IDLE:
    st.info("System IDLE. Click 'Run AOI' to analyze the selected board.")

# Completed Results Display
if st.session_state.workflow_status in (STATE_COMPLETED, STATE_PROCESSING):
    results = st.session_state.active_inspection_results
    
    # 1. PASS/FAIL Badge Header
    is_pass = results.get("status") == STATUS_PASS
    if is_pass:
        st.markdown("""
        <div style="background-color: #1e3a27; border: 2px solid #00FF66; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #00ff6633; margin-bottom: 20px;">
            <span style="color: #00FF66; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🟢 ASSEMBLY PASSED (ZERO DEFECTS FLAGGED)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #3f1e1e; border: 2px solid #FF3333; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 0 15px #ff333333; margin-bottom: 20px;">
            <span style="color: #FF3333; font-size: 28px; font-weight: bold; letter-spacing: 2px;">🔴 ASSEMBLY FAILED (ANOMALIES IDENTIFIED)</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. Metric Cards
    metrics = compute_dashboard_metrics(results)
    m_cols = st.columns(5)
    with m_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val-neutral">{metrics['expected_count']}</div>
            <div class="metric-lbl">Expected Components</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val-pass">{metrics['correct_count']}</div>
            <div class="metric-lbl">Correctly Placed</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cols[2]:
        val_color = "fail" if metrics['missing_count'] > 0 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val-{val_color}">{metrics['missing_count']}</div>
            <div class="metric-lbl">Missing Components</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cols[3]:
        val_color = "fail" if metrics['anomaly_count'] > 0 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val-{val_color}">{metrics['anomaly_count']}</div>
            <div class="metric-lbl">Anomalies Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val-neutral">{metrics['processing_time']}</div>
            <div class="metric-lbl">Inspection Duration</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Diagnostic Views Columns
    st.markdown("### Visual Inspection Columns")
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.subheader("PCB Camera Feed")
        if st.session_state.original_image:
            st.image(st.session_state.original_image, use_container_width=True)
    with col_img2:
        st.subheader("Component Bounding Boxes")
        if st.session_state.detection_image:
            st.image(st.session_state.detection_image, use_container_width=True)
    with col_img3:
        st.subheader("Solder Joint Segmentation")
        if st.session_state.segmentation_image:
            st.image(st.session_state.segmentation_image, use_container_width=True)

    # 4. Anomalies ledgers
    st.markdown("### Verification Registers")
    tab_census, tab_defects = st.tabs(["Component Census", "Discrepancy Detail Logs"])
    
    with tab_census:
        st.subheader("Inventory Breakdown")
        df_inventory = build_inventory_table(template, results.get("detected_counts", {}))
        st.dataframe(df_inventory, use_container_width=True, hide_index=True)

    with tab_defects:
        st.subheader("Discrepancy Register Detail Log")
        defect_rows = []
        
        # Missing
        for m in results.get("missing", []):
            defect_rows.append({
                "Category": "MISSING",
                "Component ID": m["id"],
                "Type": m["type"],
                "Expected Coords (x, y)": f"({m['expected_x_pct']:.2f}, {m['expected_y_pct']:.2f})",
                "Actual Coords (x, y)": "ABSENT",
                "Details": m["reason"]
            })
        # Misaligned
        for m in results.get("misaligned", []):
            defect_rows.append({
                "Category": "MISALIGNED",
                "Component ID": m["id"],
                "Type": m["type"],
                "Expected Coords (x, y)": f"({m['expected_x_pct']:.2f}, {m['expected_y_pct']:.2f})",
                "Actual Coords (x, y)": f"({m['actual_x_pct']:.2f}, {m['actual_y_pct']:.2f})",
                "Details": f"Offset: {m['distance_mm']}mm (Limit: {m['tolerance_mm']}mm)"
            })
        # Cracks
        for c in results.get("cracks", []):
            defect_rows.append({
                "Category": "SOLDER CRACK",
                "Component ID": c["id"],
                "Type": "Joint Outline",
                "Expected Coords (x, y)": "N/A",
                "Actual Coords (x, y)": f"({c['center_x_pct']:.2f}, {c['center_y_pct']:.2f})",
                "Details": f"Fracture Severity: {c['severity']}"
            })
        # Extra
        for e in results.get("extra", []):
            defect_rows.append({
                "Category": "EXTRA COMPONENT",
                "Component ID": e["id"],
                "Type": e["type"],
                "Expected Coords (x, y)": "UNREGISTERED",
                "Actual Coords (x, y)": f"({e['center_x_pct']:.2f}, {e['center_y_pct']:.2f})",
                "Details": f"Placement Confidence: {e['confidence']*100:.1f}%"
            })
            
        if defect_rows:
            st.dataframe(pd.DataFrame(defect_rows), use_container_width=True, hide_index=True)
        else:
            st.success("Zero defect markers flagged in the active verification register.")

    # 5. Report Generators
    st.markdown("### Export Logs and Reports")
    
    # Export payload package (contains distances in mm, coordinates in percentages)
    export_payload = {
        "status": results["status"],
        "inspection_date": Helper.get_current_timestamp(),
        "operator": operator_name,
        "template_name": results["template_name"],
        "processing_time": results["processing_time"],
        "component_statistics": results["component_statistics"],
        "missing": results["missing"],
        "misaligned": results["misaligned"],
        "cracks": results["cracks"],
        "extra": results["extra"]
    }
    
    col_pdf, col_csv, col_json = st.columns(3)
    log_stamp = Helper.get_log_timestamp()
    
    # PDF export call
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

    # CSV export call
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

    # JSON export call
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
