import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import logger
from src.utils.constants import STATUS_PASS

# ReportLab imports for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class BaseReportExporter(ABC):
    """
    Abstract Base Class outlining the interface for all file exporters.
    """
    @abstractmethod
    def export(self, data: Dict[str, Any], target_path: Path) -> bool:
        """
        Exports the inspection result dictionary to the specified target path.
        """
        pass


class JSONReportExporter(BaseReportExporter):
    """
    Concrete exporter generating structured JSON logs of inspection results.
    """
    def export(self, data: Dict[str, Any], target_path: Path) -> bool:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            logger.info(f"JSON Report successfully exported to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export JSON report: {e}")
            return False


class CSVReportExporter(BaseReportExporter):
    """
    Concrete exporter writing flat CSV registers of all anomalies.
    """
    def export(self, data: Dict[str, Any], target_path: Path) -> bool:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                # Header row
                writer.writerow(["Anomaly Type", "Component ID", "Type", "Expected Coords (Pct)", "Actual Coords (Pct)", "Details"])
                
                # Write Missing Components
                for m in data.get("missing", []):
                    writer.writerow([
                        "Missing", 
                        m.get("id"), 
                        m.get("type"), 
                        f"({m.get('expected_x_pct'):.2f}, {m.get('expected_y_pct'):.2f})", 
                        "N/A", 
                        m.get("reason", "")
                    ])
                
                # Write Misaligned Components
                for m in data.get("misaligned", []):
                    writer.writerow([
                        "Misaligned", 
                        m.get("id"), 
                        m.get("type"), 
                        f"({m.get('expected_x_pct'):.2f}, {m.get('expected_y_pct'):.2f})", 
                        f"({m.get('actual_x_pct'):.2f}, {m.get('actual_y_pct'):.2f})", 
                        f"Distance: {m.get('distance_mm')}mm (Tol: {m.get('tolerance_mm')}mm)"
                    ])

                # Write Cracks
                for crk in data.get("cracks", []):
                    writer.writerow([
                        "Solder Crack", 
                        crk.get("id"), 
                        "N/A", 
                        "N/A", 
                        f"({crk.get('center_x_pct'):.2f}, {crk.get('center_y_pct'):.2f})", 
                        f"Severity: {crk.get('severity')}"
                    ])

                # Write Extra Components
                for ext in data.get("extra", []):
                    writer.writerow([
                        "Extra Component", 
                        ext.get("id"), 
                        ext.get("type"), 
                        "N/A", 
                        f"({ext.get('center_x_pct'):.2f}, {ext.get('center_y_pct'):.2f})", 
                        f"Confidence: {ext.get('confidence')}"
                    ])
            logger.info(f"CSV Report successfully exported to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV report: {e}")
            return False


class PDFReportExporter(BaseReportExporter):
    """
    Concrete exporter compiling a formal visual PDF report using ReportLab.
    """
    def export(self, data: Dict[str, Any], target_path: Path) -> bool:
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab library not available. Skipping PDF generation.")
            return False
        
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            doc = SimpleDocTemplate(str(target_path), pagesize=letter,
                                    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            story = []
            styles = getSampleStyleSheet()

            # Custom Paragraph styles
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor('#1E293B'),
                spaceAfter=15
            )
            h2_style = ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading2'],
                fontSize=12,
                leading=16,
                textColor=colors.HexColor('#0F172A'),
                spaceBefore=12,
                spaceAfter=6
            )
            body_style = ParagraphStyle(
                'BodyTextCustom',
                parent=styles['BodyText'],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor('#334155')
            )

            # Title
            story.append(Paragraph("PCB ASSEMBLY VERIFICATION - AOI REPORT", title_style))
            story.append(Spacer(1, 10))

            # Header Metadata table
            status_str = data.get("status", "UNKNOWN")
            status_color = "#00FF66" if status_str == STATUS_PASS else "#FF3333"
            status_text = f"<font color='{status_color}'><b>{status_str}</b></font>"

            meta_data = [
                [Paragraph("<b>Inspection Date:</b>", body_style), Paragraph(data.get("inspection_date", "N/A"), body_style),
                 Paragraph("<b>Operator ID:</b>", body_style), Paragraph(data.get("operator", "N/A"), body_style)],
                [Paragraph("<b>Board Template:</b>", body_style), Paragraph(data.get("template_name", "N/A"), body_style),
                 Paragraph("<b>Processing Duration:</b>", body_style), Paragraph(data.get("processing_time", "N/A"), body_style)],
                [Paragraph("<b>Overall Status:</b>", body_style), Paragraph(status_text, body_style),
                 Paragraph("<b>Total Components:</b>", body_style), Paragraph(str(data.get("component_statistics", {}).get("total_expected", 0)), body_style)]
            ]

            meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
            meta_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 15))

            # --- ANOMALY LEDGER TABLES ---
            
            # 1. Missing Components Table
            missing = data.get("missing", [])
            story.append(Paragraph(f"Missing Components ({len(missing)})", h2_style))
            if missing:
                table_data = [["Component ID", "Type", "Expected Center (Pct)", "Reason"]]
                for m in missing:
                    table_data.append([
                        m.get("id", "N/A"),
                        m.get("type", "N/A"),
                        f"({m.get('expected_x_pct', 0.0):.2f}, {m.get('expected_y_pct', 0.0):.2f})",
                        m.get("reason", "N/A")
                    ])
                t = Table(table_data, colWidths=[130, 110, 140, 160])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
            else:
                story.append(Paragraph("No missing components identified.", body_style))
            story.append(Spacer(1, 10))

            # 2. Misaligned Components Table
            misaligned = data.get("misaligned", [])
            story.append(Paragraph(f"Misaligned Components ({len(misaligned)})", h2_style))
            if misaligned:
                table_data = [["Component ID", "Type", "Expected Location (Pct)", "Actual Location (Pct)", "Shift (mm)", "Limit (mm)"]]
                for m in misaligned:
                    table_data.append([
                        m.get("id", "N/A"),
                        m.get("type", "N/A"),
                        f"({m.get('expected_x_pct', 0.0):.2f}, {m.get('expected_y_pct', 0.0):.2f})",
                        f"({m.get('actual_x_pct', 0.0):.2f}, {m.get('actual_y_pct', 0.0):.2f})",
                        f"{m.get('distance_mm')} mm",
                        f"{m.get('tolerance_mm')} mm"
                    ])
                t = Table(table_data, colWidths=[110, 90, 120, 120, 50, 50])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
            else:
                story.append(Paragraph("No misaligned components identified.", body_style))
            story.append(Spacer(1, 10))

            # 3. Cracks Table
            cracks = data.get("cracks", [])
            story.append(Paragraph(f"Solder / Board Cracks ({len(cracks)})", h2_style))
            if cracks:
                table_data = [["Crack ID", "Target Component", "Coordinates (Pct)", "Severity"]]
                for c in cracks:
                    table_data.append([
                        c.get("id", "N/A"),
                        c.get("parent_component", "N/A"),
                        f"({c.get('center_x_pct', 0.0):.2f}, {c.get('center_y_pct', 0.0):.2f})",
                        c.get("severity", "N/A")
                    ])
                t = Table(table_data, colWidths=[120, 140, 140, 140])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
            else:
                story.append(Paragraph("No solder or board cracking defects identified.", body_style))
            story.append(Spacer(1, 10))

            # 4. Extra Components Table
            extra = data.get("extra", [])
            story.append(Paragraph(f"Extra Components ({len(extra)})", h2_style))
            if extra:
                table_data = [["Component ID", "Type", "Coordinates (Pct)", "Confidence"]]
                for e in extra:
                    table_data.append([
                        e.get("id", "N/A"),
                        e.get("type", "N/A"),
                        f"({e.get('center_x_pct', 0.0):.2f}, {e.get('center_y_pct', 0.0):.2f})",
                        f"{e.get('confidence', 0.0)*100:.1f}%"
                    ])
                t = Table(table_data, colWidths=[120, 120, 150, 150])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
            else:
                story.append(Paragraph("No unregistered extra components identified.", body_style))

            # Build Document
            doc.build(story)
            logger.info(f"PDF Report successfully exported to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export PDF report: {e}")
            return False


class ReportExporterFactory:
    """
    Factory to resolve the appropriate exporter based on requested target path suffix.
    """
    @staticmethod
    def get_exporter(format_suffix: str) -> BaseReportExporter:
        clean_suffix = format_suffix.strip().lower().replace(".", "")
        if clean_suffix == "json":
            return JSONReportExporter()
        elif clean_suffix == "csv":
            return CSVReportExporter()
        elif clean_suffix == "pdf":
            return PDFReportExporter()
        else:
            raise ValueError(f"No exporter implementation mapped for format suffix: '{format_suffix}'")
