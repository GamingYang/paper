# Manuscript v2 and Supplement v1 Build Report

{
  "outputs": [
    "C:\\Users\\ROG\\Desktop\\paper\\codex\\manuscript-v2-and-supplement-v1\\battery_eol_rul_manuscript_v2.docx",
    "C:\\Users\\ROG\\Desktop\\paper\\codex\\manuscript-v2-and-supplement-v1\\battery_eol_rul_manuscript_v2.md",
    "C:\\Users\\ROG\\Desktop\\paper\\codex\\manuscript-v2-and-supplement-v1\\battery_eol_rul_supplementary_materials_v1.docx",
    "C:\\Users\\ROG\\Desktop\\paper\\codex\\manuscript-v2-and-supplement-v1\\battery_eol_rul_supplementary_materials_v1.md"
  ],
  "main_result_preserved": "ExtraTrees + 1000 cycles; Batch10 RMSE/MAE 56.19/49.42 cycles",
  "new_round8_materials_integrated": [
    "feature-family/horizon audit",
    "leakage negative control",
    "physics-consistent neural negative controls",
    "cubic-base boundary statement"
  ],
  "visual_render_qa": "not completed because local LibreOffice/soffice conversion is known broken on this machine"
}

## QA

- DOCX generation completed with python-docx.
- Structural QA passed:
  - `battery_eol_rul_manuscript_v2.docx`: 111 paragraphs, 6 tables, 6 inline figures; Chinese text and main metric `56.19/49.42` detected.
  - `battery_eol_rul_supplementary_materials_v1.docx`: 42 paragraphs, 7 tables, 5 inline figures; Chinese text and main metric `56.19/49.42` detected across paragraphs/tables.
- Render QA could not be completed. Anaconda Python lacks `pdf2image`; bundled Python launched the renderer but DOCX->PDF conversion timed out through LibreOffice/soffice, consistent with the known local LibreOffice `bootstrap.ini`/startup issue. Residual `soffice` processes from the timed-out render attempt were stopped.
