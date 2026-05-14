# TRIAGE SCAN ARTIFACTS EXCLUDED FROM COMMIT

Status: COMPLETE

The triage directory contained raw partial files:
- `MOCK_AUDIT_RAW.txt`
- `SECRET_SCAN_RAW_MASKED.txt`

These files are intentionally excluded by the local `.gitignore` in this directory because raw scans can be very large and can contain excessive sensitive context even when partially masked.

Committed evidence is limited to curated masked summaries, matrices, and decision reports.
