import argparse
import json
import re
from pathlib import Path

# List of label keys for report analysis
LABEL_KEYS = [
    "Detrusor_instability", "Flow_pattern", "EMG_ES_relaxation",
    "Trabeculation", "Diverticulum", "Cystocele", "VUR",
    "Bladder_neck_relaxation", "External_sphincter_relaxation",
    "Pelvic_floor_relaxation"
]

# Types of data outputs to check within each report
DATA_TYPES = ["pftg", "pfus", "pfs", "xray"]


def is_all_na(txt_path: Path) -> bool:
    # Determine if output.txt contains only NA, N/A, or is empty
    if not txt_path.is_file():
        return True
    content = txt_path.read_text(encoding="utf-8", errors="ignore").lower()
    # If none of the label keys appear in the content, treat as missing
    if not any(k.lower() in content for k in LABEL_KEYS):
        return True
    # Filter out blank lines and count lines
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    # Count lines where the value part is NA, N/A, none, or blank
    blanks = sum(
        bool(re.search(r'\bna\b|n/a|none|^\s*$', ln.split(":")[-1], re.I))
        for ln in lines
    )
    # Return True if all lines are blanks
    return blanks == len(lines)


def scan_report(report_dir: Path):
    # Scan a single report directory and return available data types and text status
    out_root = report_dir / "output"
    has = {}
    # Count files for each data type
    for t in DATA_TYPES:
        data_dir = out_root / t
        count = len(list(data_dir.glob("*.*"))) if data_dir.exists() else 0
        has[t] = count

    # Check text report
    txt_path = out_root / "result" / "output.txt"
    text_na = is_all_na(txt_path)

    return has, text_na


def check_report(report_dir: Path):
    # Determine if a report is valid, and provide reason if invalid
    # Skip directories with confirmation prefix
    if "(" in report_dir.name:
        return False, "confirm_prefix", {}

    # Ensure output directory exists
    if not (report_dir / "output").exists():
        return False, "no_output_dir", {}

    has, text_na = scan_report(report_dir)
    # If all data types are missing and text report is blank, mark invalid
    if all(v == 0 for v in has.values()) and text_na:
        return False, "all_data_missing", has

    # Otherwise, report is valid
    return True, "", has


def scan_raw_dataset(raw_dir: Path):
    # Iterate over patients and their reports to classify each report
    valid = []
    invalid = []
    for patient_dir in sorted(raw_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for report_dir in sorted(patient_dir.iterdir()):
            if not report_dir.is_dir():
                continue
            ok, reason, has = check_report(report_dir)
            record = {
                "patient_id": patient_dir.name,
                "report_id": report_dir.name,
                "report_path": str(report_dir),
            }
            if ok:
                record["has"] = has
                valid.append(record)
            else:
                record["reason"] = reason
                record["has"] = has
                invalid.append(record)
    return valid, invalid


def verify_reports(raw_dir: str = "../../raw_dataset", 
                  out_dir: str = "../../", 
                  dry_run: bool = False) -> tuple[list, list]:
    """
    Verify reports in the raw dataset directory
    
    Args:
        raw_dir (str): Path to the original raw_dataset directory
        out_dir (str): Directory where valid/invalid JSON files will be written
        dry_run (bool): Only display statistics without writing files
    
    Returns:
        tuple[list, list]: (valid_reports, invalid_reports)
    """
    # 轉換路徑
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()
    
    # 執行驗證
    valid, invalid = scan_raw_dataset(raw_dir)

    # 輸出結果
    print(f"Total reports: {len(valid) + len(invalid)}")
    print(f"  ✔ valid    : {len(valid)}")
    print(f"  ✘ invalid  : {len(invalid)}")
    
    if not dry_run:
        # 寫入 JSON 檔案
        (out_dir / "valid_reports.json").write_text(
            json.dumps(valid, indent=2, ensure_ascii=False)
        )
        (out_dir / "invalid_reports.json").write_text(
            json.dumps(invalid, indent=2, ensure_ascii=False)
        )
        print(f"JSON files written to {out_dir}")
    
    return valid, invalid


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", default="../../raw_dataset",
        help="Path to the original raw_dataset directory"
    )
    parser.add_argument(
        "--out_dir", default="../../",
        help="Directory where valid/invalid JSON files will be written"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only display statistics without writing files"
    )
    args = parser.parse_args()
    
    # 直接調用 verify_reports 函數
    verify_reports(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
