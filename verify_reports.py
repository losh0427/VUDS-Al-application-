import argparse
import json
import re
from pathlib import Path

LABEL_KEYS = [
    "Detrusor_instability", "Flow_pattern", "EMG_ES_relaxation",
    "Trabeculation", "Diverticulum", "Cystocele", "VUR",
    "Bladder_neck_relaxation", "External_sphincter_relaxation",
    "Pelvic_floor_relaxation"
]
DATA_TYPES = ["pftg", "pfus", "pfs", "xray"]


def is_all_na(txt_path: Path) -> bool:
    """判斷 output.txt 是否全部 NA / N/A / 空"""
    if not txt_path.is_file():
        return True
    content = txt_path.read_text(encoding="utf-8", errors="ignore").lower()
    if not any(k.lower() in content for k in LABEL_KEYS):
        return True
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    blanks = sum(
        bool(re.search(r'\bna\b|n/a|none|^\s*$', ln.split(":")[-1], re.I))
        for ln in lines
    )
    return blanks == len(lines)


def scan_report(report_dir: Path):
    """
    掃描單份報告，回傳含有哪些 data 類型與 label 狀態
    """
    out_root = report_dir / "output"
    has = {}
    # 檢查每種 data type 檔案數量
    for t in DATA_TYPES:
        cnt = len(list((out_root / t).glob("*.*"))) if (out_root / t).exists() else 0
        has[t] = cnt

    # 判斷文字報告
    txt_path = out_root / "result" / "output.txt"
    text_na = is_all_na(txt_path)

    return has, text_na


def check_report(report_dir: Path):
    """
    回傳 (是否有效, 無效原因, has)
    missing only if all DATA_TYPES and text are missing
    """
    if "(請確認" in report_dir.name:
        return False, "confirm_prefix", {}

    if not (report_dir / "output").exists():
        return False, "no_output_dir", {}

    has, text_na = scan_report(report_dir)
    # 若所有四種都沒，以及文字報告全 NA，則 invalid
    if all(v == 0 for v in has.values()) and text_na:
        return False, "all_data_missing", has

    return True, "", has


def scan_raw_dataset(raw_dir: Path):
    valid, invalid = [], []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="../../raw_dataset",
                        help="原始 raw_dataset 路徑")
    parser.add_argument("--out_dir", default="../../",
                        help="valid/invalid json 要寫到哪個資料夾")
    parser.add_argument("--dry-run", action="store_true",
                        help="只顯示統計，不寫檔")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    valid, invalid = scan_raw_dataset(raw_dir)

    print(f"Total reports: {len(valid)+len(invalid)}")
    print(f"  ✔ valid    : {len(valid)}")
    print(f"  ✘ invalid  : {len(invalid)}")

    if not args.dry_run:
        (out_dir / "valid_reports.json").write_text(
            json.dumps(valid, indent=2, ensure_ascii=False)
        )
        (out_dir / "invalid_reports.json").write_text(
            json.dumps(invalid, indent=2, ensure_ascii=False)
        )
        print(f"JSON written to {out_dir}")


if __name__ == "__main__":
    main()
