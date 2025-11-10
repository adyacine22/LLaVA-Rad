import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

CONFIG_FILE_NAME = "config.json"
CONFIG_FILE_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILE_NAME)


@dataclass
class AnnotationConfig:
    section_headers: Dict[str, str]
    temporal_patterns: List[str]
    demographic_patterns: List[str]
    chexpert_labels: Dict[str, float]
    chexpert_label_keywords: Dict[str, Dict[str, List[str]]]
    data_dir: str
    metadata_file: str
    reports_dir: str
    image_dir: str
    output_dir: str
    frontal_views: List[str]
    lateral_view_keywords: List[str]
    train_size: float
    val_size: float
    seed: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationConfig":
        paths = data.get("paths", {})
        filters = data.get("filters", {})
        split = data.get("split", {})
        return cls(
            section_headers=data.get("section_headers", {}),
            temporal_patterns=data.get("temporal_patterns", []),
            demographic_patterns=data.get("demographic_patterns", []),
            chexpert_labels=data.get("chexpert_labels", {}),
            chexpert_label_keywords=data.get("chexpert_label_keywords", {}),
            data_dir=paths.get("data_dir", "data"),
            metadata_file=paths.get("metadata_file", "data/metadata.csv"),
            reports_dir=paths.get("reports_dir", "data/mimic-cxr-reports/mimic"),
            image_dir=paths.get("image_dir", "data/mimic-cxr-images/mimic"),
            output_dir=paths.get("output_dir", "annotation_dataset/output/annotations"),
            frontal_views=filters.get("frontal_views", ["AP", "PA"]),
            lateral_view_keywords=filters.get("lateral_view_keywords", ["LATERAL", "LL", "RL"]),
            train_size=split.get("train", 0.8),
            val_size=split.get("val", 0.1),
            seed=split.get("seed", 42),
        )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    target_path = os.path.abspath(config_path) if config_path else CONFIG_FILE_DEFAULT_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Configuration file not found at {target_path}")
    with open(target_path, "r") as f:
        return json.load(f)


def resolve_path(path_value: str, project_root: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(project_root, path_value)


def clean_text(text: str, config: AnnotationConfig) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text = " ".join(text.split())
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    for pattern in config.temporal_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    for pattern in config.demographic_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    if not text or not text.strip():
        return None
    return text.strip()


def extract_sections(report_text: str, config: AnnotationConfig) -> Dict[str, Optional[str]]:
    sections: Dict[str, Optional[str]] = {
        "examination": None,
        "indication": None,
        "findings": None,
        "impression": None,
    }
    if not report_text or not isinstance(report_text, str):
        return sections
    matches = []
    for name, pattern in config.section_headers.items():
        for match in re.finditer(pattern, report_text):
            matches.append({"name": name, "start": match.start(), "end": match.end()})
    if not matches:
        sections["findings"] = report_text
        return sections
    matches.sort(key=lambda x: x["start"])
    for i, match in enumerate(matches):
        section_name = match["name"]
        content_start = match["end"]
        if i + 1 < len(matches):
            content_end = matches[i + 1]["start"]
        else:
            content_end = len(report_text)
        content = report_text[content_start:content_end].strip()
        sections[section_name] = content if content else None
    return sections


def extract_chexpert_labels(text: str, config: AnnotationConfig) -> Dict[str, float]:
    if not text or not isinstance(text, str):
        return {label: np.nan for label in config.chexpert_labels.keys()}
    labels = {}
    text_lower = text.lower()
    for label_name, keywords in config.chexpert_label_keywords.items():
        positive_found = any(kw in text_lower for kw in keywords.get("positive", []))
        negative_found = any(kw in text_lower for kw in keywords.get("negative", []))
        uncertain_found = any(kw in text_lower for kw in keywords.get("uncertain", []))
        if negative_found and not positive_found:
            labels[label_name] = 0.0
        elif uncertain_found:
            labels[label_name] = -1.0
        elif positive_found:
            labels[label_name] = 1.0
        else:
            labels[label_name] = np.nan
    return labels


def process_report(report_text: str, config: AnnotationConfig) -> Dict[str, Optional[str]]:
    try:
        extracted_sections = extract_sections(report_text, config)
        return {
            name: clean_text(content, config) if content else None
            for name, content in extracted_sections.items()
        }
    except Exception as e:
        print(f"Error processing report: {e}")
        return {
            "examination": None,
            "indication": None,
            "findings": None,
            "impression": None,
        }


def generate_conversations(gpt_response_text: str) -> List[Dict[str, str]]:
    if not gpt_response_text or not isinstance(gpt_response_text, str):
        return []
    return [
        {
            "from": "human",
            "value": "<image>\nDescribe the findings of the chest x-ray.\n",
        },
        {"from": "gpt", "value": gpt_response_text},
    ]


def split_data(annotations: List[Dict[str, Any]], config: AnnotationConfig) -> tuple:
    np.random.seed(config.seed)
    patient_groups = defaultdict(list)
    for ann in annotations:
        patient_id = ann["id"].split("_")[0]
        patient_groups[patient_id].append(ann)
    patient_ids = list(patient_groups.keys())
    np.random.shuffle(patient_ids)
    train_split = int(len(patient_ids) * config.train_size)
    val_split = int(len(patient_ids) * (config.train_size + config.val_size))
    train_patient_ids = set(patient_ids[:train_split])
    val_patient_ids = set(patient_ids[train_split:val_split])
    test_patient_ids = set(patient_ids[val_split:])
    train_annotations = []
    val_annotations = []
    test_annotations = []
    for ann in annotations:
        patient_id = ann["id"].split("_")[0]
        if patient_id in train_patient_ids:
            train_annotations.append(ann)
        elif patient_id in val_patient_ids:
            val_annotations.append(ann)
        else:
            test_annotations.append(ann)
    return train_annotations, val_annotations, test_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIMIC-CXR annotations.")
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to a custom annotation config JSON file.",
    )
    return parser.parse_args()


def main(config_path: Optional[str] = None) -> None:
    config_data = load_config(config_path)
    config = AnnotationConfig.from_dict(config_data)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = resolve_path(config.data_dir, project_root)
    metadata_file = resolve_path(config.metadata_file, project_root)
    reports_dir = resolve_path(config.reports_dir, project_root)
    image_dir = resolve_path(config.image_dir, project_root)
    output_dir = resolve_path(config.output_dir, project_root)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using config: {config_path or CONFIG_FILE_DEFAULT_PATH}")
    print("Starting the annotation pipeline...")
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Looking for metadata at: {metadata_file}")
    print(f"Looking for reports at: {reports_dir}")
    print(f"Looking for images at: {image_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        print(f"Available files in {data_dir}: {os.listdir(data_dir)}")
        return
    if not os.path.exists(reports_dir):
        print(f"Warning: Reports directory not found at {reports_dir}")
    if not os.path.exists(image_dir):
        print(f"Warning: Images directory not found at {image_dir}")

    print("\nLoading metadata from CSV...")
    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"Metadata loaded successfully. Shape: {metadata_df.shape}")
        print(f"Columns: {metadata_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    print("\nIdentifying frontal-only studies...")
    study_views = (
        metadata_df.groupby("study_id")["ViewPosition"].apply(set).reset_index()
    )
    study_views["has_lateral"] = study_views["ViewPosition"].apply(
        lambda views: any(view in views for view in config.lateral_view_keywords)
    )
    frontal_only_studies = study_views[~study_views["has_lateral"]]["study_id"].tolist()
    print(f"Found {len(frontal_only_studies)} frontal-only studies")

    print("\nFiltering images from frontal-only studies...")
    frontal_images_df = metadata_df[
        (metadata_df["study_id"].isin(frontal_only_studies))
        & (metadata_df["ViewPosition"].isin(config.frontal_views))
    ]
    print(f"Found {len(frontal_images_df)} frontal images from frontal-only studies")
    print(f"Expected: ~29,194 images")

    subset_df = frontal_images_df.reset_index(drop=True)
    print(f"Processing {len(subset_df)} frontal images...")

    annotations: List[Dict[str, Any]] = []
    failed_reports = 0
    successful_reports = 0
    skipped_reports = 0

    for idx, row in tqdm(
        subset_df.iterrows(), total=len(subset_df), desc="Processing images"
    ):
        try:
            subject_id = str(int(row["subject_id"])).zfill(8)
            study_id = str(int(row["study_id"])).zfill(8)
            dicom_id = str(row["dicom_id"])
            subject_prefix = f"p{subject_id[:2]}"
            subject_folder = f"p{subject_id}"
            report_filename = f"s{study_id}.txt"
            report_path = os.path.join(
                reports_dir, subject_prefix, subject_folder, report_filename
            )

            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                    report_text = f.read()
                successful_reports += 1
            else:
                failed_reports += 1
                continue

            cleaned_sections = process_report(report_text, config)
            findings = cleaned_sections.get("findings")
            impression = cleaned_sections.get("impression")

            gpt_response_text = findings or impression

            if not gpt_response_text:
                skipped_reports += 1
                continue

            image_path_relative = os.path.join(
                "mimic",
                subject_prefix,
                subject_folder,
                f"s{study_id}",
                f"{dicom_id}.jpg",
            )

            annotation = {
                "id": f"{subject_id}_{study_id}",
                "image": image_path_relative,
                "generate_method": "rule-based",
                "reason": cleaned_sections.get("indication"),
                "impression": impression,
                "indication": None,
                "history": None,
                "view": row.get("ViewPosition", "AP"),
                "orientation": row.get("PatientOrientation", "Erect"),
                "chexpert_labels": extract_chexpert_labels(gpt_response_text, config),
                "conversations": generate_conversations(gpt_response_text),
            }
            annotations.append(annotation)

        except Exception as e:
            print(f"\n❌ Error processing row {idx}: {e}")
            continue

    print(f"\n✅ Successfully processed {successful_reports} reports")
    print(f"❌ Failed to find {failed_reports} reports")
    print(f"⏭️ Skipped {skipped_reports} reports due to empty findings")
    print(f"📊 Generated {len(annotations)} valid annotations")

    if len(annotations) == 0:
        print(
            "⚠️  No annotations were generated. Please check your data paths and report files."
        )
        return

    train_annotations, val_annotations, test_annotations = split_data(annotations, config)
    print(
        f"\n📈 Splitting data:\n  - Training: {len(train_annotations)} samples\n  - Validation: {len(val_annotations)} samples\n  - Test: {len(test_annotations)} samples"
    )

    for split_name, split_annotations in zip(
        ["train", "val", "test"], [train_annotations, val_annotations, test_annotations]
    ):
        output_file = os.path.join(output_dir, f"{split_name}.json")
        print(f"\n💾 Saving {len(split_annotations)} annotations to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(split_annotations, f, indent=2, default=str)
        print(f"✅ Saved: {output_file}")

    print("\n" + "=" * 80)
    print("ANNOTATION GENERATION STATISTICS")
    print("=" * 80)
    total_annotations = (
        len(train_annotations) + len(val_annotations) + len(test_annotations)
    )
    print(f"Total annotations: {total_annotations}")
    print(
        f"Training set: {len(train_annotations)} ({len(train_annotations)/total_annotations*100:.1f}%)"
    )
    print(
        f"Validation set: {len(val_annotations)} ({len(val_annotations)/total_annotations*100:.1f}%)"
    )
    print(
        f"Test set: {len(test_annotations)} ({len(test_annotations)/total_annotations*100:.1f}%)"
    )

    stats = {
        "total_annotations": total_annotations,
        "train_count": len(train_annotations),
        "val_count": len(val_annotations),
        "test_count": len(test_annotations),
        "successful_reports": successful_reports,
        "failed_reports": failed_reports,
        "skipped_reports": skipped_reports,
        "output_files": {
            "train": os.path.join(output_dir, "train.json"),
            "val": os.path.join(output_dir, "val.json"),
            "test": os.path.join(output_dir, "test.json"),
        },
    }
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Statistics saved to: {stats_file}")

    print("\n" + "=" * 80)
    print("✅ Pipeline execution complete!")
    print("=" * 80)

    print("\n--- Example Annotation from Train Set ---")
    if train_annotations:
        example = train_annotations[0].copy()
        if "chexpert_labels" in example:
            example["chexpert_labels"] = {
                k: (None if pd.isna(v) else v)
                for k, v in example["chexpert_labels"].items()
            }
        print(json.dumps(example, indent=2))
    print("--- End of Example ---")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
