#!/usr/bin/env python3
"""
MIMIC-CXR Dataset Analysis Script
Analyzes images, reports, metadata and identifies discrepancies
"""

import os
import pandas as pd
from pathlib import Path
from collections import Counter

# Configuration
BASE_DIR = Path(
    "/Users/mac/Documents/NTNU/Ntnu_Courses/advanced_project_work/LLaVA-Rad/data"
)
IMAGES_DIR = BASE_DIR / "mimic-cxr-images"
REPORTS_DIR = BASE_DIR / "mimic-cxr-reports"
METADATA_FILE = BASE_DIR / "metadata.csv"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def count_files_recursive(directory, extensions=None):
    """Count files recursively in a directory"""
    count = 0
    files_by_ext = Counter()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("."):  # Skip hidden files
                continue
            ext = os.path.splitext(file)[1].lower()
            if extensions is None or ext in extensions:
                count += 1
                files_by_ext[ext] += 1

    return count, files_by_ext


def get_patient_folders(directory):
    """Get all patient folders (p10, p11, etc.)"""
    if not directory.exists():
        return []

    patient_folders = []
    for item in directory.iterdir():
        if item.is_dir() and item.name.startswith("p"):
            patient_folders.append(item.name)

    return sorted(patient_folders)


def count_patients_and_studies(directory):
    """Count unique patients and studies"""
    patients = set()
    studies = set()

    for root, dirs, files in os.walk(directory):
        path_parts = Path(root).parts

        # Find patient IDs (format: p10xxxxxx)
        for part in path_parts:
            if part.startswith("p") and len(part) > 3:
                patients.add(part)

        # Find study IDs (format: s12345678)
        for part in path_parts:
            if part.startswith("s") and len(part) > 3:
                studies.add(part)

    return len(patients), len(studies)


def analyze_metadata():
    """Analyze the metadata CSV file"""
    print_section("METADATA ANALYSIS")

    if not METADATA_FILE.exists():
        print(f"❌ Metadata file not found: {METADATA_FILE}")
        return None

    try:
        df = pd.read_csv(METADATA_FILE)

        print(f"Total rows in metadata: {len(df):,}")
        print(f"\nColumns in metadata:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        # Analyze view positions if column exists
        view_cols = [
            col
            for col in df.columns
            if "view" in col.lower() or "position" in col.lower()
        ]
        if view_cols:
            print(f"\n📊 View Position Analysis:")
            for col in view_cols:
                print(f"\n  Column: {col}")
                value_counts = df[col].value_counts()
                for view, count in value_counts.items():
                    print(f"    - {view}: {count:,} ({count/len(df)*100:.2f}%)")

        # Analyze patient and study IDs
        patient_cols = [
            col
            for col in df.columns
            if "patient" in col.lower() or "subject" in col.lower()
        ]
        study_cols = [col for col in df.columns if "study" in col.lower()]

        if patient_cols:
            print(f"\n👤 Unique Patients: {df[patient_cols[0]].nunique():,}")

        if study_cols:
            print(f"📋 Unique Studies: {df[study_cols[0]].nunique():,}")

        # Check for DICOM IDs
        dicom_cols = [col for col in df.columns if "dicom" in col.lower()]
        if dicom_cols:
            print(f"🖼️  Unique DICOM IDs: {df[dicom_cols[0]].nunique():,}")

        # Check for orientation
        orientation_cols = [col for col in df.columns if "orientation" in col.lower()]
        if orientation_cols:
            print(f"\n🔄 Orientation Analysis:")
            for col in orientation_cols:
                print(f"\n  Column: {col}")
                value_counts = df[col].value_counts()
                for orient, count in value_counts.items():
                    print(f"    - {orient}: {count:,} ({count/len(df)*100:.2f}%)")

        return df

    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return None


def analyze_images():
    """Analyze the images directory"""
    print_section("IMAGE DATASET ANALYSIS")

    if not IMAGES_DIR.exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return

    print(f"📁 Images directory: {IMAGES_DIR}")

    # Count total images
    image_extensions = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    total_images, files_by_ext = count_files_recursive(IMAGES_DIR, image_extensions)

    print(f"\n🖼️  Total images: {total_images:,}")
    print(f"\nImages by file type:")
    for ext, count in files_by_ext.most_common():
        print(f"  - {ext}: {count:,}")

    # Get patient folders
    patient_folders = get_patient_folders(IMAGES_DIR)
    print(f"\n👥 Patient folder groups: {len(patient_folders)}")
    print(
        f"   Range: {patient_folders[0] if patient_folders else 'N/A'} to {patient_folders[-1] if patient_folders else 'N/A'}"
    )

    # Count patients and studies
    num_patients, num_studies = count_patients_and_studies(IMAGES_DIR)
    print(f"\n👤 Unique patient folders: {num_patients:,}")
    print(f"📋 Unique study folders: {num_studies:,}")

    # Average images per patient/study
    if num_patients > 0:
        print(f"\n📊 Average images per patient: {total_images/num_patients:.2f}")
    if num_studies > 0:
        print(f"📊 Average images per study: {total_images/num_studies:.2f}")


def analyze_reports():
    """Analyze the reports directory"""
    print_section("REPORT DATASET ANALYSIS")

    if not REPORTS_DIR.exists():
        print(f"❌ Reports directory not found: {REPORTS_DIR}")
        return

    print(f"📁 Reports directory: {REPORTS_DIR}")

    # Count total reports
    report_extensions = {".txt", ".text"}
    total_reports, files_by_ext = count_files_recursive(REPORTS_DIR, report_extensions)

    print(f"\n📄 Total report files: {total_reports:,}")
    print(f"\nReports by file type:")
    for ext, count in files_by_ext.most_common():
        print(f"  - {ext}: {count:,}")

    # Get patient folders
    patient_folders = get_patient_folders(REPORTS_DIR)
    print(f"\n👥 Patient folder groups: {len(patient_folders)}")
    print(
        f"   Range: {patient_folders[0] if patient_folders else 'N/A'} to {patient_folders[-1] if patient_folders else 'N/A'}"
    )

    # Count patients and studies
    num_patients, num_studies = count_patients_and_studies(REPORTS_DIR)
    print(f"\n👤 Unique patient folders: {num_patients:,}")
    print(f"📋 Unique study folders: {num_studies:,}")

    # Average reports per patient/study
    if num_patients > 0:
        print(f"\n📊 Average reports per patient: {total_reports/num_patients:.2f}")
    if num_studies > 0:
        print(f"📊 Average reports per study: {total_reports/num_studies:.2f}")


def compare_datasets(metadata_df):
    """Compare images, reports, and metadata"""
    print_section("DATASET COMPARISON & DISCREPANCIES")

    # Count images and reports
    total_images, _ = count_files_recursive(
        IMAGES_DIR, {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    )
    total_reports, _ = count_files_recursive(REPORTS_DIR, {".txt", ".text"})

    print(f"📊 Dataset Size Comparison:")
    print(f"  - Images: {total_images:,}")
    print(f"  - Reports: {total_reports:,}")

    if metadata_df is not None:
        print(f"  - Metadata rows: {len(metadata_df):,}")

        # Calculate differences
        diff_img_metadata = total_images - len(metadata_df)
        diff_reports_metadata = total_reports - len(metadata_df)
        diff_img_reports = total_images - total_reports

        print(f"\n🔍 Differences:")
        print(f"  - Images vs Metadata: {diff_img_metadata:+,}")
        print(f"  - Reports vs Metadata: {diff_reports_metadata:+,}")
        print(f"  - Images vs Reports: {diff_img_reports:+,}")

        if diff_img_reports != 0:
            if diff_img_reports > 0:
                print(
                    f"  ⚠️  There are {abs(diff_img_reports):,} more images than reports"
                )
            else:
                print(
                    f"  ⚠️  There are {abs(diff_img_reports):,} more reports than images"
                )

    # Patient folder comparison
    img_patients = set(get_patient_folders(IMAGES_DIR))
    report_patients = set(get_patient_folders(REPORTS_DIR))

    print(f"\n👥 Patient Folder Comparison:")
    print(f"  - Patient folders in images: {len(img_patients)}")
    print(f"  - Patient folders in reports: {len(report_patients)}")

    only_in_images = img_patients - report_patients
    only_in_reports = report_patients - img_patients

    if only_in_images:
        print(f"  ⚠️  Patient folders only in images: {sorted(only_in_images)}")

    if only_in_reports:
        print(f"  ⚠️  Patient folders only in reports: {sorted(only_in_reports)}")

    if not only_in_images and not only_in_reports:
        print(f"  ✅ Patient folders are identical in both datasets")


def analyze_mixed_view_studies(metadata_df):
    """Analyze studies that have both frontal and lateral views"""
    print_section("MIXED VIEW STUDIES ANALYSIS")

    if metadata_df is None:
        print("❌ Cannot calculate without metadata")
        return

    # Find study_id column
    study_cols = [col for col in metadata_df.columns if "study" in col.lower()]
    view_cols = [
        col
        for col in metadata_df.columns
        if "view" in col.lower() or "position" in col.lower()
    ]

    if not study_cols or not view_cols:
        print("❌ Cannot find study_id or view columns")
        return

    study_col = study_cols[0]
    view_col = view_cols[0]

    # Group by study and get unique views per study
    study_views = (
        metadata_df.groupby(study_col)[view_col]
        .apply(lambda x: set(x.dropna()))
        .reset_index()
    )
    study_views.columns = [study_col, "views"]

    # Identify different types of studies
    frontal_only = []
    lateral_only = []
    mixed_studies = []

    for idx, row in study_views.iterrows():
        views = row["views"]
        has_frontal = any(v in ["AP", "PA", "FRONTAL"] for v in views)
        has_lateral = any(
            "LATERAL" in str(v) or v in ["LL", "LAO", "RAO"] for v in views
        )

        if has_frontal and has_lateral:
            mixed_studies.append(row[study_col])
        elif has_frontal:
            frontal_only.append(row[study_col])
        elif has_lateral:
            lateral_only.append(row[study_col])

    total_studies = len(study_views)

    print(f"📊 Study Composition by View Type:")
    print(
        f"  ✅ Frontal-only studies: {len(frontal_only):,} ({len(frontal_only)/total_studies*100:.2f}%)"
    )
    print(
        f"  ⚠️  Lateral-only studies: {len(lateral_only):,} ({len(lateral_only)/total_studies*100:.2f}%)"
    )
    print(
        f"  🔀 Mixed (Frontal + Lateral) studies: {len(mixed_studies):,} ({len(mixed_studies)/total_studies*100:.2f}%)"
    )
    print(f"  📋 Total studies: {total_studies:,}")

    # Analyze images in mixed studies
    mixed_study_images = metadata_df[metadata_df[study_col].isin(mixed_studies)]
    frontal_in_mixed = mixed_study_images[
        mixed_study_images[view_col].isin(["AP", "PA", "FRONTAL"])
    ]
    lateral_in_mixed = mixed_study_images[
        mixed_study_images[view_col].str.contains("LATERAL", case=False, na=False)
        | mixed_study_images[view_col].isin(["LL", "LAO", "RAO"])
    ]

    print(f"\n🔀 Images in Mixed Studies:")
    print(f"  - Total images in mixed studies: {len(mixed_study_images):,}")
    print(f"  - Frontal views in mixed studies: {len(frontal_in_mixed):,}")
    print(f"  - Lateral views in mixed studies: {len(lateral_in_mixed):,}")

    print(f"\n💡 Implications for Annotations:")
    print(f"  - {len(mixed_studies):,} reports describe BOTH frontal AND lateral views")
    print(f"  - You need to decide: Use only frontal images from these reports?")
    print(f"  - Or exclude mixed-view studies entirely for cleaner training data?")

    # Calculate potential dataset sizes
    frontal_all = metadata_df[metadata_df[view_col].isin(["AP", "PA", "FRONTAL"])]
    frontal_from_frontal_only = frontal_all[frontal_all[study_col].isin(frontal_only)]
    frontal_from_mixed = frontal_all[frontal_all[study_col].isin(mixed_studies)]

    print(f"\n📊 Dataset Options:")
    print(f"  Option 1 (All frontal images):")
    print(f"    - Total: {len(frontal_all):,} frontal images")
    print(f"    - From frontal-only studies: {len(frontal_from_frontal_only):,}")
    print(f"    - From mixed studies: {len(frontal_from_mixed):,}")
    print(f"    - Advantage: More training data")
    print(f"    - Risk: Reports mention lateral findings not visible in frontal image")

    print(f"\n  Option 2 (Frontal-only studies):")
    print(f"    - Total: {len(frontal_from_frontal_only):,} frontal images")
    print(f"    - Only from studies with no lateral views")
    print(f"    - Advantage: Cleaner data, report fully describes the image")
    print(f"    - Trade-off: {len(frontal_from_mixed):,} fewer samples")


def analyze_image_dimensions(metadata_df):
    """Analyze image dimensions"""
    print_section("IMAGE DIMENSIONS ANALYSIS")

    if metadata_df is None:
        print("❌ Cannot calculate without metadata")
        return

    # Check for dimension columns
    row_cols = [
        col for col in metadata_df.columns if col.lower() in ["rows", "row", "height"]
    ]
    col_cols = [
        col
        for col in metadata_df.columns
        if col.lower() in ["columns", "cols", "column", "width"]
    ]

    if not row_cols or not col_cols:
        print("❌ Cannot find image dimension columns")
        return

    row_col = row_cols[0]
    col_col = col_cols[0]

    # Get unique dimensions
    dimensions = metadata_df[[row_col, col_col]].copy()
    dimensions["dimension"] = (
        dimensions[row_col].astype(str) + "x" + dimensions[col_col].astype(str)
    )

    print(f"📐 Image Dimensions Statistics:")
    print(f"  Column for height: '{row_col}'")
    print(f"  Column for width: '{col_col}'")

    # Basic statistics
    print(f"\n📊 Height (Rows):")
    print(f"  - Min: {dimensions[row_col].min():,} pixels")
    print(f"  - Max: {dimensions[row_col].max():,} pixels")
    print(f"  - Mean: {dimensions[row_col].mean():.2f} pixels")
    print(f"  - Median: {dimensions[row_col].median():.2f} pixels")

    print(f"\n📊 Width (Columns):")
    print(f"  - Min: {dimensions[col_col].min():,} pixels")
    print(f"  - Max: {dimensions[col_col].max():,} pixels")
    print(f"  - Mean: {dimensions[col_col].mean():.2f} pixels")
    print(f"  - Median: {dimensions[col_col].median():.2f} pixels")

    # Most common dimensions
    print(f"\n📊 Most Common Image Sizes:")
    top_dimensions = dimensions["dimension"].value_counts().head(10)
    for i, (dim, count) in enumerate(top_dimensions.items(), 1):
        print(
            f"  {i}. {dim} pixels: {count:,} images ({count/len(dimensions)*100:.2f}%)"
        )

    # Aspect ratio analysis
    dimensions["aspect_ratio"] = dimensions[col_col] / dimensions[row_col]
    print(f"\n📊 Aspect Ratio:")
    print(f"  - Min: {dimensions['aspect_ratio'].min():.3f}")
    print(f"  - Max: {dimensions['aspect_ratio'].max():.3f}")
    print(f"  - Mean: {dimensions['aspect_ratio'].mean():.3f}")
    print(f"  - Median: {dimensions['aspect_ratio'].median():.3f}")

    # Categorize by size
    small = len(dimensions[(dimensions[row_col] < 1000) | (dimensions[col_col] < 1000)])
    medium = len(
        dimensions[
            (dimensions[row_col] >= 1000)
            & (dimensions[row_col] < 2000)
            & (dimensions[col_col] >= 1000)
            & (dimensions[col_col] < 2000)
        ]
    )
    large = len(
        dimensions[(dimensions[row_col] >= 2000) | (dimensions[col_col] >= 2000)]
    )

    print(f"\n📊 Size Categories:")
    print(f"  - Small (< 1000px): {small:,} images ({small/len(dimensions)*100:.2f}%)")
    print(
        f"  - Medium (1000-2000px): {medium:,} images ({medium/len(dimensions)*100:.2f}%)"
    )
    print(f"  - Large (≥ 2000px): {large:,} images ({large/len(dimensions)*100:.2f}%)")


def calculate_annotation_potential(metadata_df):
    """Calculate potential for creating annotations"""
    print_section("ANNOTATION CREATION POTENTIAL")

    if metadata_df is None:
        print("❌ Cannot calculate without metadata")
        return

    # Find frontal views (AP/PA)
    view_cols = [
        col
        for col in metadata_df.columns
        if "view" in col.lower() or "position" in col.lower()
    ]

    if view_cols:
        view_col = view_cols[0]
        frontal_views = metadata_df[metadata_df[view_col].isin(["AP", "PA", "FRONTAL"])]
        lateral_views = metadata_df[
            metadata_df[view_col].str.contains("LATERAL", case=False, na=False)
        ]
        other_views = metadata_df[
            ~metadata_df[view_col].isin(["AP", "PA", "FRONTAL"])
            & ~metadata_df[view_col].str.contains("LATERAL", case=False, na=False)
        ]

        print(f"📊 View Distribution for Annotations:")
        print(
            f"  ✅ Frontal views (AP/PA): {len(frontal_views):,} ({len(frontal_views)/len(metadata_df)*100:.2f}%)"
        )
        print(
            f"  ⚠️  Lateral views: {len(lateral_views):,} ({len(lateral_views)/len(metadata_df)*100:.2f}%)"
        )
        print(
            f"  ❓ Other views: {len(other_views):,} ({len(other_views)/len(metadata_df)*100:.2f}%)"
        )

        print(f"\n💡 Recommendation:")
        print(f"  - Use {len(frontal_views):,} frontal view images for annotations")
        print(f"  - Exclude {len(lateral_views):,} lateral view images")

        # Suggested split (80/10/10)
        print(f"\n📝 Suggested Split (80/10/10):")
        print(f"  - Training: {int(len(frontal_views)*0.8):,} samples")
        print(f"  - Validation: {int(len(frontal_views)*0.1):,} samples")
        print(f"  - Test: {int(len(frontal_views)*0.1):,} samples")

        # Alternative split (70/15/15)
        print(f"\n📝 Alternative Split (70/15/15):")
        print(f"  - Training: {int(len(frontal_views)*0.7):,} samples")
        print(f"  - Validation: {int(len(frontal_views)*0.15):,} samples")
        print(f"  - Test: {int(len(frontal_views)*0.15):,} samples")


def generate_summary():
    """Generate a summary report"""
    print_section("SUMMARY")

    print(
        """
📋 Analysis Complete!

Key Findings:
1. Check the view distribution to identify frontal (AP/PA) vs lateral images
2. Verify that image counts match report counts for paired data
3. Review patient folder consistency between images and reports
4. Use the metadata file to filter for appropriate views for annotations

Next Steps:
1. Filter metadata for AP/PA views only
2. Match images with corresponding reports using patient/study/DICOM IDs
3. Apply rule-based extraction to reports
4. Create JSON annotations in LLaVA-Rad format
5. Split data according to chosen percentages (80/10/10 or 70/15/15)
    """
    )


def main():
    """Main analysis function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MIMIC-CXR Dataset Analysis Tool" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    # Check if base directory exists
    if not BASE_DIR.exists():
        print(f"\n❌ Base directory not found: {BASE_DIR}")
        return

    print(f"\n📁 Base directory: {BASE_DIR}")

    # Run all analyses
    analyze_images()
    analyze_reports()
    metadata_df = analyze_metadata()
    analyze_image_dimensions(metadata_df)
    compare_datasets(metadata_df)
    analyze_mixed_view_studies(metadata_df)
    calculate_annotation_potential(metadata_df)
    generate_summary()

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
