# LLaVA-Rad MIMIC-CXR Annotations

## Abstract

LLaVA-Rad MIMIC-CXR features more accurate section extractions from MIMIC-CXR free-text radiology reports. Traditionally, rule-based methods were used to extract sections such as the reason for exam, findings, and impression. However, these approaches often fail due to inconsistencies in report structure and clinical language. In this work, we leverage GPT-4 to extract these sections more reliably, adding 237,073 image-text pairs to the training split and 1,952 pairs to the validation split. This enhancement afforded the development and fine-tuning of LLaVA-Rad, a multimodal large language model (LLM) tailored for radiology applications, achieving improved performance on report generation tasks.

This resource is provided to support reproducibility and for the benefit of the research community, enabling further exploration in vision–language modeling. For more details, please refer to the accompanying paper [1].

## Background

Radiology reports are essential documents written by radiologists to describe the findings observed in radiological images. These reports provide a rich source of clinical information, valuable for both healthcare practices and research. The applications of radiology reports are diverse: they serve as training signals for computer vision models, facilitate the development of clinical decision-making systems, and contribute to building patient profiles for clinical practice and research purposes.

A typical radiology report includes key sections such as Indication (reason for exam), Findings, and Impression. However, reports are often composed in free-form text, with significant variability in structure, phrasing, and terminology. Extracting relevant information from these reports requires processing the text into structured data. Traditionally, rule-based approaches were used for this purpose, but these methods often fall short for several reasons:

1. **Heuristic Complexity**: Rule-based systems rely heavily on predefined patterns and rules, making them fragile and prone to failure when encountering variations.

2. **Inconsistent Formats**: Different radiologists, hospitals, and reporting systems use distinct formats and writing styles, making it difficult to build a one-size-fits-all rule-based solution.

3. **Language Variations**: Language use in reports can differ in tone, length, and phrasing. Spelling variations, typographical errors, and differences in medical terminology further complicate text processing. Additionally, negations (e.g., "no evidence of fracture") and conditional phrases (e.g., "if present, could indicate…") require nuanced handling, which rule-based approaches struggle to manage effectively.

4. **Temporal Comparisons and Historical References**: A unique challenge in radiology reports lies in the frequent use of temporal comparisons. Radiologists often compare current imaging findings to prior studies to track changes over time, such as evaluating the progression or resolution of conditions (e.g., "no interval change compared to the previous study from six months ago"). Reports may mention conditions or findings observed in prior images that are not present in the current image.

These challenges demonstrate the limitations of traditional rule-based methods for processing radiology reports.

## Methods

### Original Data Source

The original data used in this project was obtained from the MIMIC-CXR radiology reports [2].

### Structuring Process

Reports were split into corresponding sections using two approaches: rule-based and GPT-4 based section extraction. Rule-based extraction was carried out using regular expression-based section parsing as directly available in the official MIMIC Code Repository [3]. As a complement to rule-based section extraction, we performed GPT-4 based section extraction. We leveraged a set of prompting techniques to accurately extract the content of the respective radiology report sections. The key elements of our prompting strategy included:

#### 1. Detailed Task Instruction

In the prompt, we provided a detailed task description to guide GPT-4. This included the following specific steps:

**Text Cleaning:**

- **Fixing broken words**: Correct split words resulting from typographical errors or formatting issues.

- **Removing repeated words or redundant phrases**: Remove redundant or repetitive text to ensure the output was concise and easy to interpret.

- **Removing Temporal Mentions**: Our objective in this work was to focus exclusively on information relevant to the current image, thus we instructed GPT-4 to exclude references to prior studies or comparisons (e.g., "Compared to the prior study, no significant interval change was noted").

**Extracting Sections:**

Reorganize the text into appropriate sections: Indication (reason for exam), Findings, and Impression, regardless of the original section names used in the report.

#### 2. In-Context Learning

We included a sample input and desired output within the prompt. This few-shot learning technique enabled the model to better understand the task by observing demonstrations, improving the quality and reliability of the generated outputs.

#### 3. Structured Output in JSON Format

To ensure the generated data could be easily parsed and processed, we instructed the model to present the results in a structured JSON format.

#### GPT-4 Prompt

Below, we provide our GPT-4 prompt:

```
You are an expert medical assistant AI capable of modifying clinical documents to
user specifications. You make minimal changes to the original document to satisfy
user requests. You never add information that is not already directly stated in
the original document.

Extract four sections from the input radiology report: 'Examination', 'Indication',
'Findings' and 'Impression'. Leave an extracted section as null if it does not
exist in the original report. The output should be in JSON format. An Indication
section can refer to the History, Indication or Reason for Study sections in the
original report. Remove any information not directly observable from the current
imaging study. For instance, remove any patient demographic data, past medical
history, or comparison to prior images or studies. The generated 'Findings' and
'Impression' sections should not reference any changes based on prior images,
studies, or external knowledge about the patient. Rewrite such comparisons as a
status observation based only on the current image or study. Remember to remove
any numbering or bullets.

Examples of inputs and expected outputs:

INPUT:
EXAMINATION: XR CHEST AP PORTABLE
INDICATION: Small right apical pneumothorax after lung biopsy.
FINDINGS: Single portable view of the chest was obtained. Copared with 10:42 AM.
The small right apical pneumothorax has decreased slightly in size, the improvement
best appreciated laterally where it now measures 10 mm compared to 14 mm before. At
the lung apex it now measures 1.6 and compared to 2.1 cm previously. A subtle right
apical pulmonary contusion is grossly stable. Minor chest wall emphysema along the
right exilla has not changed significant delay. There is no metastatic shift. No
pleural effusion is evident.

OUTPUT:
{"EXAMINATION": "XR CHEST AP PORTABLE.",
"INDICATION": "Small right apical pneumothorax after lung biopsy.",
"FINDINGS": "Single portable view of the chest was obtained. The small right apical
pneumothorax measures 10mm. At the lung apex it measures 1.6cm. A subtle right apical
pulmonary contusion is grossly stable. Minor chest wall emphysema is noted along the
right exilla. There is no metastatic shift. No pleural effusion is evident.",
"IMPRESSION": null}
```

### Evaluation of GPT-4 Extracted Report Quality

A variety of automated methods were used to evaluate the quality of the extracted sections, with particular focus on the Findings section of the reports. This analysis was carried out using the test split of the reports, comparing GPT-4 extracted Findings with those extracted by the rule-based processing method which served as ground truth. Quality analysis of the language model extracted sections was evaluated using automated radiology report evaluation metrics, including lexical overlap metrics (BLEU, ROUGE-L), traditional factual correctness (CheXbert F1 scores, F1 RadGraph scores), semantic similarity metrics (Gatortron cosine similarity), and CheXprompt, a radiologist aligned large language model-based score [1,4-8]. 

Traditional metrics indicated high similarity between GPT-4 and rule-based extracted sections: BLEU-4 78.8, ROUGE-L 90.9, F1 CheXbert 14 96.7, F1 RadGraph 93.2. Analogous similarity was determined by the Gatortron cosine similarity metric, with average (standard deviation) similarity of 0.928, 0.084. Finally, CheXprompt error counts revealed 0.20 (0.47) average (standard deviation) reports when using the rule-based extraction as a reference, with 82.4% of reports free of any error. 

Overall, GPT-4 reliably extracted sections, with observed qualitative improvements over the rule-based extraction with corrected inadequate spelling, removal of repeated phrases, omission of references to prior examinations, and adequate assignment of content in appropriate sections regardless of the lexical content of the section header. (See Supplementary Table 7 of [1]).

## Data Description

The LLaVA-Rad MIMIC-CXR dataset consists of three JSON files, corresponding to the train, validation, and test splits. Each JSON file contains the extracted sections and metadata from the MIMIC-CXR radiology reports, formatted to be compatible with the LLaVA framework. LLaVA is a widely adopted framework for multi-modal large language models (LLMs) [9]. We recommend checking out the LLaVA-Med repository for further information and applications [10].

The dataset statistics are summarized in the table below:

| MIMIC-CXR | Training | Validation | Test |
|-----------|----------|------------|------|
| Patients | 63,169 | 487 | 289 |
| Studies | 213,365 | 1,733 | 3,041 |
| DICOMs (AP/PA) | 237,972 | 1,959 | 3,403 |
| Rule-based reports | 162,969 | 1,286 | 2,461 |
| GPT-structured reports | 237,073 | 1,952 | - |
| Total data | 400,042 | 3,238 | 2,461 |

*Number of patients, studies and DICOMs in AP/PA views across official MIMIC-CXR splits. Numbers of image-text pairs generated by official rule-based methods and augmented by GPT-4 in training and validation splits, excluding text with no Findings extracted.*

### Dataset Structure

Each JSON file in the dataset contains entries derived from the MIMIC-CXR dataset, with key-value pairs describing:

- **id**: An entry's unique identifier
- **image**: Path to corresponding image in the MIMIC-CXR dataset
- **generate_method**: Extraction method (GPT-4 vs. rule-based)
- **Report sections**: reason for exam, impression, indication, history
- **chexpert_labels**: CheXpert classification labels
- **conversations**: A conversation entry requesting a description of the image by the user, with a GPT response corresponding to the findings section of the report

### Example Entry

Below is an example from the JSON file:

```json
{
  "id": "10003502_50084553",
  "image": "mimic/p10/p10003502/s50084553/70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9.jpg",
  "generate_method": "gpt4",
  "reason": "___ year old woman with heart failure and dyspnea// eval for pulmonary edema.",
  "impression": null,
  "indication": null,
  "history": null,
  "view": "AP",
  "orientation": "Erect",
  "chexpert_labels": {
    "Atelectasis": 1.0,
    "Cardiomegaly": -1.0,
    "Consolidation": NaN,
    "Edema": 1.0,
    "Enlarged Cardiomediastinum": NaN,
    "Fracture": NaN,
    "Lung Lesion": NaN,
    "Lung Opacity": NaN,
    "No Finding": NaN,
    "Pleural Effusion": 1.0,
    "Pleural Other": NaN,
    "Pneumonia": NaN,
    "Pneumothorax": 0.0,
    "Support Devices": NaN
  },
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe the findings of the chest x-ray.\n"
    },
    {
      "from": "gpt",
      "value": "Large right and moderate left pleural effusions and severe bibasilar atelectasis are present. Cardiac silhouette is obscured. No pneumothorax. Pulmonary edema is mild, obscured radiographically by overlying abnormalities."
    }
  ]
}
```

