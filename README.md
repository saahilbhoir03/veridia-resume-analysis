# veridia-resume-analysis
Interactive Streamlit dashboard for resume analytics and ML-based categorization — Internship project at Veridia.


# Veridia Internship Tasks

## Overview
This repository contains tasks completed during my internship at Veridia. The current focus of the tasks is cleaning and preparing a dataset of resumes for further analysis or processing.

## Project Structure
Internship/
├── clean_dataset.py # Script to clean the Resume.csv dataset
├── Resume.csv # Original dataset
├── resume_clean.csv # Cleaned dataset output
└── README.md # Project documentation


---

## Task Description
**Objective:**  
Clean and preprocess the resume dataset to remove inconsistencies, missing values, and duplicates.

**Steps Performed:**
1. Loaded `Resume.csv` containing 2484 rows and 4 columns.
2. Cleaned the data:
   - Removed duplicate entries.
   - Handled missing or malformed values.
3. Saved the cleaned dataset as `resume_clean.csv` (2483 rows).

---

## Dataset Summary

|    Attribute   | Original Dataset | Cleaned Dataset |
|----------------|------------------|-----------------|
| Rows           |      2484        |      2483       |
| Columns        |       4          |       4         |
| Missing Values |     Some         |      None       |
| Duplicates     |      Yes         |    Removed      |

---

## How to Run
1. Clone the repository:
   git clone <your-repo-link>
   cd Internship
2. Ensure you have Python installed (preferably 3.10+).
3. Install required packages:
   pip install pandas
4. Run the cleaning script:
   python clean_dataset.py
5. After execution, resume_clean.csv will be generated in the same folder.
   Notes
   - resume_clean.csv is ready for further analysis, modeling, or visualization.
   - Ensure Resume.csv is present in the working directory before running the script.

Author

Sahil Bhoir
Intern, Veridia  
