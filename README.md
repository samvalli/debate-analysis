# Debate Analysis

This repository present an example of debate analysis as described in the paper **"The Shape of the Public Spheres: Measuring Online Deliberation"**.
It contains all the code to extract data from Wikidebate, Kialo and CMV, code for evaluating the aspects defined and an example of usage upon a debate.  
It is not intended to represent the full framework used, but rather as an illustration of how debate data can be collected and analyzed according to the defined dimensions.

## Overview

The project provides:
- **Data collection tools** (`data_collection/`) used to scrape and preprocess debates from the platforms **Wikidebate**, **Kialo**, and **Change My View**.
- **Aspect evaluation scripts** (`aspects/` and other scripts in the repo), which operationalize deliberation metrics such as:
  - Engagement  
  - Equality  
  - Sourcing  
  - Reasoned Opinion Expression  
  - Disagreement  
  - Topic Diversity  
  - Language Readability & Complexity  
  - Sentiment Analysis

- **Example Jupyter notebook** (`notebook_example.ipynb`) showing how to load data and compute some of these measures.

## Repository Structure

debate-analysis/
│
├── data_collection/ 
├── aspects/ 
├── notebook_example.ipynb 
└── README.md 

