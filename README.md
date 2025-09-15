# Debate Analysis

This repository present an example of debate analysis as described in the paper **"The Shape of the Public Spheres: Measuring Online Deliberation"**.
It contains all the code to extract data from Wikidebate, Kialo and CMV, code for evaluating the aspects defined and an example of usage upon a debate.  
It is not intended to represent the full framework used, but rather as an illustration of how debate data can be collected and analyzed according to the defined dimensions.

## Overview

The project provides:
- **Data collection tools** (`code/data_collection/`) used to scrape and preprocess debates from the platforms **Wikidebate**, **Kialo**, and **Change My View**. In this folder all the we find a python script for each of the platforms (`wikidebate_utils.py`,`kialo_utils.py` and `cmv_utils.py`), along with a script to manage wikidebate revisions (`revisions_utils.py`)

- **Aspect evaluation scripts** (`code/` and other scripts in the repo), which operationalize deliberation metrics such as:
  - Engagement  
  - Equality  
  - Sourcing  
  - Reasoned Opinion Expression  
  - Disagreement  
  - Topic Diversity  
  - Language Readability & Complexity  
  - Sentiment Analysis

- **Data** (`data/`) contains the data used for the example, rest of the data used in the paper is avaible upon reasonable request.

- **Example of debate analysis** (`notebook_example.ipynb`) shows how to compute the defined metrics upon three debates -one from each platform considered- about the same topic.

## Data collection

At the time of writing, this repository does not include a full tutorial on how to collect the required data.
However, we provide code to extract data from the three platforms (avaiable in `src/data_collection`), along with brief notes on the process:

* **Wikidebate**: We collected data from Wikidebates hosted on Wikiversity using the official MediaWiki API. For each debate, we downloaded the full revision history, including metadata such as revision ID, timestamp, user, size, content, and edit flags (minor/anonymous). 
The debate text was parsed with `mwparserfromhell` to extract arguments, comments, objections, and references, while additional routines ensured proper handling of special cases. We then linked references to the corresponding argumentative items and reconstructed the debate structure across revisions. The resulting code provides, for each debate item, its text, argumentative role, reference count, and upon requsurvival/modification history over time.

* **Kialo**: The most reliable way we found to download debates of interest was manually. After collecting the links to the debates, you can retrieve them directly from the Kialo website. Each debate can be downloaded as a `.txt` file containing all the posts and references (e.g. `data/raw_data_kialo/kialo_page_content/1096/should-governments-provide-a-universal-basic-income-14053.txt`). Additionally, a `.txt` file containing information about the users activity (e.g. `data/raw_data_kialo/kialo_authors/1096/should-governments-provide-a-universal-basic-income-14053.txt`). 

* **CMV**: As with Kialo, we first gathered the links to the debates of interest. To obtain all posts from these debates, we used the `praw` library, all the function to use it are contained in the `cmv_utils.py` script. within this script you can find the `preprocess_reddit_text` function. This function cleans Reddit posts by removing formatting artifacts (underscores, quotes, extra whitespace), links, Reddit-specific mentions, and special characters. It also normalizes text by decoding HTML entities and converting emojis into their textual representation.

For further information about data collection contact the authors.

## Debate analysis example

You can find a small but instructive example in `notebook_example.ipynb`. This notebook show how the provided code can be used for analyzing the aspects of debates across Wikidebates, Kialo, and ChangeMyView (CMV). We selected a debate about the topic Universal Basic Income from each these platform and evaluated the defined metrics upon them. 

## Repository Structure

debate-analysis/
│
├── src/      
|     └── data_collection/
├── data/
|      └── knowledge_dimension/
|      └── raw_data_kialo/
|      └── st_debates/
|      └── st_embeddings/
├── notebook_example.ipynb
└── README.md
```

