# 🎯 AI Job Resume Matcher (Endee & Notebook)

A sophisticated semantic search engine that uses the **Endee Vector Database** to match candidate resumes with the most relevant job postings.

---

### 🔱 Fork Acknowledgement
This project is forked from the original [endee-io/endee](https://github.com/endee-io/endee) repository. 
It has been significantly enhanced with:
*   **Vector Search Logic**: Implementation of high-performance semantic retrieval.
*   **Interactive UI**: A premium Streamlit dashboard for real-time resume matching.
*   **Visual Analytics**: Matplotlib integration for better match distribution insights.

---

## 🏗️ Core Technology: Endee Vector Database

The "heart" of this project is **Endee**, a powerful vector database designed for high-performance AI retrieval tasks.

### Why Endee?
*   **Vector Intelligence**: Endee manages 384-dimensional embeddings, allowing for searches based on **context and meaning**, not just keywords.
*   **Sub-Second Retrieval**: It enables efficient proximity search, ensuring that large datasets like `postings.csv` are queried instantly.
*   **Seamless Integration**: Endee acts as the central hub for storing job description vectors and performing real-time similarity checks against input resumes.

---

## 📓 Notebook Implementation Workflow

The core logic and system design are fully implemented and documented in the primary file: **`AI_Job_Resume_Matcher.ipynb`**.

The notebook follows a precise step-by-step pipeline:

1.  **Environment Setup**: Installing the Endee client and essential NLP libraries (`sentence-transformers`, `torch`).
2.  **Hybrid Data Acquisition**: 
    *   **External Dataset**: Loading the **`postings.csv`** dataset (LinkedIn Job Postings).
    *   **In-Memory Jobs**: Includes a curated set of **hardcoded job roles** from top companies like Google, Amazon, and Microsoft to ensure a robust test environment.
3.  **Data Cleaning**: Preprocessing text fields (titles, descriptions, skills) to ensure high-quality vector representations.
4.  **Embedding Generation**: Converting job descriptions into numerical vectors using the `all-MiniLM-L6-v2` transformer model.
5.  **Endee Indexing**: Establishing a connection and upserting the hybrid job dataset into an Endee index.
6.  **Semantic Matcher Function**: Vectorizing raw user resumes and querying the Endee index for the top $K$ relevant matches.
7.  **Output & Analysis**: 
    *   **Job Matches**: Displays role, company, and similarity scores.
    *   **Data Visualization**: Generates analytical plots using `Matplotlib` to visualize match distributions.

### 📊 Notebook Outputs
The notebook generates clear visual output to confirm the accuracy of the Endee search:

*   **Ranked Job Matches**: Precise list of matched jobs with percentage scores.
    ![Results](images/results.png)
*   **Similarity Charts**: Horizontal bar charts visualizing how well the resume aligns with various job profiles.
    ![Graph](images/graph.png)

---

*   **Source Data**: `postings.csv` + Hardcoded High-Quality Roles

---

## 💻 Bonus: Interactive Dashboard (Streamlit)

While the core research and logic reside in the notebook, an interactive web application is also provided in **`streamlit_app.py`**. 

This app serves as a real-time deployment of the Endee engine, providing a premium, high-visibility UI for end-users to find their dream jobs.

### To Run the App:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
## 🚀 Live Demo
Experience the AI Job Resume Matcher in action!
Test your resume against 300+ real-time job postings:

[**👉 Launch the Streamlit App**](https://endeeproject-4mzippyaousuyxjbdhe6qw.streamlit.app/)

---


## 📂 Project Structure

```text
AI_Job_Resume_Matcher/
├── AI_Job_Resume_Matcher.ipynb  # Core Logic & Pipeline Workflow
├── requirements.txt             # Precisely defined dependencies
├── README.md                    # Technical Documentation
├── streamlit_app.py             # Interactive Web Interface 
└── images/                      # Documentation Assets (Results & Graphs)
```

---

## 📊 Dataset Source
*   **Kaggle Source**: [LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings?select=postings.csv)

---

## 👩‍💻 Author
**Dhakshayini G**  
*The Oxford College of Engineering*
