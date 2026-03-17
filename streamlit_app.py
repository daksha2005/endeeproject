import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set page config
st.set_page_config(page_title="AI Job Resume Matcher", page_icon="🎯", layout="wide")

# Final High-Visibility CSS
st.markdown("""
    <style>
    /* 1. Global Reset - MAXIMUM CONTRAST */
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* 2. Text Elements */
    p, span, label, div, .stMarkdown {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* 3. Titles and Headers */
    h1, h2, h3 {
        color: #00D4FF !important; /* Bright Sky Blue */
        font-weight: 800 !important;
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }
    
    /* 4. Status/Empty State blocks (Fixes "Indexing jobs..." visibility) */
    .stText, div[data-testid="stText"] {
        color: #FFD700 !important; /* Bright Gold for status */
        background-color: #1A1A1A !important;
        padding: 10px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    
    /* 5. Metrics visibility */
    div[data-testid="stMetric"] {
        background-color: #111111 !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 12px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #BBBBBB !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00D4FF !important;
    }
    
    /* 6. Expanders (Results) */
    .streamlit-expanderHeader {
        background-color: #222222 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
    }
    .streamlit-expanderContent {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
    }
    
    /* 7. Input and Button */
    .stTextArea textarea {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 2px solid #00D4FF !important;
    }
    .stButton>button {
        background-color: #00D4FF !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border: none !important;
    }
    
    /* 8. Success/Alerts */
    .stAlert {
        background-color: #1A1A1A !important;
        color: #00FF00 !important;
        border: 1px solid #00FF00 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Endee Mock Implementation (from notebook) ---
class MockEndeeIndex:
    def __init__(self, name="jobs"):
        self.store = {}
        self.name  = name

    def upsert(self, items):
        for it in items:
            self.store[it["id"]] = {
                "vector": np.array(it["vector"]),
                "meta":   it.get("meta", {})
            }

    def query(self, vector, top_k=5):
        if not self.store: return []
        q = np.array(vector)
        # Calculate cosine similarity
        results = []
        for k, v in self.store.items():
            # Dot product of normalized vectors = cosine similarity
            sim = float(np.dot(q, v["vector"]))
            results.append({
                "id": k,
                "similarity": sim,
                "meta": v["meta"]
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Define a simple result object like the real Endee client might return
        class R:
            def __init__(self, res):
                self.id = res["id"]
                self.similarity = res["similarity"]
                self.meta = res["meta"]
        
        return [R(r) for r in results[:top_k]]

    def count(self):
        return len(self.store)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def get_hardcoded_jobs():
    return [
        {"id":"h001","title":"Machine Learning Engineer","company":"Google","location":"Bangalore, India","salary":"$120,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Python TensorFlow PyTorch deep learning neural networks model deployment MLOps machine learning artificial intelligence"},
        {"id":"h002","title":"Data Scientist","company":"Amazon","location":"Hyderabad, India","salary":"$110,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Python machine learning statistics pandas scikit-learn SQL data analysis prediction modelling"},
        {"id":"h003","title":"AI Research Engineer","company":"Microsoft","location":"Bangalore, India","salary":"$140,000/yr","work_type":"Full-time","experience":"Senior","skills":"deep learning NLP transformers BERT GPT research Python artificial intelligence publications"},
        {"id":"h004","title":"NLP Engineer","company":"Flipkart","location":"Bangalore, India","salary":"$95,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"natural language processing BERT text classification Python spaCy NLTK sentiment analysis AI"},
        {"id":"h005","title":"Computer Vision Engineer","company":"Ola","location":"Bangalore, India","salary":"$105,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"OpenCV image recognition object detection YOLO convolutional neural networks Python deep learning"},
        {"id":"h006","title":"Generative AI Engineer","company":"Fractal Analytics","location":"Mumbai, India","salary":"$115,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"generative AI LLMs RAG vector search Python LangChain prompt engineering GPT Claude fine-tuning"},
        {"id":"h007","title":"LLM Engineer","company":"Krutrim AI","location":"Bangalore, India","salary":"$130,000/yr","work_type":"Full-time","experience":"Senior","skills":"large language models fine-tuning RLHF Python transformers NLP LangChain artificial intelligence chatbot"},
        {"id":"h008","title":"MLOps Engineer","company":"Swiggy","location":"Bangalore, India","salary":"$105,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"MLflow Kubernetes Docker CI/CD model deployment Python cloud AWS machine learning operations pipelines"},
        {"id":"h009","title":"Full Stack Developer","company":"Razorpay","location":"Bangalore, India","salary":"$95,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"React Node.js JavaScript Python MongoDB PostgreSQL REST API AWS full stack web development"},
        {"id":"h010","title":"Backend Developer","company":"Zomato","location":"Gurgaon, India","salary":"$85,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Python Django FastAPI REST API PostgreSQL Redis microservices Docker backend engineering"},
        {"id":"h011","title":"Frontend Developer","company":"Meesho","location":"Bangalore, India","salary":"$80,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"React JavaScript TypeScript HTML CSS Redux responsive design UI UX web development"},
        {"id":"h012","title":"DevOps Engineer","company":"Freshworks","location":"Chennai, India","salary":"$90,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"AWS Docker Kubernetes CI/CD Jenkins Terraform Linux bash scripting cloud automation infrastructure"},
        {"id":"h013","title":"Data Engineer","company":"Myntra","location":"Bangalore, India","salary":"$95,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Python Spark Hadoop Kafka Airflow SQL data pipelines ETL AWS big data engineering"},
        {"id":"h014","title":"Cloud Architect","company":"TCS","location":"Mumbai, India","salary":"$120,000/yr","work_type":"Full-time","experience":"Senior","skills":"AWS Azure GCP cloud architecture serverless microservices security cost optimization infrastructure"},
        {"id":"h015","title":"Data Analyst","company":"Paytm","location":"Noida, India","salary":"$65,000/yr","work_type":"Full-time","experience":"Entry-level","skills":"SQL Excel Power BI Tableau Python data visualization business intelligence analytics reporting"},
        {"id":"h016","title":"Cybersecurity Analyst","company":"HCL","location":"Noida, India","salary":"$80,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"network security ethical hacking penetration testing SIEM firewall incident response VAPT OWASP"},
        {"id":"h017","title":"Android Developer","company":"BookMyShow","location":"Mumbai, India","salary":"$85,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Kotlin Java Android Jetpack Compose REST APIs Firebase mobile app development Play Store"},
        {"id":"h018","title":"iOS Developer","company":"MakeMyTrip","location":"Gurgaon, India","salary":"$85,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Swift iOS Xcode UIKit SwiftUI REST APIs mobile development App Store Apple"},
        {"id":"h019","title":"Blockchain Developer","company":"CoinDCX","location":"Mumbai, India","salary":"$110,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Solidity Ethereum smart contracts Web3.js blockchain DeFi cryptography decentralized finance"},
        {"id":"h020","title":"Prompt Engineer","company":"Sarvam AI","location":"Bangalore, India","salary":"$90,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"LLMs prompt engineering GPT Claude NLP Python AI applications fine-tuning chatbots language models"},
        {"id":"h021","title":"Research Scientist","company":"Samsung R&D","location":"Bangalore, India","salary":"$125,000/yr","work_type":"Full-time","experience":"Senior","skills":"deep learning computer vision research publications Python C++ algorithms artificial intelligence"},
        {"id":"h022","title":"Tech Lead","company":"Hotstar","location":"Bangalore, India","salary":"$150,000/yr","work_type":"Full-time","experience":"Senior","skills":"technical leadership system design Python Java microservices mentoring architecture engineering"},
        {"id":"h023","title":"Site Reliability Engineer","company":"Uber","location":"Hyderabad, India","salary":"$130,000/yr","work_type":"Full-time","experience":"Senior","skills":"SRE Linux Python monitoring Prometheus Grafana incident management scalability reliability"},
        {"id":"h024","title":"Product Manager","company":"Dunzo","location":"Bangalore, India","salary":"$110,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"product management roadmap agile user research analytics stakeholder management strategy OKRs"},
        {"id":"h025","title":"Flutter Developer","company":"Groww","location":"Bangalore, India","salary":"$85,000/yr","work_type":"Full-time","experience":"Mid-Senior","skills":"Flutter Dart mobile development iOS Android REST APIs Firebase UI design cross-platform"},
    ]

def main():
    st.title("🎯 AI Job Resume Matcher")
    st.markdown("""
    **Powered by Endee Vector Database**
    Match your resume with job postings based on semantic meaning, not just keywords.
    """)

    # Sidebar
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of matches", 1, 10, 5)
    
    # Load Model
    embedder = load_model()

    # Initialize Index
    index = MockEndeeIndex("linkedin_jobs")

    # Load and index data
    jobs = []
    
    # Try to load either the sample or the full CSV
    csv_file = "postings_sample.csv" if os.path.exists("postings_sample.csv") else "postings.csv"
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file).head(300)
            for i, row in df.iterrows():
                jobs.append({
                    "id": f"k{i}",
                    "title": str(row.get('title', 'Unknown')),
                    "company": str(row.get('company_name', 'Unknown')),
                    "location": str(row.get('location', 'Not specified')),
                    "salary": str(row.get('normalized_salary', 'Not disclosed')),
                    "work_type": str(row.get('formatted_work_type', 'Full-time')),
                    "experience": str(row.get('formatted_experience_level', '')),
                    "skills": str(row.get('description', ''))[:300]
                })
        except Exception as e:
            st.warning(f"Failed to load postings.csv: {e}")

    # Process vectors (only once or when jobs change)
    if 'vectors_loaded' not in st.session_state:
        status_text = st.empty()
        status_text.text("🔢 Indexing jobs...")
        vectors = []
        for job in tqdm(jobs):
            text = f"{job['title']} {job['skills']} {job.get('experience','')}"
            emb = embedder.encode(text, normalize_embeddings=True)
            vectors.append({
                "id": job["id"],
                "vector": emb.tolist(),
                "meta": {
                    "title": job["title"],
                    "company": job["company"],
                    "location": job["location"],
                    "salary": job["salary"],
                    "work_type": job["work_type"],
                    "experience": job.get("experience", ""),
                    "skills": job["skills"][:200]
                }
            })
        index.upsert(vectors)
        st.session_state.index = index
        st.session_state.vectors_loaded = True
        status_text.text(f"✅ Indexed {len(jobs)} jobs in Endee!")
    else:
        index = st.session_state.index

    # Resume Input
    st.subheader("🚀 Smart Profile Matcher")
    resume_text = st.text_area("Analyze your professional profile (Skills, Experience, or Bio)", 
                              placeholder="e.g., Python Developer with experience in Django, React, and AWS. Proficient in machine learning libraries like scikit-learn...",
                              height=200)

    if st.button("Find Best Matches"):
        if resume_text.strip() == "":
            st.error("Please enter some text to match.")
        else:
            with st.spinner("🔍 Matching your profile with job database..."):
                # Encode resume
                resume_vec = embedder.encode(resume_text, normalize_embeddings=True).tolist()
                
                # Query index
                results = index.query(vector=resume_vec, top_k=top_k)
                
                if results:
                    st.success(f"Found top {len(results)} matches!")
                    
                    # Columns for metrics
                    best_match = results[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Match Score", f"{best_match.similarity*100:.1f}%")
                    col2.metric("Top Role", best_match.meta['title'])
                    col3.metric("Top Company", best_match.meta['company'])
                    
                    # Detailed results
                    for i, r in enumerate(results):
                        with st.expander(f"#{i+1}: {r.meta['title']} at {r.meta['company']} ({r.similarity*100:.1f}%)"):
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                st.write(f"**Location:** {r.meta['location']}")
                                st.write(f"**Work Type:** {r.meta['work_type']}")
                            with c2:
                                st.write(f"**Salary:** {r.meta['salary']}")
                                st.write(f"**Experience:** {r.meta['experience']}")
                            st.write(f"**Skills/Description:** {r.meta['skills']}...")
                    
                    # Plot
                    st.markdown("---")
                    st.subheader("📊 Match Visualization")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#000000')
                    ax.set_facecolor('#111111')
                    
                    scores = [r.similarity * 100 for r in results]
                    titles = [f"{r.meta['title']} ({r.meta['company']})" for r in results]
                    
                    # Gradient-like colors
                    colors = ['#00d4aa' if s >= 70 else '#f0a500' if s >= 45 else '#ff6b6b' for s in scores]
                    
                    # Reverse for horizontal bar chart display order (top to bottom)
                    rev_titles = list(reversed(titles))
                    rev_scores = list(reversed(scores))
                    rev_colors = list(reversed(colors))
                    
                    bars = ax.barh(rev_titles, rev_scores, color=rev_colors, height=0.6)
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1f}%', 
                                va='center', color='white', fontweight='bold')

                    ax.set_xlabel('Match Score (%)', color='#94a3b8', fontweight='bold')
                    ax.tick_params(axis='both', colors='#cbd5e1')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('#334155')
                    ax.spines['left'].set_color('#334155')
                    
                    st.pyplot(fig)
                else:
                    st.error("No matches found.")

if __name__ == "__main__":
    main()
