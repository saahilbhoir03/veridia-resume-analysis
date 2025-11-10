import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(layout="wide", page_title="Resume Analyzer Dashboard")

st.title("Veridia — Resume Analysis Dashboard")
st.caption("Interactive dashboard for resume dataset (EDA, visualizations, simple ML).")

# ---- Sidebar: data upload / options ----
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload resumes CSV (or select sample)", type=["csv"])
use_sample = False
if uploaded is None:
    st.sidebar.info("No file uploaded — use sample or download from Kaggle and upload.")
    use_sample = st.sidebar.checkbox("Use sample (synthetic) dataset", value=False)

@st.cache_data
def load_sample():
    return pd.DataFrame({
        "Resume": [
            "Experienced data scientist with Python, pandas, sklearn, deep learning, NLP.",
            "Software engineer skilled in Java, Spring, microservices, docker, kubernetes.",
            "Business analyst with Excel, SQL, Tableau, Power BI, stakeholder management."
        ],
        "Category": ["Data Science", "Software Engineer", "Business Analyst"],
        "TotalExperience": [4, 5, 3],
        "Education": ["M.Tech", "B.Tech", "MBA"],
        "Skills": ["python, pandas, sklearn, nlp", "java, spring, docker", "excel, sql, tableau"]
    })

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = load_sample()
else:
    st.stop()

st.write(f"Loaded dataset with **{len(df)}** rows.")
if st.checkbox("Show raw data", value=False):
    st.dataframe(df.head(200))

# ---- Basic cleaning helpers ----
def normalize_text(s):
    if pd.isna(s):
        return ""
    return re.sub(r'\s+', ' ', str(s)).strip()

text_cols = [c for c in df.columns if df[c].dtype == object]
for c in text_cols:
    df[c] = df[c].apply(normalize_text)

# ---- Handle missing Skills column ----
if 'Skills' not in df.columns:
    resume_col = None
    for c in df.columns:
        if c.strip().lower() == "resume":
            resume_col = c
            break
    if resume_col:
        df['Skills'] = df[resume_col].astype(str)
    else:
        df['Skills'] = ""

# ---- Normalize skills list ----
def split_skills(s):
    if not s:
        return []
    parts = re.split(r'[;,|\n]', s)
    cleaned = []
    for p in parts:
        p = p.strip().lower()
        p = re.sub(r'[^a-z0-9\+\.\-# ]', ' ', p)
        p = p.strip()
        if p:
            cleaned.append(p)
    return cleaned

df['skill_list'] = df['Skills'].apply(split_skills)

# ---- Skill counts ----
all_skills = Counter()
for skills in df['skill_list']:
    all_skills.update(skills)
skills_df = pd.DataFrame(all_skills.most_common(), columns=['skill', 'count'])

# ---- KPIs ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Resumes", len(df))
col2.metric("Unique skills", skills_df['skill'].nunique() if not skills_df.empty else 0)
col3.metric("Avg Experience (yrs)", 
            round(df['TotalExperience'].dropna().astype(float).mean(), 2) 
            if 'TotalExperience' in df.columns else "N/A")
col4.metric("Unique Categories", df['Category'].nunique() if 'Category' in df.columns else "N/A")

# ---- Filters ----
st.markdown("### Filters")
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

if 'Category' in df.columns:
    cats = ["All"] + sorted(df['Category'].dropna().unique().tolist())
    sel_cat = filter_col1.selectbox("Category", cats)
else:
    sel_cat = "All"

if 'TotalExperience' in df.columns:
    min_exp = int(df['TotalExperience'].min())
    max_exp = int(df['TotalExperience'].max())
    sel_exp = filter_col2.slider("Experience (years)", min_exp, max_exp, (min_exp, max_exp))
else:
    sel_exp = None

skill_search = filter_col3.text_input("Filter by Skill (partial match)")

df_filtered = df.copy()
if sel_cat != "All" and 'Category' in df.columns:
    df_filtered = df_filtered[df_filtered['Category'] == sel_cat]
if sel_exp and 'TotalExperience' in df.columns:
    df_filtered = df_filtered[df_filtered['TotalExperience'].between(sel_exp[0], sel_exp[1])]
if skill_search:
    sval = skill_search.lower()
    df_filtered = df_filtered[df_filtered['Skills'].str.lower().str.contains(sval, na=False)]

st.write(f"Filtered resumes: **{len(df_filtered)}**")

# ---- Visualizations ----
st.markdown("## Visualizations")

st.subheader("Top Skills")
top_n = st.slider("Top N skills to show", 5, 50, 15)
if not skills_df.empty:
    top_skills = skills_df.head(top_n)
    fig1 = px.bar(top_skills.sort_values('count', ascending=True),
                  x='count', y='skill', orientation='h',
                  labels={'count': 'Count', 'skill': 'Skill'}, height=400)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No skill data available to plot.")

if 'Category' in df.columns:
    st.subheader("Category Distribution")
    cat_counts = df['Category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    fig2 = px.pie(cat_counts, names='Category', values='Count', title='Resumes by Category')
    st.plotly_chart(fig2, use_container_width=True)

if 'TotalExperience' in df.columns:
    st.subheader("Experience Distribution")
    fig3 = px.histogram(df, x='TotalExperience', nbins=15, title='Years of Experience')
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Skill frequency by Category (Top skills)")
if 'Category' in df.columns and not skills_df.empty:
    top_sk = skills_df.head(20)['skill'].tolist()
    rows = []
    for cat in df['Category'].unique():
        subset = df[df['Category'] == cat]
        counts = {s: 0 for s in top_sk}
        for skills in subset['skill_list']:
            for s in skills:
                if s in counts:
                    counts[s] += 1
        row = {'Category': cat}
        row.update(counts)
        rows.append(row)
    cross = pd.DataFrame(rows).set_index('Category')
    fig4 = px.imshow(cross, labels=dict(x="Skill", y="Category", color="Count"), aspect="auto")
    st.plotly_chart(fig4, use_container_width=True)

# ---- Resume Text Analysis ----
st.markdown("## Resume Text Analysis")
if st.checkbox("Show sample resume texts", value=False):
    cols_to_show = [c for c in ['Resume', 'Skills', 'Category'] if c in df_filtered.columns]
    if cols_to_show:
        st.dataframe(df_filtered[cols_to_show].head(50))
    else:
        st.info("No resume text available to display.")

st.markdown("### Common keywords (from Resume text)")
def top_keywords(series, n=30):
    words = Counter()
    for txt in series.dropna().astype(str):
        tokens = re.findall(r'\b[a-zA-Z\+\#]{2,}\b', txt.lower())
        words.update([t for t in tokens if len(t) > 1])
    return words.most_common(n)

kw = top_keywords(df_filtered.get('Resume', df_filtered.get('Skills', pd.Series())), n=25)
if kw:
    kw_df = pd.DataFrame(kw, columns=['keyword', 'count'])
    fig_kw = px.bar(kw_df.sort_values('count'), x='count', y='keyword', orientation='h', height=400)
    st.plotly_chart(fig_kw, use_container_width=True)
else:
    st.info("No keywords extracted from text.")

# ---- Quick ML ----
st.markdown("## Quick ML: Predict Category from Resume text (optional)")

if 'Category' in df.columns:
    use_ml = st.checkbox("Run simple ML classifier (TF-IDF + LogisticRegression)", value=False)
    if use_ml:
        resume_col = None
        for c in df.columns:
            if c.strip().lower() == "resume":
                resume_col = c
                break

        if resume_col is not None:
            X = df[resume_col].fillna(df.get('Skills', "")).astype(str)
        else:
            X = df.get('Skills', "").astype(str)
        y = df['Category'].astype(str)

        non_empty = X.str.strip().astype(bool)
        X = X[non_empty]
        y = y[non_empty]

        if X.empty or y.nunique() < 2:
            st.warning("Not enough valid text or category diversity to train the model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            pipeline = make_pipeline(
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                LogisticRegression(max_iter=200)
            )
            with st.spinner("Training model..."):
                pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Model accuracy (test): {acc:.3f}")
            st.text("Classification report:")
            st.text(classification_report(y_test, preds))
            sample_rows = X_test.sample(min(5, len(X_test)), random_state=2)
            preds_sample = pipeline.predict(sample_rows)
            out = pd.DataFrame({
                'ResumeText': sample_rows.values,
                'Predicted': preds_sample
            })
            st.dataframe(out)
else:
    st.info("No Category column found to build a classifier.")

# ---- Export ----
st.markdown("## Export")
if st.button("Export filtered resumes to CSV"):
    tmp = df_filtered.copy()
    tmp.to_csv("filtered_resumes_export.csv", index=False)
    st.success("Saved filtered_resumes_export.csv in working directory.")

st.markdown("### Next steps / Suggestions")
st.write("""
- Add better skill-entity extraction (NER or curated skill list).
- Connect to a database or upload pipeline (Airflow/GitHub Actions).
- Deploy using Streamlit Cloud, Heroku, or GCP.
""")
