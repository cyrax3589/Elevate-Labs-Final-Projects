import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text

MODEL_PATH = os.path.join("model", "vectorizer.pkl")
REPORT_PATH = os.path.join("reports", "candidate_ranking.csv")


def build_and_save_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    os.makedirs("model", exist_ok=True)
    joblib.dump(vectorizer, MODEL_PATH)
    return vectorizer


def load_vectorizer():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def rank_resumes(job_description, resumes):
    job_clean = preprocess_text(job_description)
    resume_texts = [preprocess_text(r["text"]) for r in resumes]

    all_texts = [job_clean] + resume_texts
    vectorizer = build_and_save_vectorizer(all_texts)

    vectors = vectorizer.transform(all_texts)
    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    scores = cosine_similarity(job_vector, resume_vectors).flatten()
    scores_pct = np.round(scores * 100, 2)

    results = []
    for i, resume in enumerate(resumes):
        results.append({
            "Candidate": resume["name"],
            "Score": scores_pct[i],
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df[["Rank", "Candidate", "Score"]]

    os.makedirs("reports", exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)

    return df.to_dict(orient="records")
