import os
from flask import Flask, render_template, request, redirect, url_for, send_file, session

from pdf_utils import extract_text_from_pdf
from ranking import rank_resumes, REPORT_PATH
from utils import get_candidate_name, allowed_file

app = Flask(__name__)
app.secret_key = "resume_ranker_secret_key"

DEFAULT_JD = """Software Engineer with expertise in Python, Machine Learning, SQL, Flask, REST APIs,
Data Structures, Algorithms, NumPy, Pandas, and cloud deployment. Experience with scikit-learn,
deep learning frameworks, and building scalable backend services."""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", default_jd=DEFAULT_JD)


@app.route("/rank", methods=["POST"])
def rank():
    job_description = request.form.get("job_description", "").strip()
    files = request.files.getlist("resumes")

    if not job_description:
        return render_template("index.html", error="Please enter a job description.", default_jd=DEFAULT_JD)

    if not files or all(f.filename == "" for f in files):
        return render_template("index.html", error="Please upload at least one PDF resume.", default_jd=DEFAULT_JD)

    resumes = []
    for f in files:
        if f and allowed_file(f.filename):
            text = extract_text_from_pdf(f)
            if text:
                resumes.append({
                    "name": get_candidate_name(f.filename),
                    "text": text
                })

    if not resumes:
        return render_template("index.html", error="Could not extract text from uploaded PDFs.", default_jd=DEFAULT_JD)

    results = rank_resumes(job_description, resumes)
    return render_template("results.html", results=results, total=len(results))


@app.route("/download")
def download():
    if os.path.exists(REPORT_PATH):
        return send_file(REPORT_PATH, as_attachment=True, download_name="candidate_ranking.csv")
    return redirect(url_for("index"))


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    app.run(debug=True)
