import os


def get_candidate_name(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ").title()
    return name


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"
