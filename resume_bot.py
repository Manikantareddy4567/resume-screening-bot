import re
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Synonym normalization map
synonym_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "viz": "visualization",
    "cv": "computer vision",
    "tf": "tensorflow",
    "py": "python"
}

def normalize_text(text):
    text = text.lower()
    for short, full in synonym_map.items():
        text = text.replace(short, full)
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("⚠️ PDF Extraction Error:", e)
    return text.strip()

def clean_text(text):
    text = normalize_text(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def keyword_match_score(jd_text, resume_text):
    jd_keywords = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    matches = jd_keywords.intersection(resume_words)
    return (len(matches) / len(jd_keywords)) * 100 if jd_keywords else 0

def get_ranked_resumes(resume_dict, job_description_text):
    print("🔍 JD Preview:", job_description_text[:300])

    # Cleaned text
    cleaned_jd = clean_text(job_description_text)
    cleaned_resumes = {k: clean_text(v) for k, v in resume_dict.items()}
    corpus = [cleaned_jd] + list(cleaned_resumes.values())

    # Vectorizer settings for better overlap
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=7000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    cosine_scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    cosine_scores = [round(score * 100, 2) for score in cosine_scores]

    final_scores = []
    for i, (name, cos_score) in enumerate(zip(resume_dict.keys(), cosine_scores)):
        raw_resume = resume_dict[name]
        raw_score = keyword_match_score(job_description_text, raw_resume)


        combined = round(0.15 * cos_score + 0.85 * raw_score, 2)

        print(f"🔀 {name}: Cosine = {cos_score}%, Keyword = {raw_score:.2f}%, Final = {combined}%")
        final_scores.append((name, combined))

    ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
    return ranked