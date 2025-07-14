import re
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ✅ Improved: Extract text from all pages
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.strip()

# ✅ Clean and stem text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ✅ Optional fallback: Keyword matching
def keyword_match_score(jd_text, resume_text):
    jd_keywords = [word.lower() for word in jd_text.split() if len(word) > 2]
    resume_lower = resume_text.lower()
    match_count = sum(1 for word in jd_keywords if word in resume_lower)
    return (match_count / len(jd_keywords)) * 100

# ✅ Main resume ranking logic
def get_ranked_resumes(resume_dict, job_description_text):
    # Debugging info
    print("🔍 Job Description Sample:", job_description_text[:300])
    for name, text in resume_dict.items():
        print(f"\n📄 Resume: {name} | Length: {len(text)}")
        print("First 300 chars:\n", text[:300])
    
    # Clean all text
    cleaned_jd = clean_text(job_description_text)
    cleaned_resumes = {k: clean_text(v) for k, v in resume_dict.items()}
    
    corpus = [cleaned_jd] + list(cleaned_resumes.values())
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute cosine similarity and scale
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    similarities = [round(score * 100, 2) for score in similarities]  # scaled as percentage
    
    # Optional fallback if similarity very low
    for i, score in enumerate(similarities):
        if score < 10:
            name = list(resume_dict.keys())[i]
            fallback_score = keyword_match_score(job_description_text, resume_dict[name])
            print(f"⚠️ Low cosine match for {name}. Fallback keyword score: {fallback_score:.2f}")
            similarities[i] = round(fallback_score, 2)
    
    # Combine and sort
    ranked = sorted(zip(resume_dict.keys(), similarities), key=lambda x: x[1], reverse=True)
    return ranked
