
# 🤖 AI Resume Screening Bot

A simple yet powerful **AI-powered Resume Screening Bot** that ranks uploaded resumes based on their relevance to a given job description using Natural Language Processing (NLP) and TF-IDF cosine similarity.

Built with Python, Streamlit, and Scikit-learn.

---

## 🚀 Features

- Upload multiple resumes in PDF format.
- Paste a job description to match against.
- Automatically extracts and processes resume content.
- Ranks resumes by similarity to job description using TF-IDF.
- Displays match percentages and a horizontal bar graph.
- Clean, interactive UI using Streamlit.

---

## 🧠 Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/NLP:** Python, Scikit-learn, NLTK, PDFPlumber
- **Matching Algorithm:** TF-IDF Vectorization + Cosine Similarity

---

## 📁 Project Structure

```
resume_screeningbot/
├── app.py                # Streamlit frontend app
├── resume_bot.py         # Core logic for PDF parsing, cleaning, ranking
├── requirements.txt      # Python dependencies
├── README.md             # Project overview
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-screening-bot.git
cd resume-screening-bot
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---

## 📌 Usage

1. Paste the job description into the text box.
2. Upload one or more PDF resumes.
3. Click the **"🔍 Screen Resumes"** button.
4. See the ranked results and visual graph of match scores.

---

## 🧼 Preprocessing Steps

Each resume and the job description are:

- Cleaned with regex (non-alphabetic characters removed)
- Converted to lowercase
- Tokenized and stemmed using NLTK
- Vectorized using `TfidfVectorizer`
- Scored using `cosine_similarity`

---

## 📦 Dependencies

Add this to your `requirements.txt`:

```txt
streamlit
nltk
pdfplumber
scikit-learn
matplotlib
```

Also ensure you download NLTK stopwords at runtime:

```python
import nltk
nltk.download('stopwords')
```

---

## 📌 Future Improvements

- Add resume parsing (e.g., skills, education, experience).
- Export ranked results as CSV.
- Integrate GPT-based JD matching.
- Deploy on Streamlit Cloud or AWS.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)

---

## 👨‍💻 Author

Built with ❤️ by [Manikanta Reddy](https://github.com/Manikantareddy4567)
