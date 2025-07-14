import streamlit as st
import matplotlib.pyplot as plt
from resume_bot import extract_text_from_pdf, get_ranked_resumes

st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("📄 Resume Screening Bot with AI")
st.markdown("Upload resumes and a job description to get matching scores and visual results.")

job_description = st.text_area("Paste Job Description Here", height=200)
uploaded_resumes = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if st.button("🔍 Screen Resumes"):
    if not uploaded_resumes:
        st.warning("Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste the job description.")
    else:
        resumes = {}
        for file in uploaded_resumes:
            resumes[file.name] = extract_text_from_pdf(file)

        ranked_results = get_ranked_resumes(resumes, job_description)

        st.subheader("📊 Ranked Results")
        for name, score in ranked_results:
            st.write(f"✅ **{name}** → `{score}% match`")

        st.subheader("📈 Visual Match Graph")
        names = [r[0] for r in ranked_results]
        scores = [r[1] for r in ranked_results]

        fig, ax = plt.subplots()
        ax.barh(names[::-1], scores[::-1], color='skyblue')
        ax.set_xlabel("Match %")
        ax.set_title("Resume Match Scores")
        st.pyplot(fig)
