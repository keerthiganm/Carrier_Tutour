# ============================================================
# AI CAREER TUTOR
#
# Main Streamlit application.
#
# Responsibilities:
# - Display the application interface
# - Accept user inputs
# - Call processing modules
# - Display results
#
# Resume-processing logic is handled separately
# in resume_parser.py
# ============================================================


import streamlit as st

from resume_parser import extract_resume_text
from resume_sections import split_resume_into_sections

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career Tutor",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🎯 AI Career Tutor")

st.write(
    "Your personalized journey from current skills "
    "to your target career."
)


# ============================================================
# CAREER PROFILE
# ============================================================

st.subheader("Build Your Career Profile")

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"]
)


# ============================================================
# PROCESS UPLOADED RESUME
# ============================================================

if uploaded_resume is not None:

    st.success(
        "Resume uploaded successfully!"
    )

    # Display uploaded-file information
    st.write(
        "File name:",
        uploaded_resume.name
    )

    st.write(
        "File type:",
        uploaded_resume.type
    )

    st.write(
        "File size:",
        round(
            uploaded_resume.size / 1024,
            2
        ),
        "KB"
    )

    try:

        # Resume parsing happens inside resume_parser.py
        resume_text = extract_resume_text(
            uploaded_resume
        )

        resume_sections = split_resume_into_sections(
            resume_text
        )

        # --------------------------------------------
        # Display extracted resume content
        # --------------------------------------------

        if resume_text:

            st.subheader(
                "Extracted Resume Text"
            )

            st.text_area(
                "Resume content",
                resume_text,
                height=400
            )

            st.subheader("Detected Resume Sections")


            for section_name, section_content in resume_sections.items():

                # Only display sections that contain content
                if section_content:

                    with st.expander(
                        section_name.title()
                    ):

                        st.write(
                            section_content
                        )

        else:

            st.warning(
                "The resume was uploaded, "
                "but no readable text was found."
            )

    except Exception as error:

        st.error(
            f"Unable to process the resume: {error}"
        )