# ============================================================
# RESUME SECTION PARSER
#
# Responsibility:
# Convert raw resume text into structured resume sections.
#
# Example:
#
# Raw text
#    ↓
# {
#     "summary": "...",
#     "skills": "...",
#     "experience": "...",
#     "projects": "...",
#     "education": "...",
#     "certifications": "..."
# }
#
# ============================================================


# ------------------------------------------------------------
# Standard sections understood by our Career Tutor
# ------------------------------------------------------------

SECTION_ALIASES = {

    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary"
    ],

    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "core competencies",
        "technologies",
        "technical competencies"
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history"
    ],

    "projects": [
        "projects",
        "project experience",
        "academic projects",
        "personal projects"
    ],

    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "qualifications"
    ],

    "certifications": [
        "certifications",
        "certificates",
        "professional certifications",
        "courses and certifications"
    ]
}


def identify_section(line):
    """
    Check whether a line represents a known resume heading.

    Example:

        "TECHNICAL SKILLS"
            ↓
        "skills"

    If the line is not a known section heading,
    return None.
    """

    # Standardize the text before comparing it
    cleaned_line = (
        line.strip()
        .lower()
        .replace(":", "")
    )

    # Compare the line with all known aliases
    for section_name, aliases in SECTION_ALIASES.items():

        if cleaned_line in aliases:

            return section_name

    return None


def split_resume_into_sections(resume_text):
    """
    Split extracted resume text into logical sections.

    Parameters:
        resume_text:
            Plain text produced by resume_parser.py

    Returns:
        dict:
            Resume content grouped into sections.
    """

    # Store detected resume sections
    sections = {

        "header": [],
        "summary": [],
        "skills": [],
        "experience": [],
        "projects": [],
        "education": [],
        "certifications": [],
        "other": []
    }

    # Before a section heading appears,
    # content is treated as resume header information.
    #
    # Example:
    # Aarav Mehta
    # Software Engineer
    # Chennai, India
    current_section = "header"

    # Process resume line by line
    for line in resume_text.splitlines():

        cleaned_line = line.strip()

        # Ignore empty lines
        if not cleaned_line:

            continue

        # Check whether this line is a section heading
        detected_section = identify_section(
            cleaned_line
        )

        if detected_section:

            # From this point onwards,
            # following lines belong to this section
            current_section = detected_section

            continue

        # Store normal content under the active section
        sections[current_section].append(
            cleaned_line
        )

    # Convert lists of lines back to readable text
    structured_sections = {}

    for section_name, content_lines in sections.items():

        structured_sections[section_name] = (
            "\n".join(content_lines)
        )

    return structured_sections