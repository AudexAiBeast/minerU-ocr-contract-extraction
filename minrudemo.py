import streamlit as st
import subprocess
import tempfile
import os
from pathlib import Path

st.title("Contract PDF Extractor")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_dir = os.path.join(
            temp_dir,
            "output"
        )

        os.makedirs(output_dir, exist_ok=True)

        if st.button("Process PDF"):

            with st.spinner("Running MinerU..."):

                command = [
                    "mineru",
                    "-p",
                    pdf_path,
                    "-o",
                    output_dir
                ]

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )

            st.text(result.stdout)

            # find markdown file
            md_files = list(
                Path(output_dir).rglob("*.md")
            )

            if md_files:

                md_file = md_files[0]

                with open(
                    md_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    markdown_text = f.read()

                st.success("PDF processed")

                st.subheader("Markdown Output")
                st.text_area(
                    "",
                    markdown_text,
                    height=500
                )

            else:
                st.error("No markdown file found")