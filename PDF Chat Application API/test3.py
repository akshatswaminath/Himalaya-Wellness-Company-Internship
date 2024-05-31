import os
import requests

# Define the PDF path
pdf_path = os.path.join("D:\\HDC_PDF_API", "Akshat Swaminath Resume.pdf")
pdf_name = os.path.basename(pdf_path)


# Test processing PDFs
files = {'pdf_files': open(pdf_path, 'rb')}
response = requests.post("http://127.0.0.1:8000/process_pdf", files=files)
print(response.json())

# Test asking a question
question = "what are the projects done by akshat"
response = requests.post("http://127.0.0.1:8000/ask_question", params={ "question": question, "pdf_filename": "Akshat Swaminath Resume" })
print(response.json())
