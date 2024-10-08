# DocuMate

DocuMate is a powerful web application built with Streamlit that allows users to interact with PDF documents through a chat interface. Users can upload their PDF files, ask questions about the content, and receive instant answers based on the information contained within the documents.

## Features

- **PDF Upload:** Users can upload multiple PDF files for processing.
- **Chat Interface:** Users can ask questions about the content of the uploaded PDFs and receive intelligent responses.
- **Text Extraction:** The application extracts text from PDF files and organizes it for easy querying.
- **Word Frequency Analysis:** Visualize the most common words in the uploaded PDFs with interactive charts.
- **Word Cloud Generation:** Generate visually appealing word clouds to represent the most frequent terms in the PDFs.
- **Custom Styling:** The application includes a custom CSS theme for an enhanced user experience.

## Technologies Used

- **Streamlit:** A Python library for creating interactive web applications.
- **PyPDF2:** A Python library for reading and extracting text from PDF files.
- **LangChain:** A framework for building applications with language models.
- **Google Generative AI:** Utilized for generating embeddings and responses based on user queries.
- **FAISS:** A library for efficient similarity search and clustering of dense vectors.
- **Pandas:** For data manipulation and analysis.
- **Plotly:** For creating interactive visualizations.
- **WordCloud:** For generating word clouds from text data.
- **Matplotlib:** For plotting graphs and visualizations.

## Installation

To set up and run DocuMate locally, follow these steps:

1. **Clone the Repository:**

   ``bash
   git clone https://github.com/PradheebanAnandhan/DocuMate.git
   cd documate

2. **Create a Virtual Environment (Optional but Recommended):**

``bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies:**

``bash
Copy code
pip install -r requirements.txt
Set Up Environment Variables:

Create a .env file in the root directory of the project and add your Google API key:

plaintext
Copy code
GOOGLE_API_KEY=your_google_api_key
Run the Application:

bash
Copy code
streamlit run app.py
Open your browser and go to http://localhost:8501 to access the application.

Usage
Upload PDFs: Click on the "Upload your PDF Files" button in the sidebar to upload one or multiple PDF documents.
Ask Questions: Use the text input field to type your questions about the uploaded PDF documents.
View Analysis: Explore the word frequency analysis and generated word clouds to gain insights from your PDFs.
Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to submit a pull request or open an issue on the GitHub repository.
