# Voice-Enabled ERP Assistant

This project is a voice-enabled ERP (Enterprise Resource Planning) assistant that helps users create money requests for their projects using voice (whisper) or text input. The assistant leverages advanced natural language processing (NLP) models to extract relevant information from the input and generate a structured money request.

## Technical Decisions and Architecture

### Architecture

The project consists of the following main components:

1. **ERPAssistant Class**: This class handles the processing of money requests through voice or text input. It uses transformer models for enhanced natural language processing.
2. **MoneyRequest Data Class**: A data class that represents a money request with three fields: `project_id`, `amount`, and `reason`.
3. **Gradio Interface**: A Gradio-based user interface that allows users to interact with the assistant via voice or text input.

### Technical Decisions

1. **Natural Language Processing**: The project uses transformer models from the `transformers` library for named entity recognition (NER), question-answering (QA), and zero-shot classification. These models are initialized in the `ERPAssistant` class.
2. **Speech Recognition**: The `speech_recognition` library is used to convert voice input into text. The `Recognizer` class from this library and `whisper` from `openai`  is used to process audio files.
3. **Data Storage**: The processed money requests are saved to a CSV file using the `pandas` library.
4. **User Interface**: The Gradio library is used to create an interactive web interface for the assistant. This interface allows users to provide voice or text input and receive responses from the assistant.

## Setup Instructions

### Prerequisites

Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository to your local machine:

    ```sh
    git clone https://github.com/Nehlr1/Voice-Enabled-ERP.git
    cd Voice-Enabled-ERP
    ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. Run the  script:

    ```sh
    python voice_enabled_erp.py
    ```

2. The Gradio interface will launch in your default web browser. You can interact with the assistant by providing voice or text input.

### Usage

- **Voice Input**: Click on the microphone icon to record your voice input.
- **Submit**: Click the "Submit" button to process your input.
- **Clear**: Click the "Clear" button to reset the input fields.

### Example Requests

- "I need to request money for project 223 to buy some tools the amount I need is 500 riyals"
- "I need to request money for project to buy some tools the amount I need is 50 riyals"