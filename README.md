# Medical Discharge Note Generator

A LangGraph-based application that processes medical notes and automatically generates comprehensive discharge summaries.

## Overview

This application uses a graph-based workflow to process medical notes, extract relevant information, and generate structured discharge notes. It leverages OpenAI's gpt-4o-minifor natural language processing and understanding.

## Features

- Automated processing of admission and progress notes
- Information extraction and categorization
- Consistency checking across multiple documents
- Structured discharge note generation
- State management and checkpointing
- Error handling and validation

## Architecture

The application follows a graph-based workflow with the following components:

1. Document Retrieval
2. Information Extraction
3. Consistency Checking
4. State Management
5. Discharge Note Generation

See `mermaid.txt` for a detailed workflow diagram.

## Prerequisites

- Python 3.8+
- OpenAI API key
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IyadSultan/discharge_team.git
cd discharge_team
```

2. Create and activate a virtual environment using UV:
```bash
uv venv
.\\venv\\Scripts\\activate  # Windows
source venv/bin/activate    # Unix/MacOS
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`

## Usage

1. Place medical notes in the `sample_notes` directory
2. Run the application:
```bash
python discharge_note_generator.py
```

## Input Format

The application expects medical notes in text format with the following sections:
- Admission Note
- Progress Notes
- Lab Results
- Medication Lists

## Output Format

The generated discharge note includes:
1. Patient Demographics and Admission Details
2. Hospital Course
3. Significant Lab/Test Results
4. Medications on Discharge
5. Follow-up Instructions
6. Warning Signs to Watch For

## Project Structure

```
discharge_team/
├── discharge_note_generator.py  # Main application
├── requirements.txt            # Dependencies
├── .env                       # Environment variables
├── README.md                  # Documentation
├── mermaid.txt               # Workflow diagram
└── sample_notes/             # Sample medical notes
    ├── admission_note.txt
    └── progress_notes.txt
```

## Configuration

The application can be configured through:
- `.env` file for API keys
- State configuration in the main script
- Checkpointing configuration

## Error Handling

The application includes error handling for:
- File reading operations
- API responses
- JSON parsing
- State management

## Security

- API keys are stored in `.env` file (not committed to version control)
- Input validation for medical data
- Secure state management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangGraph framework
- OpenAI GPT-4
- LangChain ecosystem 