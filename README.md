# Human-in-the-loop ReAct Agent

## Overview

This project implements a Human-in-the-loop ReAct (Reasoning and Acting) Agent, designed to tackle various tasks through a combination of AI reasoning and human feedback. The agent uses the PaLM API for natural language processing and a Gradio interface for human interaction.

## Features

- Task-based reasoning and action proposal
- Human feedback integration (approve, modify, or reject actions)
- Dynamic task execution
- Final task generation and evaluation
- User-friendly web interface

## Prerequisites

- Python 3.7+
- PaLM API key
- Internet connection

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/human-in-the-loop-react-agent.git
   cd human-in-the-loop-react-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your PaLM API key:
   - Create a `.env` file in the project root
   - Add your API key: `PALM_API_KEY=your_api_key_here`

## Usage

1. Start the Gradio interface:
   ```
   python app.py
   ```

2. Open your web browser and navigate to the URL provided by Gradio (usually `http://localhost:7860`).

3. Use the interface to:
   - Start a new task
   - View the agent's thoughts and proposed actions
   - Approve, modify, or reject actions
   - See the final task evaluation and recommendation

## Project Structure

- `app.py`: Main application file with Gradio interface
- `react_agent.py`: ReAct Agent class implementation
- `task_dataset.json`: JSON file containing task definitions
- `requirements.txt`: List of Python package dependencies

## Customization

To add or modify tasks, edit the `task_dataset.json` file. Each task should include:
- `description`: A brief description of the task
- `possible_actions`: A list of potential actions for the task
- `expected_outcome`: The desired result of the task

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- PaLM API for natural language processing capabilities
- Gradio for the interactive web interface

## Contact

For any queries or suggestions, please open an issue in the GitHub repository.
