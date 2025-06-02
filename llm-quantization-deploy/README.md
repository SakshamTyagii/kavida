# LLM Quantization and Deployment

This project aims to quantize and deploy an open-source large language model (LLM) locally on a GPU machine. The primary focus is on optimizing the model for efficient inference while ensuring ease of deployment.

## Project Structure

```
llm-quantization-deploy
├── src
│   ├── main.py               # Entry point for the application
│   ├── quantization          # Module for model quantization
│   │   ├── __init__.py
│   │   └── quantize_model.py  # Functions for loading and quantizing the model
│   ├── deployment            # Module for model deployment
│   │   ├── __init__.py
│   │   └── serve_model.py     # Functions for serving the quantized model
│   └── utils                 # Utility functions
│       ├── __init__.py
│       └── system_check.py    # System requirement checks
├── scripts                   # Scripts for various tasks
│   └── system_requirements_check.py  # Executes system checks
├── requirements.txt          # Project dependencies
├── setup.py                  # Packaging configuration
└── README.md                 # Project documentation
```

## Requirements

Before running the project, ensure that your system meets the following requirements:

- **GPU**: A compatible GPU with sufficient VRAM for model inference.
- **CUDA**: Installed and accessible for GPU acceleration.
- **Python**: Version 3.9 or higher.
- **Pip and Virtualenv**: Required for managing dependencies.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd llm-quantization-deploy
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the system requirements check, execute the following script:
```
python scripts/system_requirements_check.py
```

Once the system checks are passed, you can proceed to quantize and deploy the model using the main entry point:
```
python src/main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.