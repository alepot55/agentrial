# Contributing to Agentrial

Thank you for your interest in contributing to Agentrial!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/agentrial/agentrial.git
cd agentrial

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev,langgraph]"

# Run tests
pytest tests/ -v
```

## Code Style

- Python 3.11+ with type hints
- Format with `ruff format`
- Lint with `ruff check`
- Type check with `mypy agentrial`

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=agentrial --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with a clear message
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

Please include:
- Python version
- Agentrial version (`agentrial --version`)
- Minimal reproducible example
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
