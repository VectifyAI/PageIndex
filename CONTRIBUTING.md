# Contributing to PageIndex

Thank you for your interest in contributing to PageIndex! This guide will help you get started.

## Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/PageIndex.git
   cd PageIndex
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Set up your API key** by creating a `.env` file:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```

## Running Tests

```bash
pytest tests/ -v
```

## Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b fix/issue-123-description
   ```

2. Make your changes, keeping commits focused and atomic.

3. Run the test suite to verify nothing is broken.

4. Push your branch and open a Pull Request.

## Pull Request Guidelines

- Reference the related issue (e.g., `Closes #123`).
- Describe what changed and why.
- Add tests for new functionality or bug fixes.
- Keep PRs focused — one logical change per PR.

## Reporting Issues

When opening an issue, please include:
- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (Python version, OS, relevant dependency versions)

## Code Style

- Follow existing patterns in the codebase.
- Use type hints for new function signatures where practical.
- Keep functions focused and well-named.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
