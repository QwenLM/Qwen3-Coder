# Contributing to Qwen3-Coder

Thank you for your interest in contributing to Qwen3-Coder! We welcome contributions from the community and are grateful for any time and effort you invest in helping improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Contributing Code](#contributing-code)
  - [Improving Documentation](#improving-documentation)
- [Development Process](#development-process)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Code Style](#code-style)
  - [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing to ensure a welcoming environment for all participants.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (for model training/inference)
- Git
- pip or conda for package management

### Development Setup

For a quick start:

1. **Fork the repository**
   
   Click the "Fork" button at the top of the [Qwen3-Coder repository](https://github.com/QwenLM/Qwen3-Coder) to create your own copy.

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Qwen3-Coder.git
   cd Qwen3-Coder
   ```

3. **Set up your environment**
   ```bash
   # Create conda environment (recommended)
   conda create -n qwen3-coder python=3.10
   conda activate qwen3-coder
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

**For detailed setup instructions**, including hardware requirements, model access, and advanced configuration, please refer to our comprehensive [Developer Guide](DEVELOPER_GUIDE.md).

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, PyTorch version, GPU model)
- Error messages and stack traces
- Minimal code example that reproduces the issue

### Suggesting Features

We welcome feature suggestions! Please create an issue using our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- A clear description of the feature
- Use cases and benefits
- Potential implementation approach (if you have ideas)
- Any relevant examples or references

### Contributing Code

1. **Find or create an issue**
   - Look for issues labeled `good first issue` or `help wanted`
   - If you have a new idea, create an issue first to discuss it

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, well-documented code
   - Follow our code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   python -m pytest tests/  # Run unit tests
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Clarifying existing documentation
- Adding examples and tutorials
- Translating documentation to other languages
- Improving code comments and docstrings

## Development Process

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch (if used)
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates
- `refactor/*`: Code refactoring

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(eval): add support for new programming languages

- Added support for Rust, Go, and Swift
- Updated language detection logic
- Added corresponding test cases

Closes #123
```

### Code Style

#### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Maximum line length: 120 characters
- Use descriptive variable names

Example:
```python
def process_code_snippet(
    code: str,
    language: str,
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Process a code snippet for model input.
    
    Args:
        code: The source code to process
        language: Programming language of the code
        max_length: Maximum token length
        
    Returns:
        Dictionary containing processed code and metadata
    """
    # Implementation here
    pass
```

#### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples for complex functions

### Testing

- Write unit tests for new functionality
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use meaningful test names

Example test:
```python
def test_process_code_snippet_truncation():
    """Test that long code snippets are properly truncated."""
    long_code = "x = 1\n" * 1000
    result = process_code_snippet(long_code, "python", max_length=100)
    assert len(result["tokens"]) <= 100
```

## Pull Request Process

1. **Before submitting**
   - Ensure all tests pass
   - Update documentation if needed
   - Add an entry to CHANGELOG.md (if it exists)
   - Check that your code follows our style guidelines

2. **PR Description**
   - Use our PR template
   - Reference the issue being addressed
   - Describe what changes were made and why
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process**
   - A maintainer will review your PR
   - Address any requested changes
   - Once approved, a maintainer will merge your PR

4. **After Merging**
   - Delete your feature branch
   - Pull the latest changes from upstream
   - Celebrate your contribution! 🎉

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and discussions
- **Discord**: Join our [Discord server](https://discord.gg/CV4E9rpNSD)
- **WeChat**: Follow our WeChat group for Chinese speakers

### Communication Guidelines

- Be respectful and professional
- Be patient - maintainers are often volunteers
- Provide context and be specific
- Search existing issues before creating new ones
- Help others when you can

### Recognition

We value all contributions, including:
- Code contributions
- Bug reports
- Documentation improvements
- Community support
- Feature suggestions

Contributors will be recognized in our release notes and contributor list.

## Additional Resources

- [Qwen3-Coder Documentation](https://qwen.readthedocs.io/)
- [Blog Post](https://qwenlm.github.io/blog/qwen3-coder)
- [Model Cards on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10)
- [ArXiv Paper](https://arxiv.org/abs/2505.09388)

Thank you for contributing to Qwen3-Coder! Your efforts help make this project better for everyone in the community.