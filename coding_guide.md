# LGST Lab Coding Style Guide

## Table of Contents
1. [Code Documentation](#code-documentation)
2. [Code Formatting](#code-formatting)
3. [Naming Conventions](#naming-conventions)
4. [Project Structure](#project-structure)
5. [Version Control](#version-control)
6. [Best Practices](#best-practices)

## Code Documentation

### File Headers
Each source file must begin with a header containing:
```python
"""
File: filename.py
Author: Your Name <email@example.com>
Date: YYYY-MM-DD
Description: Brief (1-3 line) description of the file's purpose.
             Additional details if needed.

Version History:
    v1.0 (YYYY-MM-DD): Initial implementation
    v1.1 (YYYY-MM-DD): Added new feature X
"""
```

### Function Documentation
For each function, include:
```python
def calculate_orbital_parameters(semi_major_axis, eccentricity):
    """
    Calculate orbital parameters based on given inputs.
    
    Args:
        semi_major_axis (float): Semi-major axis in kilometers
        eccentricity (float): Orbital eccentricity (0-1)
        
    Returns:
        dict: Dictionary containing orbital period, velocity, etc.
        
    Raises:
        ValueError: If eccentricity is not in range [0,1)
    """
    # Function implementation
```

### Commenting Guidelines
- Use docstrings for all public modules, functions, classes, and methods
- Use inline comments to explain "why" not "what"
- Keep comments up-to-date with code changes
- Use section comments to group related code blocks

## Code Formatting

### General
- Use 4 spaces for indentation (Python)
- Maximum line length: 100 characters
- Use consistent spacing around operators and after commas
- Break long lines using parentheses, not backslashes

### Alignment
```python
# Good alignment
mass_earth    = 5.972e24      # kg
radius_earth  = 6.371e6       # meters
solar_flux    = 1361          # W/m²

# Aligned comments
grav_constant = 6.67430e-11   # m³ kg⁻¹ s⁻²
speed_light   = 2.998e8       # m/s
```

## Naming Conventions

### Variables and Functions
- Use `snake_case` for variables and function names
- Use descriptive names that reveal intent
- Avoid single-letter variables except in mathematical contexts

### Constants
- Use `UPPER_SNAKE_CASE` for constants
- Define constants at module level

### Classes
- Use `PascalCase` for class names
- Include type hints for better IDE support

## Project Structure

### Directory Layout
```
project_root/
├── main.py                   # Main entry point
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── docs/                     # Documentation
├── tests/                    # Unit and integration tests
├── src/                      # Source code
│   ├── __init__.py
│   ├── physics/              # Physics-related modules
│   ├── utils/                # Utility functions
│   └── visualization/        # Plotting and visualization
└── data/                     # Input/output data
    ├── raw/                 # Raw data (immutable)
    └── processed/           # Processed data
```

## Version Control

### Commit Messages
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Include a blank line between subject and body
- Reference issues and pull requests

### Branch Naming
- `feature/` for new features
- `bugfix/` for bug fixes
- `hotfix/` for critical production fixes
- `docs/` for documentation changes

## Best Practices

### Error Handling
- Use specific exceptions
- Include meaningful error messages
- Log errors appropriately

### Testing
- Write unit tests for all functions
- Use test-driven development when possible
- Maintain good test coverage (>80%)

### Performance
- Profile before optimizing
- Use built-in functions and libraries
- Avoid premature optimization

### Security
- Never hardcode credentials
- Validate all inputs
- Use parameterized queries for databases

## Language-Specific Guidelines

### Python
- Follow PEP 8 style guide
- Use type hints for better code clarity
- Use virtual environments for dependency management

### MATLAB
- Use meaningful function and variable names
- Split large scripts into smaller functions
- Use the `%{` and `%}` syntax for block comments

## Code Review
- Review your own code before submitting
- Be constructive in code reviews
- Address all review comments before merging

## Continuous Integration
- Set up automated testing
- Use linters and formatters
- Automate deployment where possible