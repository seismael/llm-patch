# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in llm-patch, **please do not open a public issue.**

Instead, report it privately:

1. **Email**: Send details to the maintainers via the contact information in the repository.
2. **GitHub Security Advisories**: Use [GitHub's private vulnerability reporting](https://github.com/seismael/llm-patch/security/advisories/new) to submit a confidential report.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, typically within 2 weeks for critical issues

### Scope

This policy covers the `llm-patch` Python library itself. Security issues in dependencies (PyTorch, transformers, PEFT, etc.) should be reported to those projects directly.

## Security Considerations

- **Model weights**: Generated LoRA adapters contain numerical weight matrices, not executable code. However, always load adapters from trusted sources.
- **safetensors format**: llm-patch uses the `safetensors` format specifically because it prevents arbitrary code execution during deserialization (unlike pickle-based formats).
- **File watching**: The `watchdog`-based file watcher only reads files from directories you explicitly configure. It does not execute file contents.
- **No network access**: llm-patch does not make network requests. All processing is local.
