# Contributing

Thank you for helping improve HRVConformer. Focused bug fixes, portability improvements, tests, and documentation corrections are welcome.

1. Open an issue describing the problem, expected behaviour, environment, and a minimal reproduction when possible.
2. Create a focused branch and avoid committing clinical data, derived participant data, credentials, large checkpoints, or experiment logs.
3. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
4. Run `python -m compileall -q .` and `python -m pytest -q` before submitting a pull request.
5. Explain whether a change affects compatibility with the manuscript configuration or reported metrics.

Scientific changes should identify the evaluation unit (window or one-hour epoch), cohort/split, selection procedure, and seeds. New performance claims should include enough information to distinguish exploratory validation results from independent-test results.

By contributing code, you agree that it is licensed under the repository's MIT License. Do not include data or materials you are not authorized to redistribute.
