# Challenge Documentation: Flight Delay Prediction

## Part I: Model Operationalization

### Environment & Compatibility Adjustments
During the operationalization phase, the original dependency versions (from 2021) presented critical compatibility issues with modern Python environments (Python 3.11+). Specifically:
* **Pandas 1.3.5** failed to install due to the deprecation of `pkg_resources` in recent `setuptools`.
* **Numpy 1.22.4** lacked pre-compiled distributions for newer architectures/Python versions.

**Resolution:** I upgraded the stack to **Pandas 1.5.3** and **Numpy 1.23.5**. These versions provide the necessary stability for modern interpreters while maintaining full backward compatibility with the logic used in the original exploration notebook.