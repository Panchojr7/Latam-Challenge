# Challenge Documentation: Flight Delay Prediction

## Part I: Model Operationalization

### Environment & Compatibility Adjustments
During the operationalization phase, the original dependency versions (from 2021) presented critical compatibility issues with modern Python environments (Python 3.11+). Specifically:
* **Pandas 1.3.5** failed to install due to the deprecation of `pkg_resources` in recent `setuptools`.
* **Numpy 1.22.4** lacked pre-compiled distributions for newer architectures/Python versions.

**Resolution:** I upgraded the stack to **Pandas 1.5.3** and **Numpy 1.23.5**. These versions provide the necessary stability for modern interpreters while maintaining full backward compatibility with the logic used in the original exploration notebook.

### Model Selection
While performance metrics are equivalent across both models presented, **Logistic Regression** was chosen for its ease of use and minimal dependency footprint. This pragmatic approach reduces overhead and ensures the project meets its delivery milestones without sacrificing predictive quality.

### Save and Load Model
To optimize performance, the trained model is serialized as a pickle file. By loading this file into memory at startup, the API can serve predictions instantly without the need to re-train the model on each request.

For a scalable production setup, the training process should be handled by a separate pipeline. The API would then simply mount or download the required model version from a centralized cloud storage provider, ensuring a clean separation of concerns and proper model lineage/versioning."

## Part II: API Development

During API development, I encountered a compatibility issue with **FastAPI 0.86.0**, where the `anyio` module lacked the `start_blocking_portal` attribute.

**Resolution:** I upgraded **FastAPI** to **0.103.0** and added **httpx** as a testing dependency. This resolved the conflict and ensured a seamless development and testing environment.

 The existing version of **Locust 1.6** was incompatible with the current httpx installation.

**Resolution:** I upgraded **Locust** to **2.17.0** and resolved these dependency conflicts and enabled proper execution of the load tests.

## Part III: Deployment in GCP
The application was successfully containerized using Docker and deployed to **Google Cloud Run**. The deployment process was streamlined by ensuring all dependencies were correctly specified in the `requirements.txt` file and that the Dockerfile was properly configured to expose the necessary ports and run the application.

Due to posible format issues with `# syntax=docker/dockerfile:1.2`, I removed it from the Dockerfile to ensure compatibility with the Docker version used in the deployment environment. This change did not affect the functionality of the Dockerfile and allowed for a successful build and deployment process.


## Part IV: CI/CD Pipeline
The CI/CD pipeline was implemented using **GitHub Actions**, which provides a robust and scalable platform for automating the build, test, and deployment processes. The pipeline is configured to trigger on every push to the main branch, ensuring that all changes are automatically tested and deployed without manual intervention.


## Best Practices & Development Standards
To ensure code quality, maintainability, and a professional workflow, the following standards were implemented:

**Code Quality:** Used Pylint as the primary static code analyzer and linter, complemented by the Pylance extension in VS Code for enhanced type checking and real-time error detection.

**Version Control:** Followed Conventional Commits (e.g., feat:, fix:, docs:, refactor:) to maintain a clear, traceable, and organized project history.
        
**Environment Management:** A Python virtual environment (venv) was used to isolate dependencies, ensuring a clean and reproducible development setup.

**Project Hygiene:** A comprehensive .gitignore was configured to exclude sensitive files, virtual environments, and transient artifacts such as local reports and cache files.

**Note:** While these local configuration files (like .venv or specific local reports) are fundamental to the workflow, they were intentionally excluded from the final submission to keep the repository focused on the core solution.