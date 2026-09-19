# ORBIT-X

**Causal Temporal World Model for Enterprise Systems**

### What is this project?
ORBIT‑X is a research‑grade prototype that learns a *temporal‑graph world model* of a synthetic software enterprise. It can predict the downstream consequences of interventions (e.g., service failures, traffic spikes) that were **not observed during training**. The system integrates a deterministic simulator, data pipelines, a suite of baseline and advanced ML models, uncertainty estimation, and a decision‑optimization layer.

### Final Result
When fully built, the repository will provide:
- A configurable synthetic enterprise simulator that generates telemetry, failure events, and intervention logs.
- End‑to‑end data generation pipelines producing reproducible datasets (IID, OOD, compositional, etc.).
- Baseline and state‑of‑the‑art temporal, graph, and temporal‑graph models, all trained and tracked with MLflow.
- A counterfactual engine that accepts natural‑language queries (via the LLM agent) and returns machine‑readable predictions, uncertainty intervals, and business‑impact estimates.
- A FastAPI backend exposing health, simulation, prediction, and optimisation endpoints.
- An interactive React/TypeScript frontend visualising the enterprise graph, telemetry, and counterfactual outcomes.
- Comprehensive research documentation, experiment reports, model cards, and a benchmark suite.

### Technical Stack & Rationale
| Layer | Technologies (pinned) | Why we chose them |
|-------|-----------------------|-------------------|
| Core language | Python 3.11+ | Mature ecosystem for scientific computing, type‑checking, and ML libraries. |
| ML & Graph | PyTorch 2.3, PyTorch‑Geometric 2.5, NumPy 2.0, Pandas 2.2 | Fast GPU/CPU tensor ops, native support for GNNs, strong community. |
| Classical ML | scikit‑learn, XGBoost | Strong baselines for fairness comparisons. |
| Data tracking | MLflow 2.12 | Automatic logging of parameters, metrics, artifacts, and model versions. |
| API | FastAPI 0.110, Pydantic 2.7 | High‑performance, typed request/response schemas, auto‑generated OpenAPI docs. |
| Frontend | React 18, TypeScript, Three.js, Chart.js | Modern, type‑safe UI framework with 3D graph visualisation capabilities. |
| Containerisation | Docker & docker‑compose | Guarantees reproducible environment across machines. |
| CI/CD | GitHub Actions (ruff, mypy, pytest) | Enforces code quality, type safety, and test coverage on every push. |

### How it works (high‑level pipeline)
1. **Simulator** generates a deterministic enterprise topology, workloads, and failure events based on a seed‑driven configuration.
2. **Data pipeline** converts raw simulation logs into structured telemetry, intervention logs, and ground‑truth outcomes (stored in PostgreSQL or parquet files).
3. **Model training** – baseline, temporal, graph, and temporal‑graph models are trained on the generated datasets; metrics and checkpoints are logged to MLflow.
4. **Counterfactual engine** receives a structured intervention (e.g., `{"target":"payment_db","type":"shutdown","duration":900}`), encodes the current world state, conditions the model on the intervention, rolls the latent state forward, and decodes predictions.
5. **Uncertainty layer** (ensemble / MC‑dropout / conformal) provides prediction intervals and calibration diagnostics.
6. **Business impact model** translates system‑level predictions into revenue, cost, and risk estimates.
7. **Decision optimisation** evaluates alternative actions against a weighted objective (loss + operational cost + risk).
8. **LLM agent** parses natural‑language queries, validates parameters, invokes the engine, and returns a human‑readable explanation.
9. **Frontend** visualises the graph, telemetry dashboards, and counterfactual results in real time.

### Local Setup (PC / Development machine)
**Prerequisites**
- Windows 10/11 (PowerShell) or WSL2 with Bash.
- Python 3.11+ (https://www.python.org/downloads/).
- Node 20+ and npm (for the frontend).
- Docker Desktop (optional but recommended for full stack).

**Step‑by‑step**
```bash
# 1. Clone the repository
git clone <repo‑url>
cd orbit-x

# 2. Create a virtual environment and activate it
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell
# or: source .venv/bin/activate   # WSL/Linux

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install development extras (lint, type‑check, test)
pip install -e .[dev]

# 5. Set up environment variables
cp .env.example .env   # edit .env if you need custom DB credentials etc.

# 6. Initialise the database (PostgreSQL) – see docs/database/setup.md for Docker or local install instructions.
# Example with Docker Compose (if Docker is installed):
docker-compose up -d postgres

# 7. Run the initial data generation (demo config) – creates a small deterministic enterprise.
python -m orbit_x.demo   # populates the DB and stores a checkpoint

# 8. Start the backend API
uvicorn orbit_x.api.main:app --reload

# 9. In another terminal, start the frontend
cd frontend
npm install
npm start   # launches http://localhost:3000

# 10. Verify everything works
open http://localhost:3000   # UI shows the demo enterprise and allows you to ask
# Example natural‑language query (via the UI):
# "What happens if payment‑db goes offline for 15 minutes?"
```
**Running tests & benchmark**
```bash
make test          # unit, integration, and end‑to‑end tests
make benchmark     # trains all models on the generated datasets and produces JSON/CSV reports
```
All experiment artefacts are stored under `artifacts/` and logged in MLflow (`mlflow ui` to explore).

---

For detailed architecture diagrams, research methodology, and full API specification, see the `docs/` folder.

