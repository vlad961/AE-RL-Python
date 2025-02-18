import os
from models.helpers import logger_setup
from test import test_trained_agent_quality
from datetime import datetime

cwd = os.getcwd()
defender_model_path = os.path.join(cwd, "models/trained-models/2025-02-08-22-36-WIN-attacker-5L-defender-3L-lr-0.001-best-run-so-far/defender_model.keras")
plots_path = os.path.join(cwd, "models/trained-models/2025-02-08-22-36-WIN-attacker-5L-defender-3L-lr-0.001-best-run-so-far/plots/")


if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin)
    test_trained_agent_quality(defender_model_path, plots_path)