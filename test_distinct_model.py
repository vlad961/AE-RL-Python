import os
from log_config import logger_setup
from test_multiple_agents import test_trained_agent_quality
from datetime import datetime

cwd = os.getcwd()
defender_model_path = os.path.join(cwd, "models/trained-models/2025-03-19-14-05-WIN-multiple-attackers-balanced-data-att-5L-def-3L-lr-0.001/defender_model.keras")
plots_path = os.path.join(cwd, "models/trained-models/2025-03-19-14-05-WIN-multiple-attackers-balanced-data-att-5L-def-3L-lr-0.001/plots/")

if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin)
    test_trained_agent_quality(defender_model_path, plots_path)