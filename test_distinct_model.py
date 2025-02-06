from models.helpers import logger_setup
from test import test_trained_agent_quality
defender_model_path = "models/trained-models/2025-02-06-09-53/defender_model.keras"
plots_path = "models/trained-models/2025-02-06-09-53/plots/"
from datetime import datetime

if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin)
    test_trained_agent_quality(defender_model_path, plots_path)