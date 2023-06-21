from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


PROJECT_DIR = Path(__file__).parent.parent.parent

data_var_name = "SYN_AREA_LABEL_DATA"

data_dir_env = os.environ.get(data_var_name)
if data_dir_env is None:
    DATA_DIR = PROJECT_DIR/ "data"
else:
    DATA_DIR = Path(data_dir_env)
    logger.info("DATA_DIR set to %s based on %s variable", DATA_DIR, data_var_name)

CREDENTIALS_DIR = PROJECT_DIR / "credentials"
