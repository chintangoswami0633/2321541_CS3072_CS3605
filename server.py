import sys

import uvicorn

from backend.utils.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("server")


def _validate_models(settings) -> None:
    model_dir = settings.model_dir
    if not model_dir.exists() or not any(model_dir.glob("*.pkl")):
        logger.error(f"No model files found in {model_dir}. Cannot start.")
        sys.exit(1)


if __name__ == "__main__":
    settings = get_settings()
    _validate_models(settings)

    logger.info(f"Starting server at http://{settings.host}:{settings.port}")
    uvicorn.run(
        "backend.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
