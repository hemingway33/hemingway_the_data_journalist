"""Main entry point for the Credit Policy Agent."""

import uvicorn
import logging
from src.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    config = get_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level=config.monitoring.log_level.lower()
    ) 