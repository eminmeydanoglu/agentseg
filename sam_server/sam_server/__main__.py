import logging

import uvicorn

from .app import create_app
from .config import ServerConfig
from .node import SamNode


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ServerConfig.from_argv(argv)
    app = create_app(SamNode(cfg))
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
