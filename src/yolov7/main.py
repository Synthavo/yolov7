import logging
from argparse import ArgumentParser
from pathlib import Path

from flask import Flask

from yolov7.runtime.JobExecutionAgent import JobExecutionAgent


def setup_logging(logfile: Path = None):
    fstring = (
        "%(asctime)s - %(name)s[%(funcName)s:%(lineno)s] - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fstring)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(
        logging.NOTSET
    )  # use lowest possible level for root as otherwise file will not receive logs lower than INFO
    root.handlers = [console]

    if logfile is not None:
        file = logging.FileHandler(filename=logfile)
        file.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fstring)
        file.setFormatter(formatter)

        root.handlers = [console, file]

    # change log level of modules we don't care about
    logging.getLogger("urllib3").setLevel(logging.INFO)


setup_logging()
log = logging.getLogger(__name__)

from yolov7 import __version__
from yolov7.configuration import save_config, log


def configure_service(config_path: Path):
    global config
    config = save_config(config_path)

    setup_logging(logfile=config.logfile)

    log.info(f"Starting engine-controller {__version__}...\n\n\n")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-file",
        help="YAML configuration file for yolov7 module",
        required=True,
    )
    args = parser.parse_args()

    configure_service(config_path=args.config_file)
    if config.env_mode_auto:
        agent = JobExecutionAgent()
        agent.run()
    else:
        server = Flask(__name__)
        server.run(port=5003)


if __name__ == "__main__":
    main()
