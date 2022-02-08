import logging

from .dist import is_master

logger_initialized = set([])


def get_logger(name, log_file: str = None, log_level: int = logging.INFO, file_mode: str = 'w') -> logging.Logger:
    """Return a logger with the specified name, creating it if necessary.

    Notes:
        If current process is master process, return `Logger(log_level)` with FileHandler.
        If current process is not master process, return `Logger(logging.ERROR)`

    Args:
        name (str): specified name of logger
        log_file (str): logger file name
        log_level (int): logger level
        file_mode (str): logger file mode

    Returns:
        logger (logging.Logger)
    """

    logger = logging.getLogger(name)
    logger.propagate = False

    if name in logger_initialized:
        return logger

    logger_handlers = [logging.StreamHandler()]

    if is_master() and log_file is not None:
        logger_handlers.append(logging.FileHandler(log_file, file_mode))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    for handler in logger_handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_master():
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized.add(name)

    return logger
