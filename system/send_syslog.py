import logging
import logging.handlers
import datetime
from typing import Optional

# Local defaults; prefer config via caller
Qradar_address_default = "192.168.153.123"
Qradar_port_default = 514
HEADER_default = "AIR-RF"
MESSAGE_default = "This is a testing message"

_LOGGER_NAME = "air.syslog"
_handler_cache = {}

def _get_logger(address: str, port: int, header: str) -> logging.Logger:
    """Create or reuse a dedicated logger with a SysLogHandler.
    We avoid touching the root logger to not interfere with application logging.
    """
    key = (address, port, header)
    logger = logging.getLogger(f"{_LOGGER_NAME}.{address}:{port}.{header}")
    logger.setLevel(logging.INFO)

    if key not in _handler_cache:
        handler = logging.handlers.SysLogHandler(address=(address, port))
        # Use dynamic timestamp per message
        formatter = logging.Formatter(f'%(asctime)s {header} %(message)s', datefmt='%b %d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _handler_cache[key] = handler
    else:
        # Ensure the handler is attached (in case logger was recreated)
        handler = _handler_cache[key]
        if handler not in logger.handlers:
            logger.addHandler(handler)

    return logger

#### Send a syslog message to the syslog server
def send_syslog(Qradar_address: str = Qradar_address_default,
                Qradar_port: int = Qradar_port_default,
                HEADER: str = HEADER_default,
                MESSAGE: Optional[str] = MESSAGE_default) -> None:
    if MESSAGE is None:
        return
    # Ensure single-line message
    msg = str(MESSAGE).replace('\n', ' ').replace('\r', ' ')
    logger = _get_logger(Qradar_address, Qradar_port, HEADER)
    try:
        logger.info(msg)
    except Exception:
        # Best-effort: drop errors silently to avoid crashing callers
        pass

#### default
if __name__ == "__main__":
    send_syslog(Qradar_address_default, Qradar_port_default, HEADER_default, MESSAGE_default)
