import syslog
import logging
import logging.handlers
import datetime

Qradar_address_default = "192.168.153.123"
Qradar_port_default = 514
HEADER_default = "AIDA-ML"
MESSAGE_default = "This is a testing message"

#### Send a syslog message to the syslog server
def send_syslog(Qradar_address = Qradar_address_default, Qradar_port = Qradar_port_default, HEADER = HEADER_default, MESSAGE = MESSAGE_default):
    ## Get the root logger
    logger = logging.getLogger()
    ## Create a SysLogHandler to send logs to the syslog server
    syslog_handler = logging.handlers.SysLogHandler(address = (Qradar_address, Qradar_port))
    ## Set the logging level to INFO
    logger.setLevel(logging.INFO)

    ## Define the custom header and format for the syslog message
    time = datetime.datetime.now().strftime('%b %d %H:%M:%S')
    formatter = logging.Formatter(f'{time} {HEADER} %(message)s')

    ## Add the formatter to the handler
    syslog_handler.setFormatter(formatter)

    ## Remove existing handlers to avoid duplicate logs
    for each_handlers in logger.handlers:
        logger.removeHandler(each_handlers)

    ## Add the syslog handler to the logger
    logger.addHandler(syslog_handler)

    ## Send the message with the custom header
    logger.info(MESSAGE)

#### default
if __name__ == "__main__":
    send_syslog(Qradar_address_default, Qradar_port_default, HEADER_default, MESSAGE_default)
