"""TqdmLoggingHandler — logs via tqdm.write so progress bars aren't corrupted."""
import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
