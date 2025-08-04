import os
import shutil
import datetime
import logging
import json
import sys

class RunLogger:
    def __init__(self, config_path="config.ini", base_dir="runs"):
        self.start_time = datetime.datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Copy config file
        self.config_filename = os.path.basename(config_path)
        shutil.copy(config_path, os.path.join(self.run_dir, self.config_filename))

        # Setup logger
        self.log_path = os.path.join(self.run_dir, "log.txt")
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger()

        # Redirect stdout and stderr
        sys.stdout = self._StreamToLogger(self.logger.info)
        # sys.stderr = self._StreamToLogger(self.logger.error)

        print(f"[RunLogger] Logging started in {self.run_dir}")

    def save_summary(self, summary_dict):
        duration = (datetime.datetime.now() - self.start_time).total_seconds()
        summary_dict["start_time"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        summary_dict["duration_seconds"] = round(duration, 2)

        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_dict, f, indent=4)

        print("[RunLogger] Summary saved.")

    class _StreamToLogger:
        def __init__(self, log_func):
            self.log_func = log_func
        def write(self, message):
            if message.strip():
                self.log_func(message.strip())
        def flush(self): pass
