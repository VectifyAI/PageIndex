import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
from .pdf import get_pdf_name

class JsonLogger:
    """
    A simple JSON-based logger that writes distinct log files for each run session.
    """
    def __init__(self, file_path: Union[str, Any]):
        """
        Initialize the logger.

        Args:
            file_path (Union[str, Any]): The source file path (usually PDF) to derive the log filename from.
        """
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data: List[Dict[str, Any]] = []

    def log(self, level: str, message: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        """
        Log a message.

        Args:
            level (str): Log level (INFO, ERROR, etc.)
            message (Union[str, Dict]): The message content.
        """
        entry: Dict[str, Any] = {}
        if isinstance(message, dict):
            entry = message
        else:
            entry = {'message': message}
        
        entry['level'] = level
        entry['timestamp'] = datetime.now().isoformat()
        entry.update(kwargs)
        
        self.log_data.append(entry)
        
        # Write entire log data to file (inefficient for large logs, but simple for now)
        with open(self._filepath(), "w", encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

    def info(self, message: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        self.log("INFO", message, **kwargs)

    def error(self, message: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        self.log("ERROR", message, **kwargs)

    def debug(self, message: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        self.log("DEBUG", message, **kwargs)

    def exception(self, message: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self) -> str:
        return os.path.join("logs", self.filename)
