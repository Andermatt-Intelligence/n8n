from enum import Enum
from dataclasses import dataclass
from multiprocessing.context import ForkServerProcess
import subprocess


class TaskStatus(Enum):
    WAITING_FOR_SETTINGS = "waiting_for_settings"
    RUNNING = "running"
    ABORTING = "aborting"


# Process type can be either a multiprocessing ForkServerProcess (for exec() mode)
# or a subprocess.Popen (for UV mode)
ProcessType = ForkServerProcess | subprocess.Popen | None


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus
    process: ProcessType = None
    uv_script_path: str | None = None  # Path to temp script for UV cleanup
    workflow_name: str | None = None
    workflow_id: str | None = None
    node_name: str | None = None
    node_id: str | None = None

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = TaskStatus.WAITING_FOR_SETTINGS
        self.process = None
        self.uv_script_path = None
        self.workflow_name = None
        self.workflow_id = None
        self.node_name = None
        self.node_id = None

    def context(self):
        return {
            "node_name": self.node_name,
            "node_id": self.node_id,
            "workflow_name": self.workflow_name,
            "workflow_id": self.workflow_id,
        }
