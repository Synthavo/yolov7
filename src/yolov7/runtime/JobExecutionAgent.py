import logging
import time
import requests

from yolov7.configuration import get_config
from yolov7.runtime.worker import JobExecutor

log = logging.getLogger(__name__)


class JobExecutionAgent(object):

    def __init__(self):
        self.keep_running = True
        self.config = get_config()
        self.capabilities = self._get_client_capabilities()
        self.master_url = self.config.master_url
        self.executor = JobExecutor()
        return

    def run(self):
        while self.keep_running:
            log.info('--- JobExecutionAgent checking for pending jobs ---')
            success = False
            log.info(f"Check for pending jobs...")
            success = success or self._check_for_pending_job()
            if not success:
                wait_minutes = self.config.pending_jobs_check_interval
                log.info(f'No jobs available, waiting {wait_minutes} minutes')
                time.sleep(60*wait_minutes)
        return

    def _check_for_pending_job(self):
        try:
            customer = self.config.executing_for_customer
            resp = requests.get(f"{self.config.requests_scheme}://{self.master_url}/api/v1/pendingJobs",
                                json={"client": "train-agent", "capabilities": self.capabilities,
                                      "customer_mnemonic": customer, "worker": self.config.worker_name})
            if resp.status_code >= 400:
                return False
            job = resp.json()
            resp = requests.get(f"{self.config.requests_scheme}://{self.master_url}/api/v1/acceptJob",
                                json={"id": job["id"], "worker": self.config.worker_name})
            if resp.status_code >= 400:
                return False
            resp_data = resp.json()
            if "data" in resp_data:
                job["data"] = resp_data["data"]
        except Exception as err:
            print(f"Failed to fetch new job: {err}")
            return False
        self.job_id = job["id"]
        self.executor.execute(job)
        self.job_id = None
        return True

    def _get_client_capabilities(self):
        capabilities = {
            "version": float(self.config.capabilities_version),
            #"memory": float(self.config.get_config('capabilities.memory', None))
            }
        return capabilities

