import base64
import os.path
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import traceback
import pandas as pd
import json

import requests
from flask import logging

import train
import detect
from yolov7 import helper as model_eval

from yolov7.configuration import get_config
from yolov7.entities.constant import JOB_CODE_BUILD_MODEL_HIERARCHY, JOB_CODE_TRAIN_YOLO, JOB_CODE_EVALUATE_MODEL


class AbstractRunner(object):

    def __init__(self, config=None):
        self.logger = logging.logging.getLogger('file')
        self.capabilities = []
        self.device = '0'
        if config is None:
            self.configuration = get_config()
        else:
            self.configuration = config
        self._load_configuration()
        self.keep_running = True
        self.data = None
        self.job = None
        return

    def execute(self, job):
        try:
            self.data = job["data"]
            self.job = job
            self.pre_execution()
            self.execution()
            self.get_follow_up_job()
            self.post_execution()
            self.update_job_state(job['id'], 'finished')
        except Exception as ex:
            self.logger.exception(ex)
            traceback_str = traceback.format_exc()
            self.update_job_state(job['id'], 'failed', traceback_str=traceback_str)
        self.cleanup()
        return

    def update_job_state(self, job_id, new_state, traceback_str=None, retry_job=True):
        try:
            response = requests.post(
                f"{self.configuration.requests_scheme}://{self.configuration.master_url}/api/v1/updateJobStatus",
                json={"id": job_id, "state": new_state,
                      'traceback_str': traceback_str, "retry_job": retry_job})
        except RuntimeError as err:
            self.logger.exception(f"Failed to abort job")
            self.logger.error(f"Failed to execute request!: {err}")
            return 404
        return response.status_code

    def _download_dependency(self, dep):
        pass

    def _load_configuration(self):
        return

    def pre_execution(self):
        if 'dependencies' in self.job:
            dependecies = self.job['dependencies']
            for dep in dependecies:
                self._download_dependency(dep)

        return

    def execution(self):
        raise NotImplementedError

    def get_follow_up_job(self):
        return

    def post_execution(self):
        return

    def cleanup(self):
        return


class TrainYoloRunner(AbstractRunner):
    default_epochs = 150
    default_batch = 8
    default_device = 'cpu'  # 0 # - GPU
    default_model = "yolov5m"

    def __init__(self, config=None):
        super().__init__(config)
        return

    def _check_gpu_capability(self):

        return False

    def pre_execution(self):
        if "device" not in self.data:
            self.data["device"] = self.default_device
        if "batch" not in self.data:
            self.data["batch"] = self.default_batch
        if "epochs" not in self.data:
            self.data["epochs"] = self.default_epochs

    def execution(self):
        success = self._create_training_config(self.data)
        if success:
            success = self._download_training_data("http://{}/api/v1/download?type=dataset&id=" + str(self.job['id']))
        if success:
            model_type = self.default_model
            if "model_type" in self.data:
                model_type = self.data["model_type"]
            success = self._prepare_training_weights(model_type)
        if success:
            success = self._run_training(self.data)
        return success

    def post_execution(self):
        self._publish_results()

    def cleanup(self):
        self._cleanup_env()

    def _create_training_config(self, config):
        if not os.path.exists(self.configuration.training_config_path):
            os.makedirs(self.configuration.training_config_path)
        with open(f"{self.configuration.training_config_path}/dataset.yaml",
                  "w") as dataset_file:
            config['classes'].sort(key=lambda x: x['class_id'])
            dataset_file.writelines([
                f"train: {self.configuration.data_path}/train\n",
                f"val: {self.configuration.data_path}/val\n",
                f"nc: {len(config['classes'])}\n",
                f"names: {[c['name'] for c in config['classes']]}\n"
            ])
        with open(f"{self.configuration.training_config_path}/hyp.scratch.yaml",
                  "r") as hyp_file:
            content = hyp_file.readlines()
        hyper_param_dict = {}
        for line in content:
            if line.startswith("#") or len(line) == 0:
                continue
            if ":" in line:
                key_set = line.split(":")
                key = key_set[0].strip()
                val = key_set[1].split("#")[0].strip()
                hyper_param_dict[key] = val
        if "hyperparams" in config:
            for k in config["hyperparams"]:
                hyper_param_dict[k] = config["hyperparams"]
        hyperparameter_lines = []
        for k in hyper_param_dict:
            hyperparameter_lines.append(f"{k}: {hyper_param_dict[k]}\n")
        with open(f"{self.configuration.training_config_path}/hyp.scratch.yaml",
                  "w") as hyp_file_write:
            hyp_file_write.writelines(hyperparameter_lines)
        return True

    def _download_training_data(self, data_url):
        sub_paths = ["train/images", "train/labels", "val/images", "val/labels"]
        for sub_path in sub_paths:
            if not os.path.exists(f"{self.configuration.data_path}/{sub_path}"):
                os.makedirs(f"{self.configuration.data_path}/{sub_path}")
        resp = urlopen(data_url.format(self.configuration.master_url))
        zipfile = ZipFile(BytesIO(resp.read()))
        for line in zipfile.namelist():
            if os.path.isdir(f"{self.configuration.data_path}/{line}"):
                continue
            content = zipfile.open(line).readlines()
            with open(f"{self.configuration.data_path}/{line}", "wb") as datafile:
                for c in content:
                    datafile.write(c)
        return True

    def _prepare_training_weights(self, selected_model):
        try:
            if not os.path.exists(self.configuration.training_weight_path):
                os.makedirs(self.configuration.training_weight_path)
            shutil.copy(f"{self.configuration.model_weights_path}/{selected_model}.pt",
                        f"{self.configuration.training_weight_path}/best.pt")
        except Exception as err:
            self.logger.error(f"{err}")
            return False
        return True

    def _publish_epoch_update(self, log_vals, epoch, best_fitness, fi):
        try:
            print(f"Epoch update: log_vals: {log_vals}\n epoch: {epoch}\n best_fitness: {best_fitness}\n fi: {fi}")
            resp = requests.post(f"http://{self.configuration.master_url}/api/v1/updateJobStatus",
                                 json={"status": epoch / int(self.data['epochs']), "id": self.job_id})
            if resp.status_code < 400:
                return True
        except Exception as err:
            self.logger.error(f"Failed to update job status: {err}")
        return False

    def _publish_training_finish_update(self, last, best, plots, epoch, results):
        try:
            print(f"Epoch update: last: {last}\n best: {best}\n plots: {plots}\n epoch: {epoch}\n results: {results}")
            resp = requests.post(f"http://{self.configuration.master_url}/api/v1/updateJobStatus",
                                 json={"status": "finished", "id": self.job_id})
            if resp.status_code < 400:
                return True
        except Exception as err:
            self.logger.error(f"Failed to update job status: {err}")
        return False

    def _run_training(self, params):
        callbacks = []  # Callbacks()
        # callbacks.register_action("on_fit_epoch_end", "epoch_finish_status_update", self._publish_epoch_update)
        # callbacks.register_action("on_train_end", "training_finish_status_update", self._publish_training_finish_update)
        train.run(device=params["device"], img=512, batch=int(params["batch"]), epochs=int(params["epochs"]),
                  data=f"{self.configuration.training_config_path}/dataset.yaml",
                  hyp=f"{self.configuration.training_config_path}/hyp.scratch.yaml",
                  weights=f"{self.configuration.training_weight_path}/best.pt",
                  custom_callbacks=callbacks)
        # TODO:
        # export.run(weights="/app/runs/train/exp/weights/best.pt", include='tfjs')
        print("------------finish training---------")
        return True

    def _publish_results(self):
        with open(f"/app/runs/train/exp/weights/best.pt", "rb") as weights_file:
            data = weights_file.read()
            data_str = base64.b64encode(data).decode("utf-8")
        request_data = {
            "data_raw": data_str
        }
        resp = requests.post(f"http://{self.configuration.master_url}/api/v1/upload?obj=model&id={self.data['model_id']}",
                             json=request_data)
        if resp.status_code < 400:
            return True
        # todo: publish training stats and confusion matrix

        # todo: publish js model:
        zip_name = "./best_web_model"
        return False

    def _cleanup_env(self):
        clean_dirs = ["/app/runs/train/exp", "/tmp/dataset/train", "/tmp/dataset/val"]  # TODO update to config vars
        for cd in clean_dirs:
            shutil.rmtree(cd)
        return True

    def get_follow_up_job(self):
        job_dependency = {
            "type": "model",
            "url": 'http://{}/api/v1/download?type=model&id=' + f'{self.data["model_id"]}',
            "name": None
        }

        req_payload = {
            "model_id": self.data['model_id']
        }
        req_json = {
            "code": JOB_CODE_EVALUATE_MODEL,
            "payload": req_payload,
            # "customer_mnemonic": self.job['tenant'],   # TODO provide actual mnemonic not id
            "ref_job": self.job.id,
            "dependencies": [job_dependency]
        }
        # TODO add authorisation header after merging with DSE-42
        resp = requests.post(f"http://{self.configuration.master_url}/api/v1/addJob", json=req_json)
        if resp.status_code >= 400:
            return False
        return


class BuildModelHierarchyRunner(TrainYoloRunner):
    default_epochs = 5

    def __init__(self, config=None):
        super().__init__(config)
        self.dataset_clusters = []
        self.confusion_threshold = 0  # Normalised value
        return

    def post_execution(self):
        if "confusion_threshold" in self.data:
            self.confusion_threshold = self.data["confusion_threshold"]
        success = self._cluster_datasets()
        return super().post_execution()

    def _cluster_datasets(self):
        confusion_matrix = pd.read_csv(
            "confusion_matrix.csv")  # TODO define a right path {save_dir}/confusion_matrix.csv
        # Get the number of classes from the confusion matrix, -1 to exlude the last row/column for background
        num_classes = len(confusion_matrix) - 1

        # Iterate over the rows of the confusion matrix
        for i in range(num_classes):
            # Check if the row has any mispredictions above the threshold, mispredictions is a list of elements that are mispredicted with the i'th element
            mispredictions = confusion_matrix.columns[confusion_matrix.iloc[i] > self.confusion_threshold].tolist()
            if len(mispredictions) > 0:
                # Check if the mispredictions belong to existing clusters
                matched_clusters = []
                for cluster in self.dataset_clusters:
                    if any(m in cluster for m in mispredictions):
                        matched_clusters.append(cluster)

                if len(matched_clusters) > 0:
                    # Add the mispredictions to the first matched cluster
                    matched_clusters[0].extend(mispredictions)
                    # Merge the matched clusters if there are multiple
                    if len(matched_clusters) > 1:
                        merged_cluster = []
                        for cluster in matched_clusters:
                            merged_cluster.extend(cluster)
                        self.dataset_clusters = [merged_cluster] + [cluster for cluster in self.dataset_clusters if
                                                                    cluster not in matched_clusters]
                else:
                    # Create a new cluster for the mispredictions
                    self.dataset_clusters.append(mispredictions)

        return True

    def get_follow_up_job(self):
        # For each cluster post seperate PrepareDataset job
        for cluster in self.dataset_clusters:
            classes = []
            for product_class in self.data["classes"]:
                # Get only classes that belong to current cluster
                if product_class['name'] in cluster:
                    classes.append(product_class)
            req_payload = {
                "follow_up_job": "JobBuildModelHierarchy",
                "classes": classes
            }
            if "perspective_filter" in self.data:
                req_payload["perspective_filter"] = self.data["perspective_filter"]
            if "batch" in self.data:
                req_payload["batch"] = self.data["batch"]
            if "epochs" in self.data:
                req_payload["epochs"] = self.data["epochs"]
            if "retrain" in self.data:
                req_payload["retrain"] = self.data["retrain"]
            if "confusion_threshold" in self.data:
                req_payload["confusion_threshold"] = self.data["confusion_threshold"]
            req_json = {
                "code": "JobPrepareTrainingFiles",
                "payload": req_payload,
                # "customer_mnemonic": self.job.tenant,   # TODO provide actual mnemonic not id
                "ref_job": self.job.id
            }
            # TODO add authorisation header
            resp = requests.post(f"http://{self.configuration.master_url}/api/v1/addJob", json=req_json)
            if resp.status_code >= 400:
                return False

        return True


class EvaluateModelRunner(AbstractRunner):
    default_imgsz = (512, 512)
    default_conf_thres = 0.25
    default_iou_thres = 0.45
    default_nosave_img = True
    default_save_conf = False

    def __init__(self, config=None):
        super().__init__(config)
        self.model_id = None

        return

    def _download_dependency(self, dep):
        if dep['type'] == 'model':
            resp = urlopen(dep["url"].format(self.configuration.master_url))
            download_path = f"/tmp/eval_weights/{self.model_id}.pt"
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, "wb") as datafile:
                datafile.write(resp.read())
        return

    def pre_execution(self):
        super().pre_execution()

        self.test_data_dir = f"{self.configuration.data_path}"
        self._create_dir(self.test_data_dir)
        self.save_dir = f"{self.test_data_dir}/test_results_combined"
        self._create_dir(self.save_dir)
        self._download_evaluation_data()

        if "imgsz" not in self.data:
            self.data["imgsz"] = self.default_imgsz
        if "conf_thres" not in self.data:
            self.data["conf_thres"] = self.default_conf_thres
        if "iou_thres" not in self.data:
            self.data["iou_thres"] = self.default_iou_thres
        if "nosave_img" not in self.data:
            self.data["nosave_img"] = self.default_nosave_img
        if "nosave" not in self.data:
            self.data["nosave_conf"] = self.default_nosave

        return

    def _download_evaluation_data(self):
        json = {
            "tenant_id": self.job['tenant'],
            "model_id": self.data['model_id']}
        # TODO add authorisation
        resp = requests.post(f'http://{self.configuration.master_url}/api/v1/downloadEvaluationDataset', json=json)
        if resp.status_code == 404:
            self.update_job_state(self.job.id, 'failed', retry_job=False)
            raise Exception(resp.message)

        zipfile = ZipFile(BytesIO(resp.read()))
        # Extract all contents of the ZIP file to the specified directory
        zipfile.extractall(path=f"{self.test_data_dir}/eval_data")
        return True

    def execution(self):
        eval_dataset_dir = f"{self.test_data_dir}/eval_data"
        classes = None
        if os.path.exists(f"{eval_dataset_dir}/classes.txt"):
            with open(f"{eval_dataset_dir}/classes.txt", "r") as file:
                classes = [line.strip() for line in file]

        for dataset_dir in os.listdir(eval_dataset_dir):  # can be both video and photos
            if os.path.isdir(f"{eval_dataset_dir}/{dataset_dir}") and dataset_dir in ['video_dataset',
                                                                                      "images_dataset"]:
                self.dataset_type = dataset_dir.rstrip("_dataset")
                detections_dir = self._run_detection()
                self._create_dir(detections_dir)
                labels_dir = f"{eval_dataset_dir}/{dataset_dir}/labels"
                results, confidences, bboxes, filenames, is_labeled = model_eval.extract_results(detections_dir,
                                                                                                 labels_dir)
                self.combined_results = model_eval.combine_results(results, confidences, bboxes,
                                                                   #    save_dir=self.save_dir,
                                                                   dataset_type=self.dataset_type, class_names=classes)
                self.problematic_classes = model_eval.get_problematic_classes(self.combined_results)
                model_eval.create_gannt_diagram(results, filenames, class_names=classes,
                                                problematic_classes=self.problematic_classes, model_id=self.model_id,
                                                path=self.save_dir, dataset_type=self.dataset_type,
                                                is_labeled=is_labeled)
        return

    def _run_detection(self):
        detections_dir = f"{self.test_data_dir}/{self.dataset_type}_detections"
        self._create_dir(detections_dir)

        detect.run(
            weights=f"/tmp/eval_weights/{self.model_id}.pt",
            source=f"{self.test_data_dir}/{self.dataset_type}_dataset",
            data=self.configuration.training_config_path,
            imgsz=self.data["imgsz"],
            conf_thres=self.data["conf_thres"],
            iou_thres=self.data["iou_thres"],
            device=self.data['device'],
            save_txt=True,
            save_conf=self.data['nosave_conf'],
            nosave=self.data['nosave_img'],
            project=detections_dir,
        )
        return detections_dir

    def _create_dir(self, full_path):
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return

    def post_execution(self):
        self._publish_results()
        return

    def _publish_results(self):
        json_results = {"results": self.combined_results, "problematic_classes": self.problematic_classes}
        with open(f"{self.save_dir}/detection_results.json", 'w') as fp:
            json.dump(json_results, fp)

        # Create a zip file containing all json files and gannt diagrams
        zip_filename = f"{self.save_dir}/eval_results_{self.model_id}.zip"
        shutil.make_archive(zip_filename, 'zip', self.save_dir)

        with open(zip_filename, "rb") as test_results_zip:
            data = test_results_zip.read()
            data_str = base64.b64encode(data).decode("utf-8")
        request_data = {
            "data_raw": data_str
        }
        resp = requests.post(
            f"http://{self.configuration.master_url}/api/v1/upload?obj=model_eval_results&id={self.data['model_id']}",
            json=request_data)
        if resp.status_code < 400:
            return True
        print('Publishing test results didnt succeed!')  # TODO: handle fail
        return False

    def cleanup(self):
        shutil.rmtree(self.test_data_dir)
        shutil.rmtree(f"/tmp/eval_weights")
        return


class JobExecutor(object):

    def execute(self, job):
        if job['code'] == JOB_CODE_TRAIN_YOLO:
            runner = TrainYoloRunner()
            runner.execute(job)
        elif job['code'] == JOB_CODE_BUILD_MODEL_HIERARCHY:
            runner = BuildModelHierarchyRunner()
            runner.execute(job)
        elif job['code'] == JOB_CODE_EVALUATE_MODEL:
            runner = EvaluateModelRunner()
            runner.execute(job)
            return
        return
