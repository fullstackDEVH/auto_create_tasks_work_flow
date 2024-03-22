from typing import Dict, List
import logging
import json
import os
import zipfile
import shutil
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobFlow:
    def __init__(self, project_config: Dict, n_task: int):
        if not project_config:
            raise ValueError("JobFlow needed project config")
        logger.info("Initializing jobflow")
        self.n_task = n_task

        job_flow_base, node_length_clone = self.create_job_flow_by_config(
            project_config
        )
        self.flow = job_flow_base
        self.node_length = node_length_clone
        self.dataset: Dict = {}
        self.tasks: Dict = {}
        self.task_data: Dict = {}
        self.batchs: Dict = {"data": [], "current_index_batch": None}

    def create_job_flow_by_config(self, project_config: Dict):
        logger.info("create_job_flow_by_config called")

        node_length_clone = {"OCR": 0, "OCR_EXT": 0, "ET": 0, "CK": 0, "LC": 0}

        nodes: Dict = project_config["nodes"]
        edges: Dict = project_config["edges"]

        job_flow_base: Dict = {}

        for node in nodes:
            source_node_id = node["id"]
            node_name_origin = node["data"]["label"]

            if node_name_origin in node_length_clone:
                node_length_clone[node_name_origin] += 1

            for edge in edges:
                source_edge_id = edge["source"]
                target_edge_id = edge["target"]

                if source_node_id != source_edge_id:
                    continue

                child_flow = {
                    "target_id": target_edge_id,
                    "edge_condition": (
                        edge["label"] if "COND" in source_edge_id else True
                    ),
                }

                if source_edge_id not in job_flow_base:
                    job_flow_base[source_edge_id] = [child_flow]
                else:
                    job_flow_base[source_edge_id].append(child_flow)
        return job_flow_base, node_length_clone

    def extract_zip_and_create_folder_contain_file(self, zip_file):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for member in zip_ref.namelist():
                member_path = os.path.join(extract_to_folder, member)
                # if is a folder
                if member.endswith("/"):
                    os.makedirs(member_path, exist_ok=True)
                # Nếu thành viên là file, trích xuất và ghi nó vào đường dẫn tương ứng
                else:
                    os.makedirs(os.path.dirname(member_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(
                        member_path, "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)

    def create_task_by_job_flow(self):
        tasks = {}
        """
            {
                "n-ET-1-1" : {
                    "dataset" : "dataset-1",    
                    "task_flow" : "ET1"
                }
            }
        """

        for node_name, total_node_name in self.node_length.items():
            for index in range(1, total_node_name + 1):
                for i in range(self.n_task):
                    dataset_name = f"DATASET-{i+1}"
                    task_name = f"n-{node_name}-{index}-{i+1}"
                    data = {
                        "dataset": dataset_name,
                        "task_flow": f"n-{node_name}-{index}",
                    }
                    if task_name not in tasks:
                        tasks[task_name] = {}
                    tasks[task_name] = data

        self.tasks = tasks
        return tasks

    def split_images_by_n_task(self, data_each_batchs: Dict, n_task: int):
        dataset_tmp: Dict = {}
        for batch_name, values_by_batch in data_each_batchs.items():
            total_img_by_batch = len(values_by_batch)
            image_each_part = math.ceil(total_img_by_batch / n_task)

            for index in range(n_task):
                end_index = (index + 1) * image_each_part
                start_index = end_index - image_each_part

                sliced_dict = {
                    k: values_by_batch[k]
                    for k in list(values_by_batch.keys())[start_index:end_index]
                }
                dataset_name = f"DATASET-{index+1}"

                if dataset_name not in dataset_tmp:
                    dataset_tmp[dataset_name] = {}
                if batch_name not in dataset_tmp[dataset_name]:
                    dataset_tmp[dataset_name][batch_name] = {}
                dataset_tmp[dataset_name][batch_name] = sliced_dict

        return dataset_tmp

    def import_data_working(self, zip_file):
        self.extract_zip_and_create_folder_contain_file(zip_file)
        batchs: List = []
        data_each_batchs: Dict = {}

        batch_folder = os.path.join(extract_to_folder, "0108")

        for batch_name in os.listdir(batch_folder):
            batch_path = os.path.join(batch_folder, batch_name)
            if os.path.isdir(batch_path):
                batchs.append(batch_name)
                data_each_batchs.setdefault(batch_name, {})

                txt_file_path = os.path.join(batch_path, f"{batch_name}.txt")
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, "r") as file:
                        for line in file:
                            values_line = [
                                item.strip('""') for item in line.strip().split(",")
                            ]
                            image_name = "".join(values_line[1:4])
                            data = {
                                "systax_limit": values_line[6],
                                "ocr": values_line[25],
                            }
                            data_each_batchs[batch_name].setdefault(image_name, data)

                txt_file_path_ext = os.path.join(batch_path, "rainscales_result.txt")
                if os.path.exists(txt_file_path_ext):
                    with open(txt_file_path_ext, "r") as file:
                        for line in file:
                            line_values = line.strip().split(",")
                            image_name = line_values[0][: line_values[0].rfind(".")]
                            data = data_each_batchs[batch_name].setdefault(
                                image_name, {}
                            )
                            data["ocr_ext"] = line_values[1]

        self.batchs: Dict = {"data": batchs, "current_index_batch": 0}

        self.dataset = self.split_images_by_n_task(data_each_batchs, self.n_task)

    def check_result_in_task(self, task_name):
        task = self.tasks[task_name]
        if "result" not in task:
            return False
        return True

    # manager task
    def check_pass_flow(self, nodes_name: List) -> bool:
        is_continue = False
        for node_name in nodes_name:
            task = {}
            if node_name in self.task_data:
                task = self.task_data[node_name]
            if (
                "status" not in task
                or (task["status"] != "FINISHED")
                # or "result" not in task
            ):
                is_continue = False
            else:
                is_continue = True
        return is_continue

    def get_task_by_name(self, task_name: str) -> Dict:
        return self.tasks[task_name]

    def compares_values(self, obj_one: Dict, obj_two: Dict) -> tuple:
        diffrent: Dict = {}
        common: Dict = {}

        for img_name, value in obj_one.items():
            if value is None or img_name not in obj_two:
                continue

            if value != obj_two[img_name]:
                diffrent[img_name] = obj_two[img_name]
            else:
                common[img_name] = value

        return common, diffrent

    def get_current_batch_working(self) -> str:
        return self.batchs["data"][self.batchs["current_index_batch"]]

    def lop(self, name_flow: str, index_data_set: int) -> Dict:

        current_batch_working = self.get_current_batch_working()
        current_task = {}
        task_key = f"{name_flow}-{index_data_set}"
        if task_key in self.task_data:
            current_task = self.task_data[task_key]
        # handle ocr
        if "OCR" in name_flow:
            convert_result: Dict = {}
            name_dataset: Dict = self.dataset[f"DATASET-{index_data_set}"]

            if "OCR-EXT" in name_flow:
                current_task["assigned"] = "OCR-EXT-bot"
                for batch_name, values in name_dataset.items():
                    if batch_name not in convert_result:
                        convert_result[batch_name] = {}
                    for img_name, obj_value in values.items():
                        convert_result[batch_name][img_name] = obj_value.get("ocr_ext")
                current_task["result"] = convert_result
            else:
                current_task["assigned"] = "OCR-bot"
                for batch_name, values in name_dataset.items():
                    if batch_name not in convert_result:
                        convert_result[batch_name] = {}
                    for img_name, obj_value in values.items():
                        if img_name == "A72306301887221A002":
                            convert_result[batch_name][img_name] = "5"
                        else:
                            convert_result[batch_name][img_name] = obj_value.get("ocr")
                current_task["result"] = convert_result

            current_task["status"] = "FINISHED"

        if "COND" in name_flow:
            if "status" in current_task and current_task["status"] == "FINISHED":
                self.open_base_task_by_flow(name_flow)
                return current_task

            node_name_has_target_is_COND: Dict = []

            for k, v in self.flow.items():
                for t in v:
                    if t["target_id"] == name_flow:
                        node_name_has_target_is_COND.append(f"{k}-{index_data_set}")
            # Kiểm tra nếu đầu vào của COND có đầy đủ, nếu không thì dừng flow của task
            is_continue: bool = self.check_pass_flow(node_name_has_target_is_COND)
            if is_continue is False:
                current_task["status"] = "INPROGRESS"
                return current_task
            else:
                current_task["status"] = "FINISHED"
            # compares values
            node_0_result = self.task_data[node_name_has_target_is_COND[0]]["result"]
            node_1_result = self.task_data[node_name_has_target_is_COND[1]]["result"]

            if (
                current_batch_working in node_0_result
                and current_batch_working in node_1_result
            ):
                common, different = self.compares_values(
                    node_0_result[current_batch_working],
                    node_1_result[current_batch_working],
                )
            else:
                common, different = (None, None)
            current_flow_COND: List = self.flow[name_flow]

            for flow in current_flow_COND:
                new_flow_target = flow["target_id"]
                edge_condition = flow["edge_condition"]
                task_data = {"status": "not_started", "assigned": None, "result": {}}

                if "result" not in task_data:
                    task_data["result"] = {}

                if current_batch_working not in task_data["result"]:
                    task_data["result"][current_batch_working] = {}

                if (
                    f"{new_flow_target}-{index_data_set}" in self.task_data
                    and "result"
                    in self.task_data[f"{new_flow_target}-{index_data_set}"]
                    and current_batch_working
                    in self.task_data[f"{new_flow_target}-{index_data_set}"]["result"]
                ):
                    old_value = self.task_data[f"{new_flow_target}-{index_data_set}"][
                        "result"
                    ][current_batch_working]

                    print(old_value)
                    if len(old_value) > 0:
                        task_data["result"][current_batch_working] = old_value

                if len(common) > 0 and edge_condition.lower() == "true":
                    task_data["result"][current_batch_working].update(common)
                    task_data["status"] = "open"
                    self.task_data[f"{new_flow_target}-{index_data_set}"] = task_data

                elif len(different) > 0 and edge_condition.lower() == "false":
                    task_data["result"][current_batch_working].update(different)
                    task_data["status"] = "open"
                    self.task_data[f"{new_flow_target}-{index_data_set}"] = task_data

                else:
                    task_data["result"][current_batch_working].update(different)
                    task_data["status"] = "FINISHED"
                    self.task_data[f"{new_flow_target}-{index_data_set}"] = task_data

            self.open_base_task_by_flow(name_flow)

        elif "status" in current_task and current_task["status"] == "FINISHED":
            self.open_base_task_by_flow(name_flow)
        elif "END" in name_flow:

            logger.info("LC Chưa END")
            return None
        elif "status" in current_task and current_task["status"] == "INPROGRESS":
            return current_task
        else:
            node_name_has_target: Dict = []
            print(name_flow)
            print(current_task)
            for k, v in self.flow.items():
                for t in v:
                    if t["target_id"] == name_flow:
                        node_name_has_target.append(f"{k}-{index_data_set}")
            is_continue: bool = self.check_pass_flow(node_name_has_target)
            # if is_continue is False or (
            #     current_task.get("result") is not None
            #     and current_task.get("result") != {}
            # ):
            #     current_task["status"] = "not_started"
            # else:
            #     current_task["status"] = "open"

        return current_task

    def lop1(self, name_flow: str, index_data_set: int) -> Dict:
        current_batch_working = self.get_current_batch_working()

        current_task = {"status": "not_started", "assigned": None, "result": {}}

        task_key = f"{name_flow}-{index_data_set}"
        if task_key in self.task_data:
            current_task = self.task_data[task_key]

        if name_flow.find("OCR") != -1:
            convert_result: Dict = {}
            name_dataset: Dict = self.dataset[f"DATASET-{index_data_set}"]

            if "OCR-EXT" in name_flow:
                current_task["assigned"] = "OCR-EXT-bot"
                for batch_name, values in name_dataset.items():
                    if batch_name not in convert_result:
                        convert_result[batch_name] = {}
                    for img_name, obj_value in values.items():
                        convert_result[batch_name][img_name] = obj_value.get("ocr_ext")
                current_task["result"] = convert_result
            else:
                current_task["assigned"] = "OCR-bot"
                for batch_name, values in name_dataset.items():
                    if batch_name not in convert_result:
                        convert_result[batch_name] = {}
                    for img_name, obj_value in values.items():
                        convert_result[batch_name][img_name] = obj_value.get("ocr")
                current_task["result"] = convert_result

            current_task["status"] = "FINISHED"

        for item in self.flow[name_flow]:
            target: str = item["target_id"]

            if target.find("END") != -1:
                node_name_has_target_is_END: Dict = []

                for k, v in self.flow.items():
                    for t in v:
                        if t["target_id"] == target:
                            node_name_has_target_is_END.append(f"{k}-{index_data_set}")

                # Kiểm tra nếu đầu vào của END có đầy đủ, nếu không thì dừng flow của task
                is_continue = self.check_pass_flow(node_name_has_target_is_END)
                if is_continue is False:
                    break

                # handle end when all dataset of batch is full value, after batch is finish => switch to other batch in data
                continue

            elif target.find("COND") != -1:
                node_name_has_target_is_COND: Dict = []

                for k, v in self.flow.items():
                    for t in v:
                        if t["target_id"] == target:
                            node_name_has_target_is_COND.append(f"{k}-{index_data_set}")

                # Kiểm tra nếu đầu vào của COND có đầy đủ, nếu không thì dừng flow của task
                is_continue = self.check_pass_flow(node_name_has_target_is_COND)

                if is_continue is False:
                    break

                # kiểm tra result của node flow đầu vào, nếu giá trị của các nodes flow giống nhau trong cùng 1 dataset thì chạy sang TRUE , ngược lại là False

                # dataset_result_by_batch = self.dataset[f"DATASET-${index_data_set}"][current_batch_working]

                tasks_data_compares: List = []

                for node_name in node_name_has_target_is_COND:
                    result = self.tasks[node_name]["result"][current_batch_working]
                    tasks_data_compares.append(result)
                    print(f"data : {result}")

                common, diffrent = self.compares_values(
                    tasks_data_compares[0], tasks_data_compares[1]
                )

                if len(diffrent) > 0:
                    for item in self.flow[name_flow]:
                        if item["edge_condition"] == False:
                            pass
                            # thay đổi target

            elif "OCR" not in name_flow:
                is_check = False
                task_name = f"{name_flow}-{index_data_set}"
                # Kiểm tra task hiên tại có result chưa nếu chưa thì dừng.
                is_check = self.check_result_in_task(task_name)

                if is_check == False:
                    current_task["status"] = "OPEN"
                    break

                # kiểm tra có result và trạng thái task laf finish thì mới sang flow tiếp theo
                # Nếu có thì sang flow tiếp theo bằng target

            self.lop(target)

        return current_task

    def open_base_task_by_flow(self, name_flow="n-START-1"):
        tasks_base: Dict = {}
        start_flow = self.flow[name_flow]

        for index_dataset in range(1, self.n_task + 1):
            for node_flow in start_flow:
                target = node_flow["target_id"]
                task_name = f"{target}-{index_dataset}"

                if task_name not in tasks_base:
                    tasks_base[task_name] = {}
                data = self.lop(target, index_dataset)
                if data is not None:
                    tasks_base[f"{task_name}"].update(data)

        # print(tasks_base)
        # for task_name, values in self.tasks.items():
        #     """
        #         values = {
        #             "dataset":"DATASET-1",
        #             "task_flow":"n-OCR-1"
        #         }
        #     """
        #     task_flow = values["task_flow"]
        #     dataset_name :str = values["dataset"]

        #     if task_name not in tasks_base:
        #         tasks_base[task_name] = {}

        #     tasks_base[task_name] = self.lop(task_flow, dataset_name.split("-")[1])
        # print(tasks_base)
        self.task_data.update(
            {
                key: value
                for key, value in tasks_base.items()
                if key not in self.task_data
            }
        )

    def import_tasks_data(self, task_data: Dict = {}):
        self.task_data = task_data

    def export_tasks_data(self) -> Dict:
        return self.task_data


file_path = "/home/huy/vscode/project_config.json"
task_data_file = "/home/huy/vscode/dop_testing/task_data_et.json"

zipFile = "/home/huy/Downloads/0108.zip"

extract_to_folder = "/home/huy/vscode/dop_testing/data"

with open(file_path, "r") as json_file:
    project_config = json.load(json_file)

with open(task_data_file, "r") as json_file:
    task_data = json.load(json_file)

job_flow_instance = JobFlow(project_config, 1)
job_flow_instance.import_tasks_data(task_data)
job_flow_instance.import_data_working(zipFile)

tasks = job_flow_instance.create_task_by_job_flow()
# print(job_flow_instance.flow)
# print(job_flow_instance.tasks)
# job_flow_instance.open_base_task_by_flow()

job_flow_instance.lop("n-ET-1", 1)
data = job_flow_instance.export_tasks_data()
print(data)

task_base = {
    "n-OCR-1-1": {"dataset": "DATASET-1", "task_flow": "n-OCR-1"},
    "n-OCR_EXT-1-1": {"dataset": "DATASET-1", "task_flow": "n-OCR_EXT-1"},
    "n-ET-1-1": {"dataset": "DATASET-1", "task_flow": "n-ET-1"},
    "n-CK-1-1": {"dataset": "DATASET-1", "task_flow": "n-CK-1"},
    "n-LC-1-1": {"dataset": "DATASET-1", "task_flow": "n-LC-1"},
}

job_flow = {
    "n-START-1": [
        {"target_id": "n-OCR-1", "edge_condition": "TRUE"},
        {"target_id": "n-OCR_EXT-1", "edge_condition": "TRUE"},
    ],
    "n-OCR_EXT-1": [
        {"target_id": "n-COND-1", "edge_condition": "TRUE"},
        {"target_id": "n-COND-2", "edge_condition": "TRUE"},
    ],
    "n-OCR-1": [{"target_id": "n-COND-1", "edge_condition": "TRUE"}],
    "n-COND-1": [
        {"target_id": "n-ET-1", "edge_condition": "FALSE"},
        {"target_id": "n-LC-1", "edge_condition": "TRUE"},
    ],
    "n-ET-1": [{"target_id": "n-COND-2", "edge_condition": "TRUE"}],
    "n-COND-2": [
        {"target_id": "n-CK-1", "edge_condition": "FALSE"},
        {"target_id": "n-LC-1", "edge_condition": "TRUE"},
    ],
    "n-CK-1": [{"target_id": "n-LC-1", "edge_condition": "TRUE"}],
    "n-LC-1": [{"target_id": "n-END-1", "edge_condition": "TRUE"}],
}
