import json
import os
from typing import List
import pandas as pd

import datasets
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset


_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "Human preference data about helpfulness and harmlessness."
_CITATION = ""
_HOMEPAGE = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf"
_LICENSE = "mit"
_URL = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf/resolve/main/"
_URLS = {
    "train": [
        _URL + "harmless-base/train.jsonl.gz",
        _URL + "helpful-base/train.jsonl.gz",
        _URL + "helpful-online/train.jsonl.gz",
        _URL + "helpful-rejection-sampled/train.jsonl.gz",
    ],
    "test": [
        _URL + "harmless-base/test.jsonl.gz",
        _URL + "helpful-base/test.jsonl.gz",
        _URL + "helpful-online/test.jsonl.gz",
        _URL + "helpful-rejection-sampled/test.jsonl.gz",
    ],
}
_EXPORT_JSON = True
_CONSTRUCT_DATA = True


class HhRlhfEn(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "instruction": datasets.Value("string"),
                "chosen": datasets.Value("string"),
                "rejected": datasets.Value("string"),
                "history": datasets.Sequence(
                    datasets.Sequence(datasets.Value("string"))
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_path["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepaths": file_path["test"]}
            ),
        ]

    def _generate_examples(self, filepaths: List[str]):
        key = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    assist_idx = rejected.rfind("\n\nAssistant: ")
                    r_reject = rejected[assist_idx + 13 :].strip()
                    assist_idx = chosen.rfind("\n\nAssistant: ")
                    r_accept = chosen[assist_idx + 13 :].strip()

                    human_idx = chosen.rfind("\n\nHuman: ")
                    query = chosen[human_idx + 9 : assist_idx].strip()
                    prompt = chosen[:human_idx]
                    history = []

                    while prompt.rfind("\n\nAssistant: ") != -1:
                        assist_idx = prompt.rfind("\n\nAssistant: ")
                        human_idx = prompt.rfind("\n\nHuman: ")
                        if human_idx != -1:
                            old_query = prompt[human_idx + 9 : assist_idx].strip()
                            old_resp = prompt[assist_idx + 13 :].strip()
                            history.insert(0, (old_query, old_resp))
                        else:
                            break
                        prompt = prompt[:human_idx]

                    yield key, {
                        "instruction": query,
                        "chosen": r_accept,
                        "rejected": r_reject,
                        "history": history,
                    }
                    key += 1


def convert_to_finetuning_format(dataset, output_file):
    """
    转换数据集为微调格式，并保存为JSON文件。

    Args:
    - dataset: Hugging Face 数据集
    - output_file: 输出文件的路径
    """
    finetuned_data = []

    for sample in dataset:
        # 创建微调数据格式
        finetuned_sample = {
            "instruction": sample["instruction"],
            "input": "",  # input字段为空
            "output": sample["chosen"],  # output字段使用chosen
            "system": "",  # 系统提示词为空
            "history": sample["history"],  # history字段保持原样
        }
        finetuned_data.append(finetuned_sample)

    # 保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(finetuned_data, f, ensure_ascii=False, indent=2)
    print(f"数据已成功保存为：{output_file}")


def convert_to_preference_format(dataset, output_file):
    """
    转换数据集为偏好数据集格式，并保存为JSON文件。

    Args:
    - dataset: Hugging Face 数据集
    - output_file: 输出文件的路径
    """
    finetuned_data = []

    for sample in dataset:
        # 创建微调数据格式
        finetuned_sample = {
            "instruction": sample["instruction"],
            "input": sample["history"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
        finetuned_data.append(finetuned_sample)

    # 保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(finetuned_data, f, ensure_ascii=False, indent=2)
    print(f"数据已成功保存为：{output_file}")


if __name__ == "__main__":
    # 加载训练集和测试集
    dataset = load_dataset(
        path="data/hh_rlhf_en/hh_rlhf_en.py",  # 脚本所在目录路径
        name="hh_rlhf_en",  # 数据集名称（与类名一致）
    )

    # 查看数据集结构
    print(dataset)

    # 访问训练集前5个样本
    train_samples = dataset["train"].select(range(5))
    for sample in train_samples:
        print("Instruction:", sample["instruction"])
        print("Chosen Response:", sample["chosen"])
        print("Rejected Response:", sample["rejected"])
        print("History:", sample["history"])
        print("\n---\n")

    # 数据集根目录
    data_dir = "data/hh_rlhf_en/"

    # 导出数据集为 JSON 文件
    if _EXPORT_JSON == True:
        test_data = dataset["test"]
        df = pd.DataFrame(test_data)
        df.to_json(
            data_dir + "test_data.json",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print("测试集已经导出为合法的 JSON 文件。")

        train_data = dataset["train"]
        df = pd.DataFrame(train_data)
        df.to_json(
            data_dir + "train_data.json",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print("训练集已经导出为合法的 JSON 文件。")

    if _CONSTRUCT_DATA == True:
        train_data = dataset["train"]
        test_data = dataset["test"]
        # 微调数据集
        convert_to_finetuning_format(train_data, data_dir + "train_finetuned.json")
        convert_to_finetuning_format(test_data, data_dir + "test_finetuned.json")

        # 偏好数据集
        convert_to_preference_format(train_data, data_dir + "train_preference.json")
        convert_to_preference_format(test_data, data_dir + "test_preference.json")
