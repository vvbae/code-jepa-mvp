import json

import torch
import tree_sitter_python as tspython
from datasets import load_dataset
from torch_geometric.data import Data
from tqdm import tqdm
from tree_sitter import Language, Parser

# === 配置 ===
LIMIT = 20000  # 实验阶段先搞 2 万条，MBPP 的 6 倍
SAVE_PATH = "csn_python_graphs.pt"
VOCAB_PATH = "node_type_vocab.json"

# 初始化 Parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


class CSNGraphProcessor:
    def __init__(self):
        self.type_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.data_list = []

    def get_type_id(self, type_name):
        if type_name not in self.type_to_idx:
            self.type_to_idx[type_name] = len(self.type_to_idx)
        return self.type_to_idx[type_name]

    def code_to_graph(self, code_str):
        try:
            tree = parser.parse(bytes(code_str, "utf8"))
            nodes_type_ids = []
            edge_index = [[], []]

            # 递归遍历 AST
            def traverse(node):
                curr_id = len(nodes_type_ids)
                nodes_type_ids.append(self.get_type_id(node.type))

                for child in node.children:
                    child_id = traverse(child)
                    # 添加双向边，增强 GNN 的消息传递
                    edge_index[0].append(curr_id)
                    edge_index[1].append(child_id)
                    edge_index[0].append(child_id)
                    edge_index[1].append(curr_id)
                return curr_id

            traverse(tree.root_node)

            x = torch.tensor(nodes_type_ids, dtype=torch.long).unsqueeze(1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            return x, edge_index
        except Exception:
            return None, None

    def process(self):
        print("Loading CodeSearchNet Python subset...")
        dataset = load_dataset(
            "code_search_net", "python", split="train", streaming=True
        )

        count = 0
        pbar = tqdm(total=LIMIT, desc="Parsing Graphs")

        for entry in dataset:
            if count >= LIMIT:
                break

            code = entry["whole_func_string"]
            x, edge_index = self.code_to_graph(code)

            if x is not None and x.size(0) > 1:
                # 存储为 PyG Data 对象
                graph_data = Data(x=x, edge_index=edge_index)
                # 顺便把原始代码存进去，方便之后跟 CodeBERT 对齐
                graph_data.raw_code = code
                self.data_list.append(graph_data)
                count += 1
                pbar.update(1)

        pbar.close()
        print(f"Saving {len(self.data_list)} graphs to {SAVE_PATH}...")
        torch.save(self.data_list, SAVE_PATH)

        with open(VOCAB_PATH, "w") as f:
            json.dump(self.type_to_idx, f)
        print(f"Vocab saved to {VOCAB_PATH}. Total types: {len(self.type_to_idx)}")


if __name__ == "__main__":
    processor = CSNGraphProcessor()
    processor.process()
``