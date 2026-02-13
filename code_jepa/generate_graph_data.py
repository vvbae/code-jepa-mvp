import torch
from datasets import load_dataset
from tqdm import tqdm
import graph_tracer  # 调用你刚才写好的 tracer
import os

# === 配置 ===
OUTPUT_FILE = "mbpp_graphs.pt"
MAX_SAMPLES = None  # 设为 100 可以先做快速测试，None 跑全量
# 注意：图数据比较占内存，如果 OOM，可以减小 batch 或分块保存

def main():
    print("Loading MBPP dataset...")
    # 记得用系统盘缓存
    dataset = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True, cache_dir="/root/.cache/huggingface")
    
    all_samples = []
    iterable_dataset = dataset if MAX_SAMPLES is None else dataset.select(range(MAX_SAMPLES))

    print(f"Generating Graph Data...")
    
    success_count = 0
    
    for sample in tqdm(iterable_dataset):
        try:
            full_code = sample['code'] + "\n" + sample['test_list'][0]
            code_lines = full_code.splitlines()
            
            # 1. 获取图序列 [G_0, G_1, G_2 ...]
            snapshots = graph_tracer.trace_execution(full_code)
            
            if len(snapshots) < 2: continue
            
            # 2. 构建训练对 (Prev Graph + Code Line -> Next Graph)
            for i in range(1, len(snapshots)):
                prev_graph = snapshots[i-1]
                curr_graph = snapshots[i]
                
                # 获取对应的代码行
                # graph_tracer 里我们把 lineno 挂在了 graph 对象上
                line_idx = curr_graph.lineno - 1
                if 0 <= line_idx < len(code_lines):
                    code_text = code_lines[line_idx].strip()
                else:
                    continue
                
                if not code_text or code_text.startswith('#'):
                    continue
                
                # 存储数据: (Prev_Graph, Code_Text, Next_Graph)
                # PyG 的 Data 对象可以直接存进 list
                all_samples.append({
                    "prev_graph": prev_graph,
                    "code": code_text,
                    "next_graph": curr_graph
                })
                success_count += 1
                
        except Exception as e:
            continue

    print(f"Successfully generated {success_count} graph transitions.")
    print(f"Saving to {OUTPUT_FILE}...")
    torch.save(all_samples, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()