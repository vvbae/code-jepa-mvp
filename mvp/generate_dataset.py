import torch
from datasets import load_dataset
from tqdm import tqdm
import tracer
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# === 配置 ===
OUTPUT_FILE = "mbpp_traces_v2.pt"
MAX_SAMPLES = None 
NUM_WORKERS = 16  # <--- 并行数量，A40机器通常CPU核心很多，拉满！

def format_state(vars_dict):
    if not vars_dict: return "<empty>"
    items = sorted(vars_dict.items())
    return ", ".join([f"{k}: {v}" for k, v in items])

def process_single_sample(sample):
    """
    单个样本的处理逻辑，为了并行化，必须封装成独立的函数
    """
    try:
        full_code = sample['code'] + "\n" + sample['test_list'][0]
        code_lines = full_code.splitlines()
        
        # 这一步最耗时，现在它会在独立的 CPU 核心上跑
        traces = tracer.trace_execution(full_code)
        
        if len(traces) < 2: 
            return []
            
        sample_data = []
        for i in range(1, len(traces)):
            prev_step = traces[i-1]
            curr_step = traces[i]
            
            line_idx = curr_step['lineno'] - 1
            if 0 <= line_idx < len(code_lines):
                code_text = code_lines[line_idx].strip()
            else:
                continue

            if not code_text or code_text.startswith('#'):
                continue

            data_point = {
                "prev_state": format_state(prev_step['vars']),
                "code": code_text,
                "next_state": format_state(curr_step['vars'])
            }
            sample_data.append(data_point)
            
        return sample_data
    except Exception:
        return []

def main():
    # 强制设置环境变量，防止 fork 后丢失
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Loading MBPP dataset...")
    # cache_dir 指定到系统盘，避开 RunPod 的慢速网络盘
    dataset = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True, cache_dir="/root/.cache/huggingface")
    
    all_training_data = []
    iterable_dataset = dataset if MAX_SAMPLES is None else dataset.select(range(MAX_SAMPLES))
    total_samples = len(iterable_dataset)

    print(f"Generating Phase 2 Data with {NUM_WORKERS} workers...")
    
    # === 并行处理核心 ===
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_sample, sample) for sample in iterable_dataset]
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=total_samples, desc="Tracing"):
            result = future.result()
            if result:
                all_training_data.extend(result)

    print(f"Saving {len(all_training_data)} samples to {OUTPUT_FILE}...")
    torch.save(all_training_data, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()