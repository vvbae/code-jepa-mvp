import torch
from datasets import load_dataset
from tqdm import tqdm
import tracer  # 导入我们在上一步写的 tracer.py

# === 配置 ===
OUTPUT_FILE = "mbpp_traces.pt"
MAX_SAMPLES = None # 设置为 None 则跑全量，设置为 10 可以快速测试

def format_state(vars_dict):
    """
    把变量字典转成字符串，方便后续 Tokenizer 处理
    例如: {'x': 1, 'y': 2} -> "x: 1, y: 2"
    """
    if not vars_dict:
        return "<empty>"
    # 排序保证顺序一致性
    items = sorted(vars_dict.items())
    return ", ".join([f"{k}: {v}" for k, v in items])

def main():
    # 1. 加载数据
    print("Loading MBPP dataset...")
    dataset = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
    
    all_training_data = []
    success_count = 0
    
    # 2. 遍历每个函数
    # 如果你想快速测试，可以用 dataset.select(range(10))
    iterable_dataset = dataset if MAX_SAMPLES is None else dataset.select(range(MAX_SAMPLES))

    print(f"Start tracing {len(iterable_dataset)} functions...")
    
    for sample in tqdm(iterable_dataset):
        # 拼接代码：函数定义 + 第一个测试用例
        # 这样 exec() 才会真正运行函数逻辑
        full_code = sample['code'] + "\n" + sample['test_list'][0]
        
        # 运行沙盒追踪
        traces = tracer.trace_execution(full_code)
        
        # 如果追踪结果少于2步，说明没跑起来或者报错了，跳过
        if len(traces) < 2:
            continue
            
        success_count += 1
        
        # 3. 构建 (Current Line, Previous State) -> (Next State) 样本对
        # 我们从第1步遍历到最后一步
        for i in range(1, len(traces)):
            prev_step = traces[i-1]
            curr_step = traces[i]
            
            # 获取当前行的代码文本
            # full_code 是字符串，我们需要按换行符切分来找行号
            # 注意：tracer 返回的 lineno 是绝对行号，这比较麻烦
            # 为了简化 MVP，我们暂时只存 State 转换，假设模型能通过 Embedding 知道是哪一行
            # 或者我们简单地存一下行号，以后再通过 Encoder 找对应代码
            
            data_point = {
                "task_id": sample['task_id'],
                "lineno": curr_step['lineno'],
                "prev_state": format_state(prev_step['vars']),
                "next_state": format_state(curr_step['vars'])
            }
            all_training_data.append(data_point)

    # 4. 保存结果
    print(f"-" * 30)
    print(f"Tracing finished!")
    print(f"Successfully traced functions: {success_count}/{len(iterable_dataset)}")
    print(f"Total training samples generated: {len(all_training_data)}")
    
    print(f"Saving to {OUTPUT_FILE}...")
    torch.save(all_training_data, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()