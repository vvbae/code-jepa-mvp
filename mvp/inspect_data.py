from datasets import load_dataset

print("Loading MBPP dataset...")
dataset = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)

sample = dataset[0]
print("-" * 50)
print(f"Task ID: {sample['task_id']}")
print("Code Snippet:")
print(sample["code"])
print("-" * 50)
print("Test Cases:")
print(sample['test_list'][0])
print("-" * 50)

print(f"Total number of samples in the dataset: {len(dataset)}")