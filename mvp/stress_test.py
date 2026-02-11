import tracer
import sys

# === 模拟一个复杂的内存场景 ===
# 这是一个旧版 Tracer 绝对处理不了的场景
code_heavy = """
# 1. 一个巨大的列表 (10万个元素)
huge_list = [i for i in range(100000)]

# 2. 一个巨大的字典
huge_dict = {f"key_{i}": i for i in range(10000)}

# 3. 一个深层递归对象
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

head = Node(0)
curr = head
for i in range(100):
    curr.next = Node(i+1)
    curr = curr.next
"""

print("-" * 50)
print("🚀 开始压力测试...")
print("-" * 50)

# 运行你的 Smart Summary Tracer
traces = tracer.trace_execution(code_heavy)

# 取最后一步的状态
last_state = traces[-1]['vars']

print(f"\n✅ 捕获到的变量: {list(last_state.keys())}")

# 检查 huge_list 被压缩成了什么样
summary_list = last_state['huge_list']
print(f"\n📦 Huge List 摘要:\n{summary_list}")
print(f"   - 原始长度: ~600,000 字符 (如果用 str())")
print(f"   - 压缩后长度: {len(summary_list)} 字符")

# 检查 huge_dict
print(f"\n📦 Huge Dict 摘要:\n{last_state['huge_dict']}")

# 检查链表
print(f"\n📦 Linked List 摘要:\n{last_state['head']}")

print("-" * 50)
print("结论: 如果没有这一步，CodeBERT 早就因为 Token > 512 报错或截断崩溃了。")