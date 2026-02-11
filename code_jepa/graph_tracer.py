import sys
import torch
from torch_geometric.data import Data
import io
import contextlib

# === 配置: 节点特征定义 ===
# 我们用一个简单的 3维向量代表每个节点: [Type_ID, Size, Value_Hash]
# Type_ID: 0=Unknown, 1=Int/Float, 2=Str, 3=List/Tuple, 4=Dict, 5=Object
TYPE_MAP = {
    int: 1, float: 1, 
    str: 2, 
    list: 3, tuple: 3, 
    dict: 4
}

def get_type_id(obj):
    return TYPE_MAP.get(type(obj), 5) # 默认为 5 (Object)

def get_value_hash(obj):
    # 将值简单哈希化，归一化到 -1~1 之间
    try:
        if isinstance(obj, (int, float)):
            return float(obj) % 1000 / 1000.0
        if isinstance(obj, str):
            return float(len(obj)) / 100.0
        return 0.0
    except:
        return 0.0

def build_memory_graph(local_vars):
    """
    核心黑科技：把当前的 local_vars 变成一张 PyG 的图 (Data 对象)
    """
    node_map = {} # id(obj) -> node_index
    nodes = []    # 存储节点特征 [Type, Size, Value]
    edges = []    # 存储边 [Source, Target]
    
    # 待遍历队列 (BFS)
    queue = []
    
    # 1. 先把所有根变量 (Root Variables) 加进图
    for name, value in local_vars.items():
        if name.startswith('__') or isinstance(value, type(sys)):
            continue
            
        # 这是一个简单的 Trick：把变量名也当作一种特殊的节点或者属性
        # 这里为了简化，我们直接从 Value 开始遍历
        obj_id = id(value)
        if obj_id not in node_map:
            node_map[obj_id] = len(nodes)
            # 特征: [Type, Size/Len, Hash]
            size = len(value) if hasattr(value, '__len__') else 0
            nodes.append([get_type_id(value), size, get_value_hash(value)])
            queue.append(value)

    # 2. 开始爬取内存 (BFS)
    # 限制步数防止全图遍历太慢
    steps = 0
    MAX_NODES = 100 # 限制一张图最多 100 个节点，防止显存爆炸
    
    while queue and len(nodes) < MAX_NODES:
        curr_obj = queue.pop(0)
        curr_idx = node_map[id(curr_obj)]
        
        # 提取子对象 (Children)
        children = []
        
        if isinstance(curr_obj, (list, tuple)):
            children = curr_obj
        elif isinstance(curr_obj, dict):
            children = list(curr_obj.keys()) + list(curr_obj.values())
        elif hasattr(curr_obj, '__dict__'):
            children = list(curr_obj.__dict__.values())
            
        # 建立边
        for child in children:
            child_id = id(child)
            
            # 如果是新节点
            if child_id not in node_map:
                if len(nodes) >= MAX_NODES: break
                
                node_map[child_id] = len(nodes)
                size = len(child) if hasattr(child, '__len__') else 0
                nodes.append([get_type_id(child), size, get_value_hash(child)])
                queue.append(child)
            
            # 添加边: curr_obj -> child
            edges.append([curr_idx, node_map[child_id]])

    # 3. 转换为 PyTorch Geometric Data 格式
    if len(nodes) == 0:
        # 空状态: 只有一个虚拟节点
        x = torch.tensor([[0, 0, 0]], dtype=torch.float)
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        x = torch.tensor(nodes, dtype=torch.float)
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            
    return Data(x=x, edge_index=edge_index)

def trace_execution(code_str):
    """
    执行代码，返回 [Data, Data, Data] (图序列)
    """
    snapshots = []
    
    def trace_calls(frame, event, arg):
        if event != 'line': return trace_calls
        
        # 每一行执行完，我们就拍一张“内存快照”
        # 注意：深拷贝图结构很慢，我们这里构建的是轻量级图
        try:
            graph = build_memory_graph(frame.f_locals)
            # 把 lineno 挂在 graph 对象上，方便 dataset 使用
            graph.lineno = frame.f_lineno 
            snapshots.append(graph)
        except Exception:
            pass # 忽略构图错误，保证代码不崩
            
        return trace_calls

    capture_io = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture_io):
            sys.settrace(trace_calls)
            exec(code_str, {}, {})
            sys.settrace(None)
    except:
        sys.settrace(None)

    return snapshots

# === 自测代码 ===
if __name__ == "__main__":
    print("Testing Graph Tracer...")
    code = """
x = [1, 2]
y = x        # 引用!
x.append(3)  # y 应该也变了
    """
    graphs = trace_execution(code)
    
    print(f"Captured {len(graphs)} steps.")
    for i, g in enumerate(graphs):
        print(f"Step {i+1} (Line {g.lineno}): {g.num_nodes} Nodes, {g.num_edges} Edges")
        # 验证引用：如果 y = x，它们应该指向同一个节点索引吗？
        # 在这个简易版里，我们是通过 id() 映射的，所以拓扑结构会体现出来