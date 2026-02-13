import sys
import trace
import io
import contextlib

# === 配置 ===
MAX_LIST_ITEMS = 3      # 列表只看前3个和后3个
MAX_STRING_LEN = 50     # 字符串只看前50个字符
MAX_DICT_KEYS = 3       # 字典只看前3个Key

def summarize_value(value, depth=0):
    """
    核心函数：把任意 Python 对象压缩成简短的特征字符串
    """
    # 防止递归太深 (比如对象里套对象)
    if depth > 2:
        return "..."

    type_name = type(value).__name__

    # 1. 处理列表和元组
    if isinstance(value, (list, tuple)):
        length = len(value)
        if length == 0:
            return f"{type_name}[]"
        
        # 如果列表很短，直接显示
        if length <= MAX_LIST_ITEMS * 2:
            items_str = ", ".join([summarize_value(v, depth+1) for v in value])
            return f"{type_name}[{items_str}]"
        
        # 如果列表很长，截断：Head ... Tail
        head = ", ".join([summarize_value(v, depth+1) for v in value[:MAX_LIST_ITEMS]])
        tail = ", ".join([summarize_value(v, depth+1) for v in value[-MAX_LIST_ITEMS:]])
        return f"{type_name}(len={length})[{head}, ..., {tail}]"

    # 2. 处理字典
    elif isinstance(value, dict):
        length = len(value)
        if length == 0:
            return "Dict{}"
        
        keys = list(value.keys())
        if length <= MAX_DICT_KEYS:
            # 显示所有 Key (Value 简略)
            content = ", ".join([f"{k}: {summarize_value(value[k], depth+1)}" for k in keys])
            return f"Dict{{{content}}}"
        else:
            # 截断
            shown_keys = keys[:MAX_DICT_KEYS]
            content = ", ".join([f"{k}" for k in shown_keys])
            return f"Dict(len={length}){{{content}, ...}}"

    # 3. 处理字符串
    elif isinstance(value, str):
        if len(value) <= MAX_STRING_LEN:
            return repr(value)
        return f"Str(len={len(value)}) {repr(value[:MAX_STRING_LEN])}..."

    # 4. 处理基本类型 (int, float, bool, None)
    elif isinstance(value, (int, float, bool, type(None))):
        return str(value)

    # 5. 处理函数和类对象
    elif callable(value):
        return f"<Func: {value.__name__}>"
    
    # 6. 其他对象 (Object)
    else:
        # 尝试获取对象的属性
        try:
            attrs = vars(value)
            # 如果属性不多，显示出来
            if len(attrs) < 3:
                return f"<{type_name}: {summarize_value(attrs, depth+1)}>"
            return f"<{type_name} at {hex(id(value))}>"
        except:
            return f"<{type_name}>"

def trace_execution(code_str):
    """
    执行代码并记录每一步的 Smart Summary 状态
    """
    traces = []
    
    # 局部作用域 (Local Scope)
    local_scope = {}
    
    def trace_calls(frame, event, arg):
        if event != 'line':
            return trace_calls
        
        # 获取当前行号
        lineno = frame.f_lineno
        
        # 获取当前局部变量
        current_vars = frame.f_locals.copy()
        
        # === 关键修改：只记录用户定义的变量，且使用 summarize_value ===
        filtered_vars = {}
        for k, v in current_vars.items():
            # 过滤掉 Python 内置变量 (__) 和 模块导入
            if not k.startswith('__') and not isinstance(v, type(sys)):
                filtered_vars[k] = summarize_value(v)
        
        traces.append({
            'step': len(traces) + 1,
            'lineno': lineno,
            'vars': filtered_vars
        })
        return trace_calls

    # 捕获 stdout 防止打印干扰
    capture_io = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(capture_io):
            sys.settrace(trace_calls)
            exec(code_str, {}, local_scope)
            sys.settrace(None)
    except Exception as e:
        sys.settrace(None)
        # 如果代码报错，我们也记录一下错误状态
        traces.append({
            'step': len(traces) + 1,
            'lineno': -1,
            'vars': {"ERROR": str(e)}
        })

    return traces

# === 简单测试 ===
if __name__ == "__main__":
    code = """
x = [i for i in range(1000)]
y = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
z = "Hello " * 50
    """
    print("Testing Smart Summary Tracer...")
    results = trace_execution(code)
    for step in results:
        print(f"Line {step['lineno']}: {step['vars']}")