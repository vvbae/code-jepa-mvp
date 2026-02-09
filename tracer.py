import sys
import io
import contextlib
import signal

# === 配置 ===
MAX_STEPS = 100  # 防止死循环
TIMEOUT_SEC = 1  # 防止卡死

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException("Time Limit Exceeded")

def simple_serializer(obj):
    """把复杂对象转成字符串，防止报错"""
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, list):
        return [simple_serializer(x) for x in obj[:5]] # 只取前5个
    if isinstance(obj, dict):
        return {str(k): simple_serializer(v) for i, (k, v) in enumerate(obj.items()) if i < 5}
    return str(type(obj).__name__)

def trace_execution(code_str):
    """核心函数：执行代码并返回轨迹"""
    traces = []
    step_counter = 0

    # === 定义追踪钩子 ===
    def audit_hook(frame, event, arg):
        nonlocal step_counter
        
        # 必须返回自己，否则下一行就不追踪了！
        if event == 'call':
            return audit_hook
            
        if event == 'line':
            step_counter += 1
            if step_counter > MAX_STEPS:
                raise TimeoutException("Step Limit Exceeded")
            
            # 只追踪字符串里的代码，不追踪系统库
            if frame.f_code.co_filename != "<string>":
                return audit_hook

            line_no = frame.f_lineno
            
            # 抓取局部变量
            current_vars = {}
            # f_locals 包含了当前作用域的所有变量
            for k, v in frame.f_locals.items():
                if not k.startswith('_'): 
                    current_vars[k] = simple_serializer(v)
            
            traces.append({
                'step': step_counter,
                'lineno': line_no,
                'vars': current_vars
            })
            
            # === 关键修复：必须返回自身 ===
            return audit_hook
        
        return audit_hook

    # === 设置超时与执行 ===
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SEC)

    try:
        # 捕捉 print 输出，保持清爽
        with contextlib.redirect_stdout(io.StringIO()):
            sys.settrace(audit_hook) # 开启追踪
            exec(code_str, {})       # 执行代码
    except TimeoutException:
        pass 
    except Exception as e:
        # 代码本身报错也没关系，我们只要能跑通的那部分
        pass
    finally:
        sys.settrace(None) # 关闭追踪
        signal.alarm(0)

    return traces

# === 单元测试 ===
if __name__ == "__main__":
    sample_code = """
x = 0
for i in range(3):
    x = x + 1
    y = x * 10
"""
    print("Tracing sample code...")
    result = trace_execution(sample_code)
    
    if not result:
        print("Error: No traces captured!")
    else:
        for step in result:
            print(f"Step {step['step']} (Line {step['lineno']}): Vars = {step['vars']}")