# Python 基础笔记

记录编程中遇到的 Python 知识，按主题逐步补充。

## 1. Python 代码的执行方式

以下是三种常用方式。命令以 PowerShell 为例，当前目录为项目根目录 `F:\github\手撕`。

| 方式 | 命令 | 说明 |
|---|---|---|
| 脚本方式 | `python .\mce\mce.py` | 按文件路径执行 `.py` 文件 |
| 模块方式 | `python -m mce.mce` | 按模块名查找并执行；点号表示包层级，不带 `.py` 后缀 |
| 命令行代码方式 | `python -c "print(1 + 2)"` | 直接执行引号中的代码，无需创建文件；本例输出 `3` |

### 1.1 脚本方式与模块方式的导入区别

默认情况下，Python 会将以下目录放到模块搜索路径 `sys.path` 的开头：

- 脚本方式：脚本所在目录，本例为 `F:\github\手撕\mce`。
- 模块方式：当前工作目录，本例为 `F:\github\手撕`。
- `-c` 方式：用空字符串表示当前工作目录。

因此，当 `mce.py` 需要导入同级目录中的 `softmax/softmax.py` 时：

```python
from softmax.softmax import softmax
```

在项目根目录运行 `python -m mce.mce`，便能从根目录找到 `softmax`。直接运行 `python .\mce\mce.py` 时，根目录不会仅因为它是当前工作目录就自动加入搜索路径，可能报 `ModuleNotFoundError`。

## 2. `if __name__ == "__main__":` 的作用

Python 会从上到下执行文件的顶层代码；首次通过 `import` 加载模块时，也会执行这些代码。因此，这个判断不是必需的启动语句，而是用来让某段代码仅在文件作为程序入口运行时执行。

| 使用方式 | 该文件中的 `__name__` | 是否执行判断内的代码 |
|---|---|---|
| `python .\mce\mce.py` | `"__main__"` | 是 |
| `python -m mce.mce` | `"__main__"` | 是 |
| `from mce.mce import cross_entropy` | `"mce.mce"` | 否 |

### 2.1 使用方法

可复用的函数定义放在判断外，测试调用、打印结果或程序入口调用放在判断内：

```python
def add(a: int, b: int) -> int:
    """返回两个整数的和。"""
    return a + b


if __name__ == "__main__":
    # 作为程序入口运行时执行测试；被其他模块导入时跳过。
    print(add(1, 2))
```

没有这个判断，脚本仍然可以运行；但放在顶层的测试和打印也会在导入时执行。只有函数、类等定义的模块，可以不写这个判断。
