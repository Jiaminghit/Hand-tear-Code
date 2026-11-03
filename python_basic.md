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

## 3. 继承与多态

以 `mse&mae&huber/modelization.py` 为例：`class MSELoss(RegressionLoss)` 表示 `MSELoss` **继承** `RegressionLoss`。

- **继承**：子类复用基类的方法，例如 `forward`、`backward` 和 mask 处理。
- **重写**：子类重新定义同名方法，例如 `_elementwise_loss`。
- **多态**：相同调用根据对象的实际类型执行不同实现。基类 `forward` 中的 `self._elementwise_loss(residual)`，在 `self` 是 `MSELoss` 时计算平方误差，是 `MAELoss` 时计算绝对误差。

Python 普通方法也支持重写和多态，无需先声明成 C++ 的 `virtual`。只要对象提供调用方需要的方法，也可以在没有共同基类时统一使用，这通常称为“鸭子类型”。

## 4. `abc`、`ABC` 与 `abstractmethod`

```python
from abc import ABC, abstractmethod
```

| 名称 | 是什么 | 本例的作用 |
|---|---|---|
| `abc` | Python 标准库模块 | 提供抽象基类相关工具 |
| `ABC` | 定义抽象基类的辅助类 | `RegressionLoss` 继承它，启用抽象方法约束 |
| `abstractmethod` | 装饰器 | 将指定方法标记为抽象方法 |

`RegressionLoss` 已实现公共流程，但逐元素公式留给子类。下面是类内部的方法定义：

```python
@abstractmethod
def _elementwise_loss(self, residual: np.ndarray) -> np.ndarray:
    """由具体损失子类提供逐元素公式。"""
    raise NotImplementedError
```

`@abstractmethod` 可以类比 C++ 的纯虚函数：含有尚未实现的抽象方法时，**实例化会抛出 `TypeError`**。子类可以继续保持抽象；要创建可用实例，必须实现或继承所有抽象方法的具体实现。本例要补齐 `_elementwise_loss` 和 `_elementwise_grad`。

两者的作用不要混淆：

- `@abstractmethod` 配合 `ABC`：在创建实例时检查是否还缺少具体实现；不会验证公式是否正确或严格检查方法签名。
- `raise NotImplementedError`：只有执行到这一句才报错。单独写它，不会禁止实例化。

抽象类可以有完整方法和初始化逻辑，抽象方法也可以有方法体。仅继承 `ABC`、但没有未实现的抽象方法，并不会禁止实例化。参见 [Python 抽象基类文档](https://docs.python.org/3/library/abc.html)。

## 5. `@` 装饰器语法

函数也是对象，可以作为参数传给另一个函数。写在函数定义上方的 `@decorator`，表示定义函数后，用装饰器处理它，再将返回结果绑定到原来的函数名。

以下两种写法在理解上近似等价；`decorator` 代表已定义的装饰器：

```python
@decorator
def func() -> None:
    """示意被装饰的函数。"""
    pass
```

```python
def func() -> None:
    """示意原始函数。"""
    pass


func = decorator(func)
```

因此，`@abstractmethod` 相当于用 `abstractmethod` 给方法添加抽象标记。**`@` 是通用语法，具体作用取决于后面的装饰器**，不是所有装饰器都用于抽象类。参见 [Python 装饰器语法](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)。

常见示例：

| 写法 | 作用 |
|---|---|
| `@abstractmethod` | 标记抽象方法 |
| `@property` | 将方法定义成可以通过 `obj.name` 读取的属性 |
| `@staticmethod` | 定义不自动传入 `self` 或 `cls` 的静态方法 |
| `@classmethod` | 定义自动接收类 `cls` 的类方法 |
| `@torch.no_grad()` | 让被装饰函数中的一般张量运算不记录反向计算图 |

`@torch.no_grad()` 多了括号：先调用 `torch.no_grad()` 得到可用的装饰器，再用它装饰函数。可理解为 `func = torch.no_grad()(func)`；关闭梯度记录的效果在调用被装饰函数时生效。参见 [PyTorch no_grad 文档](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)。

如果 `@` 出现在两个表达式中间，例如 `A @ B`，则表示矩阵乘法，与装饰器语法不同。

## 6. 子类的 `__init__` 与 `super()`

`__init__` 是实例初始化方法，常被称为构造函数；对象创建后，Python 用它设置初始状态。`self` 表示正在初始化的实例。

- 子类不定义 `__init__`：沿继承关系查找并使用已有的初始化方法。本例 `MSELoss` 直接继承基类的初始化逻辑。
- 子类定义自己的 `__init__`：会覆盖继承的初始化入口；Python 不会自动再调用基类的 `__init__`。需要复用基类初始化时，显式调用 `super().__init__(...)`。[Python 初始化方法说明](https://docs.python.org/3/reference/datamodel.html#object.__init__)

本例 `HuberLoss` 的初始化流程可概括为：

```python
def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
    """初始化共同的汇总配置，再检查并保存 Huber 阈值。"""
    super().__init__(reduction=reduction)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta 必须为有限的正数")
    self.delta = delta
```

关键语句 `super().__init__(reduction=reduction)` 可以拆成三部分：

1. `super()`：按方法解析顺序（MRO），从当前定义该方法的类之后继续查找；在本例中找到 `RegressionLoss`。
2. `.__init__(...)`：调用找到的初始化方法，在**同一个 HuberLoss 实例**上执行。无需手动传入 `self`，也不会新建一个基类对象。
3. `reduction=reduction`：左边是基类接收的参数名，右边是当前函数中变量的值。

例如 `HuberLoss(delta=2.0, reduction="sum")`：基类先验证 `"sum"`，设置 `self.reduction = "sum"`；随后子类验证 `2.0`，设置 `self.delta = 2.0`。跳过基类初始化，本例就缺少 `self.reduction`，后续前向或反向会失败。

在当前单继承结构中，这行与 `RegressionLoss.__init__(self, reduction=reduction)` 效果相同；通常优先使用 `super()`，避免写死基类名称，并支持协作式多重继承。它更准确的含义是“沿 MRO 继续查找”，不总是“调用直接父类”。参见 [Python 对 super 的说明](https://docs.python.org/3/faq/programming.html#how-do-i-call-a-method-defined-in-a-base-class-from-a-derived-class-that-extends-it)。

签名中的 `delta: float`、`reduction: str` 是类型提示，`= 1.0`、`= "mean"` 是默认值，`-> None` 表示初始化方法不返回业务结果。类型提示本身不会自动转换或校验传入值。

## 7. `from __future__ import annotations` 与前向引用

在本练习使用的 Python 3.11 中，这条语句让当前模块的类型标注以字符串保存。它应放在模块开头，可以写在模块说明字符串之后。

```python
from __future__ import annotations


class Node:
    """树节点的子节点也可以是 Node。"""

    left: Node | None = None
    right: Node | None = None
```

执行 `left: Node | None` 时，`Node` 类还在定义中，这种提前引用类型名称的写法称为**前向引用。** 有了上面的导入，解释器可以先保存标注，等类定义完成后再按需求值。[Python 类型标注的求值规则](https://docs.python.org/3/library/annotationlib.html#annotation-semantics)

## 8. `Sequence` 序列类型标注

`Sequence` 表示支持取长度、按位置索引和遍历的有序序列，列表和元组都符合这个接口。KDTree 的 `Sequence[Sequence[float]]` 中，外层表示点集，内层表示一个点的坐标序列。

```python
from collections.abc import Sequence

points: Sequence[Sequence[float]] = [(2.0, 3.0, 1.0), (5.0, 0.0, 4.0)]
points[0]     # 第一个点 (2.0, 3.0, 1.0)
points[0][2]  # 第一个点的 z 坐标 1.0
```

标注描述元素类型，三维点的长度由 `point3d()` 中的 `len(point) != 3` 显式检查。[Python 序列接口](https://docs.python.org/3.11/library/collections.abc.html#collections.abc.Sequence)

## 9. `@dataclass` 数据类装饰器

`@dataclass` 用于装饰类。它根据带类型标注的字段，默认生成初始化方法 `__init__`、便于打印的 `__repr__` 和按字段比较的 `__eq__`。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class NearestResult:
    """保存最近邻的坐标、原始下标和距离。"""

    point: tuple[float, float, float]
    index: int
    distance: float


result = NearestResult(point=(6.0, 4.0, 0.0), index=5, distance=1.5)
```

普通的 `@dataclass` 允许给 `node.left` 等字段重新赋值。`frozen=True` 限制实例创建后的字段赋值，适合保存一次查询的结果。[Python 数据类文档](https://docs.python.org/3.11/library/dataclasses.html#dataclasses.dataclass)

## 10. 常用内置函数 `enumerate`、`zip`、`sum` 和 `sorted`

| 写法 | 作用 | 示例结果 |
|---|---|---|
| `enumerate(items)` | 遍历时依次给出 `(下标, 元素)`，下标默认从 0 开始 | `list(enumerate(["a", "b"]))` 得到 `[(0, "a"), (1, "b")]` |
| `zip(a, b)` | 将对应位置的元素配成元组，默认到最短输入结束 | `list(zip((1, 2, 3), (4, 5)))` 得到 `[(1, 4), (2, 5)]` |
| `sum(values)` | 累加可迭代对象中的数值 | `sum([1, 4, 9])` 得到 `14` |
| `sorted(items)` | 将输入元素按升序排入新列表，可用 `key` 指定排序键 | `sorted([3, 1, 2])` 得到 `[1, 2, 3]` |

最近邻搜索中，`zip` 对齐同一坐标轴上的值，`sum` 把三项平方差相加。

```python
target, point = (4, 2, 1), (1, 2, 5)
distance_sq = sum((a - b) ** 2 for a, b in zip(target, point))  # 9 + 0 + 16 = 25
```

## 11. `lambda` 表达式与排序键

`lambda 参数: 表达式` 定义一个匿名函数，调用时返回表达式的值。`sorted(..., key=函数)` 会把每条记录传给这个函数，用返回值作为排序键。[Python 排序指南](https://docs.python.org/3.11/howto/sorting.html#key-functions)

```python
entries = [((2, 4, 1), 2), ((1, 3, 5), 1), ((2, 8, 1), 0)]  # (点, 原始下标)
axis = 0  # 按 x 轴划分
ordered = sorted(entries, key=lambda entry: (entry[0][axis], entry[1]))
print([index for point, index in ordered])  # [1, 0, 2]
```

`entry[0][axis]` 取当前轴的坐标，`entry[1]` 取原始下标。上面三条记录的排序键依次为 `(2, 2)`、`(1, 1)`、`(2, 0)`。比较元组时先比第一项，因此 x 为 1 的点排在最前。剩下两个点的 x 同为 2，再比第二项，下标 0 排在下标 2 前面。
