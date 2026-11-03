"""二维 KDTree：中位数建树，近侧优先搜索，按分割线距离回溯剪枝。

输入有限二维坐标，输出一个精确最近邻的原始索引、坐标和欧氏距离。
仅依赖 Python 3.10+ 标准库；示例运行：python -X utf8 demo.py。
本文件聚焦 1-NN；半径查询、k-NN 和三维实现留作后续练习。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Point2D = tuple[float, float]


@dataclass
class _Node:
    """一个节点保存一个数据点、分割轴以及左右子树。"""

    point: Point2D
    index: int
    axis: int
    left: _Node | None = None
    right: _Node | None = None


@dataclass(frozen=True)
class NearestResult:
    """最近邻结果；index 是输入顺序中的索引，visited_nodes 便于观察剪枝。"""

    point: Point2D
    index: int
    distance: float
    visited_nodes: int


def _as_point_2d(point: Sequence[float]) -> Point2D:
    """统一坐标类型，拒绝维数错误以及会破坏距离比较的 NaN/Inf。"""
    if len(point) != 2:
        raise ValueError("每个点必须恰好包含两个坐标")
    result = (float(point[0]), float(point[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("坐标必须是有限数值")
    return result


class KDTree2D:
    """二维欧氏距离的教学实现：交替按 x/y 轴取中位数，精确搜索 1-NN。

    每层重新排序使建树复杂度为 O(n log² n)。搜索最坏仍为 O(n)。
    并列最近邻选择原始索引最小者；不会原地重排传入的数据。
    """

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        """复制并校验二维点，再建立平衡二叉树；允许空点集。"""
        entries = [(_as_point_2d(point), index) for index, point in enumerate(points)]
        self.root = self._build(entries, depth=0)

    def _build(self, entries: list[tuple[Point2D, int]], depth: int) -> _Node | None:
        """按当前轴排序并取中位点，其余点递归组成两侧子树。"""
        if not entries:
            return None
        axis = depth % 2 # 分割轴
        ordered = sorted(entries, key=lambda entry: (entry[0][axis], entry[1]))
        middle = len(ordered) // 2
        point, index = ordered[middle]
        return _Node(
            point=point,
            index=index,
            axis=axis,
            left=self._build(ordered[:middle], depth + 1),
            right=self._build(ordered[middle + 1 :], depth + 1),
        )

    def nearest(self, query: Sequence[float], *, verbose: bool = False) -> NearestResult | None:
        """近侧优先搜索，必要时回到分叉点搜索远侧，返回精确最近邻。

        Args:
            query: 与数据点使用相同坐标系和单位的二维查询点。
            verbose: 是否打印节点访问、最优候选更新和回溯剪枝过程。

        Returns:
            最近邻结果。空树返回 None；等距时返回原始索引最小的点。

        Raises:
            ValueError: 查询维数不是二，或坐标包含 NaN/Inf。
        """
        target = _as_point_2d(query)
        best_node: _Node | None = None
        best_distance_sq = math.inf
        visited_nodes = 0

        def log(depth: int, message: str) -> None:
            """用缩进显示递归层次，不参与算法判断。"""
            if verbose:
                print("  " * depth + message)

        def search(node: _Node | None, depth: int) -> None:
            """维持已访问点中的最优候选，并用距离下界排除整棵远侧子树。"""
            nonlocal best_node, best_distance_sq, visited_nodes
            if node is None:
                return

            # 步骤 1：当前节点本身也是数据点，不能只检查叶子。
            visited_nodes += 1
            distance_sq = sum((a - b) ** 2 for a, b in zip(target, node.point))
            axis_name = "xy"[node.axis]
            log(depth, f"访问 {node.point}，分割 {axis_name}={node.point[node.axis]:g}，距离²={distance_sq:g}")
            if (
                best_node is None
                or distance_sq < best_distance_sq
                or (distance_sq == best_distance_sq and node.index < best_node.index)
            ):
                best_node, best_distance_sq = node, distance_sq
                log(depth, f"更新当前最佳：{node.point}，r²={best_distance_sq:g}")

            # 步骤 2：先搜索查询点所在的半空间。“近侧”并不保证有最近邻。
            delta = target[node.axis] - node.point[node.axis]
            if delta < 0:
                near_child, far_child = node.left, node.right
            else:
                near_child, far_child = node.right, node.left
            search(near_child, depth + 1)

            if far_child is None:
                return

            # 步骤 3：近侧搜索结束后，使用更新过的 r² 判断远侧是否可能更好。
            # 远侧任意点的欧氏距离² >= 查询点到分割线的距离² = delta²。
            # 保留等号，才能发现远侧可能存在的等距但原始索引更小的点。
            lower_bound_sq = delta * delta
            if lower_bound_sq <= best_distance_sq:
                log(depth, f"回到 {node.point}：下界²={lower_bound_sq:g} <= r²={best_distance_sq:g}，搜索远侧")
                search(far_child, depth + 1)
            else:
                log(depth, f"回到 {node.point}：下界²={lower_bound_sq:g} > r²={best_distance_sq:g}，剪掉远侧")

        search(self.root, depth=0)
        if best_node is None:
            return None
        return NearestResult(best_node.point, best_node.index, math.sqrt(best_distance_sq), visited_nodes)
