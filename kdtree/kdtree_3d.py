from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass

Point3D = tuple[float, float, float]

@dataclass
class Node:
    point : Point3D
    index : int
    axis : int
    left : Node | None = None
    right : Node | None = None

@dataclass(frozen=True)
class NearestResult:
    point : Point3D
    index : int
    distance: float

def point3d(point: Sequence[float]) -> Point3D:
    if len(point) != 3:
        raise ValueError("每个点必须恰好包含3个坐标")
    result = (float(point[0]), float(point[1]), float(point[2]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("坐标必须是有限数值")
    return result

class KDTree3D:
    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        entries = [(point3d(point), index) for index, point in enumerate(points)]
        self.root = self.build(entries, depth=0)
        
    def build(self, entries: list[tuple[Point3D, int]], depth: int) -> Node | None :
        if not entries:
            return None
        axis = depth % 3
        ordered = sorted(entries, key=lambda entry: (entry[0][axis], entry[1]))
        middle = len(ordered) // 2
        point, index = ordered[middle]
        return Node(
            point=point,
            index=index,
            axis=axis,
            left=self.build(ordered[:middle], depth + 1),
            right=self.build(ordered[middle + 1 :], depth + 1),
        )
    
    def nearest(self, query: Sequence[float]) -> NearestResult | None :
        target = point3d(query)
        best_node : Node | None = None
        best_distance_sq = math.inf
        
        def search(node: Node | None, depth: int) -> None:
            nonlocal best_node, best_distance_sq
            if node is None:
                return
            
            # step 1 : 向下搜索
            # 计算当前节点与查询点的距离 并 判断更新最优距离
            distance_sq = sum((a - b) ** 2 for a, b in zip(target, node.point))
            if (
                best_node is None
                or distance_sq < best_distance_sq
            ):
                best_node, best_distance_sq = node, distance_sq
            # 向下搜索（query所在侧）
            delta = target[node.axis] - node.point[node.axis]
            if delta < 0:
                near_child, far_child = node.left, node.right
            else:
                near_child, far_child = node.right, node.left
            search(near_child, depth= depth + 1)
            
            if far_child is None:
                return
            
            # 回溯与剪枝
            if delta * delta <= best_distance_sq:
                search(far_child, depth= depth + 1)
            
        search(self.root, depth= 0)
        
        if best_node is None:
            return None
        return NearestResult(point=best_node.point, 
                             index=best_node.index,
                             distance=math.sqrt(best_distance_sq))