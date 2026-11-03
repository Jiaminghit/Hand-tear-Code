"""运行与 README 一致的七点示例，输出精确 1-NN 的搜索轨迹。

在本目录运行：python -X utf8 demo.py
预期从左子树找到临时候选 (2, 3)，跨过根分割线后找到真正最近邻 (6, 4)。
"""

from kdtree_2d import KDTree2D, Point2D

DEMO_POINTS: list[Point2D] = [
    (5.0, 0.0),
    (3.0, 8.0),
    (2.0, 3.0),
    (1.0, 9.0),
    (7.0, 6.0),
    (6.0, 4.0),
    (9.0, 9.0),
]
DEMO_QUERY: Point2D = (4.5, 4.0)


def main() -> None:
    """建树并打印路径；只有真正计算点距离才计入访问节点数。"""
    print(f"输入点：{DEMO_POINTS}")
    print(f"查询点：{DEMO_QUERY}\n")
    tree = KDTree2D(DEMO_POINTS)
    result = tree.nearest(DEMO_QUERY, verbose=True)
    assert result is not None
    print(f"\n最近点：{result.point}；原始索引：{result.index}；距离：{result.distance:g}")
    print(f"访问节点：{result.visited_nodes}/{len(DEMO_POINTS)}；跳过 (1, 9) 和 (9, 9)")


if __name__ == "__main__":
    main()
