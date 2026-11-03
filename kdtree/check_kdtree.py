"""用独立的暴力枚举校验二维 KDTree，覆盖回溯、并列、重复及随机数据。

在本目录运行：python -X utf8 check_kdtree.py
仅依赖标准库。这里的基准使用 math.dist 枚举所有点，不复用树的搜索逻辑。
"""

import math
import random

from demo import DEMO_POINTS, DEMO_QUERY
from kdtree_2d import KDTree2D, Point2D


def check_case(points: list[Point2D], queries: list[Point2D]) -> int:
    """对比最近点原始索引与距离，确认建树不修改输入列表。"""
    original = points.copy()
    tree = KDTree2D(points)
    assert points == original
    for query in queries:
        actual = tree.nearest(query)
        if not points:
            assert actual is None
            continue
        distances = [math.dist(query, point) for point in points]
        expected_index = min(range(len(points)), key=lambda index: (distances[index], index))
        assert actual is not None
        assert actual.index == expected_index, (points, query, actual, expected_index)
        assert actual.point == points[expected_index]
        assert math.isclose(actual.distance, distances[expected_index], rel_tol=1e-12, abs_tol=1e-12)
        assert 1 <= actual.visited_nodes <= len(points)
    return len(queries)


def main() -> None:
    """运行固定边界样例与固定随机种子的多组查询，失败时直接抛出断言。"""
    cases: list[tuple[list[Point2D], list[Point2D]]] = [
        ([], [(0.0, 0.0)]),
        ([(2.0, 3.0)], [(2.0, 3.0), (100.0, -100.0)]),
        (DEMO_POINTS, [DEMO_QUERY, (5.0, 4.0), (6.0, 4.0), (-100.0, 100.0)]),
        ([(1.0, 1.0)] * 5, [(1.0, 1.0), (2.0, 3.0)]),
        ([(0.0, 0.0), (0.0, 2.0), (0.0, 4.0)], [(0.0, 1.0), (0.0, 3.0)]),
        ([(-1.0, 0.0), (1.0, 0.0)], [(0.0, 0.0)]),
        # 根为 (1,5)，近侧 (-1,0) 的索引为 2；h=r=1 时仍须查远侧的索引 1。
        ([(1.0, 5.0), (1.0, 0.0), (-1.0, 0.0)], [(0.0, 0.0)]),
        ([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)], [(0.0, 0.0)]),
    ]
    checked = sum(check_case(points, queries) for points, queries in cases)

    rng = random.Random(20261103)
    for count in (1, 2, 3, 7, 32, 100):
        for _ in range(10):
            points = [(float(rng.randrange(-20, 21)), float(rng.randrange(-20, 21))) for _ in range(count)]
            # 半整数坐标使距离平方可以精确表示，避免近似相等干扰并列规则验证。
            queries = [(rng.randrange(-60, 61) / 2.0, rng.randrange(-60, 61) / 2.0) for _ in range(50)]
            checked += check_case(points, queries)

    # 再覆盖非整数、非半整数坐标，验证通常的浮点输入。
    points = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(100)]
    queries = [(rng.uniform(-150, 150), rng.uniform(-150, 150)) for _ in range(100)]
    checked += check_case(points, queries)

    for invalid in ((1.0,), (1.0, 2.0, 3.0), (math.nan, 0.0), (0.0, math.inf)):
        try:
            KDTree2D([invalid])
        except ValueError:
            pass
        else:
            raise AssertionError(f"建树未拒绝非法坐标：{invalid}")
        try:
            KDTree2D(DEMO_POINTS).nearest(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"查询未拒绝非法坐标：{invalid}")

    print(f"通过：{checked} 次查询与暴力枚举一致；维数与 NaN/Inf 校验通过。")


if __name__ == "__main__":
    main()
