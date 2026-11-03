"""绘制二维示例的分割线和候选搜索圆，输出到本目录 search_geometry.png。

可选依赖：Matplotlib；核心 KDTree 和验证脚本不依赖此库。
运行：python -X utf8 plot_search.py。图中文字使用英文以避免系统中文字体差异。
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from demo import DEMO_POINTS, DEMO_QUERY


def main() -> None:
    """展示必须跨根分割线回溯，以及最终可以剪枝的两个区域。"""
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis in axes:
        axis.set(xlim=(-0.5, 10), ylim=(-1, 10), xlabel="x", ylabel="y", aspect="equal")
        axis.grid(alpha=0.12)
        axis.axvline(5, color="#334155", linewidth=1.8)
        axis.plot([-0.5, 5], [8, 8], color="#64748b", linestyle="--")
        axis.plot([5, 10], [6, 6], color="#64748b", linestyle="--")
        axis.text(5.12, -0.65, "root: x = 5", color="#334155")
        axis.text(-0.25, 8.15, "y = 8", color="#64748b")
        axis.text(8.7, 6.15, "y = 6", color="#64748b")
        for x, y in DEMO_POINTS:
            axis.scatter(x, y, s=40, color="#475569", zorder=3)
            axis.annotate(f"({x:g}, {y:g})", (x, y), xytext=(5, 7), textcoords="offset points", fontsize=9)
        axis.scatter(*DEMO_QUERY, s=120, marker="*", color="#2563eb", zorder=5)
        axis.annotate("q = (4.5, 4)", DEMO_QUERY, xytext=(-5, -22), textcoords="offset points", color="#2563eb")

    left, right = axes
    left.set_title("1. Best in left subtree: (2, 3)\nThe search circle crosses x = 5", pad=12)
    left.add_patch(Circle(DEMO_QUERY, 7.25**0.5, fill=False, color="#d97706", linewidth=2))
    left.plot([4.5, 2], [4, 3], color="#d97706", linewidth=1.5)
    left.scatter(2, 3, s=90, color="#d97706", zorder=4)
    left.plot([4.5, 5], [4, 4], color="#2563eb", linewidth=3)
    left.text(4.1, 4.4, "h = 0.5", color="#2563eb")
    left.text(1.4, 1.6, "r = sqrt(7.25) = 2.693", color="#b45309")
    left.add_patch(Rectangle((-0.5, 8), 5.5, 2, color="#94a3b8", alpha=0.12))

    right.set_title("2. Search right subtree: best becomes (6, 4)\nAt y = 6: h = 2 > r = 1.5, prune above", pad=12)
    right.add_patch(Circle(DEMO_QUERY, 1.5, fill=False, color="#059669", linewidth=2))
    right.plot([4.5, 6], [4, 4], color="#059669", linewidth=1.5)
    right.scatter(6, 4, s=90, color="#059669", zorder=4)
    right.text(5.1, 4.25, "r = 1.5", color="#047857")
    right.plot([4.5, 4.5], [4, 6], color="#2563eb", linestyle=":", linewidth=2)
    right.plot([4.5, 5], [6, 6], color="#64748b", linestyle=":", linewidth=1)
    right.text(3.3, 5.65, "h = 2", color="#2563eb")
    right.add_patch(Rectangle((-0.5, 8), 5.5, 2, color="#94a3b8", alpha=0.12))
    right.add_patch(Rectangle((5, 6), 5, 4, color="#94a3b8", alpha=0.12))
    right.scatter([1, 9], [9, 9], s=100, marker="x", color="#b91c1c", zorder=5)
    right.text(6.1, 7.2, "Pruned: no point-distance\nevaluation in this region", fontsize=9, color="#64748b")
    output = Path(__file__).with_name("search_geometry.png")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
