"""临时评测脚本：扫不同向量/BM25 权重，对比 10 道题的 top3 命中率。跑完即删。"""
import vector_store as vs

# (题目, 期望命中的文档标题)
CASES = [
    # --- Python 笔记（doc8）---
    ("Python 文件直接运行时 __name__ 变量的值是什么", "Python疑问汇总_2026-07-21.md"),      # 关键词直问
    ("Python 里怎么跳过这一次循环不执行", "Python疑问汇总_2026-07-21.md"),                    # 口语换说法（continue）
    ("写函数时怎么判断要不要给它加参数", "Python疑问汇总_2026-07-21.md"),                      # 口语换说法
    ("一边遍历列表一边删除元素该怎么做", "Python疑问汇总_2026-07-21.md"),                      # 口语换说法（enumerate/pop）
    # --- 算法复习题（doc6）---
    ("回溯法使用的搜索策略是深度优先还是广度优先", "复习题(1)(1)(1).pdf"),                      # 关键词直问
    ("以下哪个序列不是小根堆", "复习题(1)(1)(1).pdf"),                                          # 关键词直问
    ("时间复杂度 O(n平方) 的算法输入规模扩大100倍执行时间增加多少倍", "复习题(1)(1)(1).pdf"),   # 口语换说法
    # --- 民族共同体（doc7）---
    ("土司制度在哪个朝代达到全盛", "中华民族共同体概论 复习资料.docx"),                        # 口语换说法
    ("辽亡后耶律大石建立的是什么", "中华民族共同体概论 复习资料.docx"),                        # 关键词直问（人名）
    ("马家窑文化马厂类型的彩陶豆源自什么时期", "中华民族共同体概论 复习资料.docx"),            # 关键词直问
]

CONFIGS = [
    (1.0, 0.0, "纯向量"),
    (0.7, 0.3, "向量偏重"),
    (0.6, 0.4, "当前默认"),
    (0.5, 0.5, "平权"),
    (0.4, 0.6, "BM25偏重"),
    (0.0, 1.0, "纯BM25"),
]

print(f"{'配置':<8} {'命中数/10':<8} 平均最佳排名")
for vec_w, bm_w, name in CONFIGS:
    vs.VECTOR_WEIGHT = vec_w
    vs.BM25_WEIGHT = bm_w
    hits = 0
    rank_sum = 0
    misses = []
    for q, expect in CASES:
        results = vs.search_chunks(q, n_results=3)
        titles = [r["title"] for r in results]
        if expect in titles:
            hits += 1
            rank_sum += titles.index(expect) + 1
        else:
            misses.append(q[:14])
    avg_rank = f"{rank_sum / hits:.1f}" if hits else "-"
    print(f"{name:<8} {hits}/10     {avg_rank:<8} 未命中: {misses if misses else '无'}")
