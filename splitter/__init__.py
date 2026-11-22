"""训练数据湖可重复数据集划分工具。

模块划分：
- dataset:  样本/关系数据模型、CSV 读写、版本哈希
- graph:    主体关系图与连通组（并查集）
- timesplit: 时间边界与组级时间资格
- allocate: 以连通组为单位的贪心/小规模穷举分配
- manifest: 冻结配置、清单生成、偏差解释报告
- revision: 关系撤回/合并、泄漏分析、版本对比
- store:    基于文件的不可变快照存储
- examples: 大组、三层间接关系、迟到修订示例
- api/server: HTTP 接口与静态页面
"""

__version__ = "1.0.0"
