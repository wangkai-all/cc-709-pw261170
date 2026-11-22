# 新旧版本对比

- 旧版 `gen_82fb9d01bdc1`（relations-v1）
- 新版 `gen_11f5ed854109`（relations-v2）
- 跨集合移动样本数: **6**

| 集合 | 目标 | 旧实际 | 新实际 |
|---|---|---|---|
| train | 70.00% | 60.00% (9) | 75.00% (9) |
| val | 15.00% | 20.00% (3) | 0.00% (0) |
| test | 15.00% | 20.00% (3) | 25.00% (3) |

## 移动明细

| sample_id | from | to |
|---|---|---|
| s_S01_1 | test | quarantine |
| s_S01_2 | test | quarantine |
| s_S01_3 | test | quarantine |
| s_S04_1 | val | test |
| s_S04_2 | val | test |
| s_S04_3 | val | test |
