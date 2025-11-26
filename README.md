# MultiModelStudy
Trying to learn Large-Multimodal-Models.

## 小红书白连衣裙 vs 白婚纱分类测试 2025.11.25 23:29:05

- 数据集：12张真实小红书图片（爬取于2025.11.25）
- 模型：CLIP ViT-B/32 Zero-Shot
- 结果：100%准确率
- 发现：模型对"白色"敏感，但对"婚纱头纱"等细节依赖视觉patch的局部attention

## CLIP模型规模化验证 2025.11.25 23:03:47
- 在Cats vs Dogs标准数据集（23,410张图）上完成Zero-Shot分类，准确率99.7%
- 独立搭建批量推理pipeline，工程验证CLIP在大规模数据下的稳定性与泛化能力
- 工程代码与结果已开源至GitHub