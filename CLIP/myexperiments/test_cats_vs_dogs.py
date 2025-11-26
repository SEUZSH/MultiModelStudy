#!/usr/bin/env python3
"""
CLIP Zero-Shot Classification Test on Cats vs Dogs Dataset
使用预训练CLIP模型对猫狗数据集进行零样本分类测试
"""

import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
import time
import json
from tqdm import tqdm

def load_clip_model(model_name="ViT-B/32"):
    """加载CLIP预训练模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    return model, preprocess, device

def load_dataset(dataset_path="CLIP/myexperiments/data/cats_vs_dogs"):
    """加载cats_vs_dogs数据集"""
    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']
        print(f"数据集加载成功: {len(train_data)} 张图片")
        return train_data
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return None

def predict_single_image(model, preprocess, device, image, text_prompts):
    """对单张图片进行CLIP预测"""
    # 预处理图片
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # 文本token化
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        # 计算特征
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        
        # 计算相似度
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # 获取预测结果
        predicted_class = similarity.argmax().item()
        confidence = similarity[0][predicted_class].item()
        probabilities = similarity[0].cpu().numpy()
    
    return predicted_class, confidence, probabilities

def test_single_sample():
    """测试单个样本"""
    print("=== 单样本测试 ===")
    
    # 加载模型和数据
    model, preprocess, device = load_clip_model()
    dataset = load_dataset()
    
    if dataset is None:
        return
    
    # 文本描述
    text_prompts = ["a photo of a cat", "a photo of a dog"]
    class_names = ["cat", "dog"]
    
    # 测试第一张图片
    sample = dataset[0]
    image = sample['image']
    true_label = sample['labels']  # 0: cat, 1: dog
    
    # 进行预测
    pred_class, confidence, probs = predict_single_image(
        model, preprocess, device, image, text_prompts
    )
    
    # 显示结果
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"真实: {class_names[true_label]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, probs)
    bars[pred_class].set_color('red')
    plt.title(f"预测: {class_names[pred_class]} (置信度: {confidence:.2f})")
    plt.ylabel('概率')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"真实标签: {class_names[true_label]}")
    print(f"预测结果: {class_names[pred_class]}")
    print(f"置信度: {confidence:.3f}")
    print(f"预测{'正确' if pred_class == true_label else '错误'}")

def batch_test(num_samples=100):
    """批量测试"""
    print(f"=== 批量测试 ({num_samples} 样本) ===")
    
    # 加载模型和数据
    model, preprocess, device = load_clip_model()
    dataset = load_dataset()
    
    if dataset is None:
        return
    
    # 文本描述
    text_prompts = ["a photo of a cat", "a photo of a dog"]
    class_names = ["cat", "dog"]
    
    # 随机选择测试样本
    test_samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    correct_predictions = 0
    total_samples = len(test_samples)
    results = []
    
    print(f"开始测试 {total_samples} 张图片...")
    
    for i, sample in enumerate(tqdm(test_samples, desc="测试进度")):
        image = sample['image']
        true_label = sample['labels']
        
        # 进行预测
        pred_class, confidence, _ = predict_single_image(
            model, preprocess, device, image, text_prompts
        )
        
        # 记录结果
        is_correct = pred_class == true_label
        correct_predictions += is_correct
        
        results.append({
            'index': i,
            'true_label': true_label,
            'predicted_label': pred_class,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # 计算准确率
    accuracy = correct_predictions / total_samples
    
    print(f"\n=== 测试结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"正确预测: {correct_predictions}")
    print(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # 分析错误案例
    errors = [r for r in results if not r['correct']]
    print(f"错误案例数: {len(errors)}")
    
    if errors:
        print(f"错误案例置信度范围: {min(e['confidence'] for e in errors):.3f} - {max(e['confidence'] for e in errors):.3f}")
    
    # 保存结果
    save_results(results, accuracy)
    
    return results, accuracy

def save_results(results, accuracy, filename="CLIP/myexperiments/results/test_results.json"):
    """保存测试结果"""
    output = {
        'total_samples': len(results),
        'accuracy': accuracy,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {filename}")

def main():
    """主函数"""
    print("CLIP Cats vs Dogs 零样本分类测试")
    print("=" * 50)
    
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 单样本测试")
    print("2. 批量测试 (100样本)")
    print("3. 批量测试 (1000样本)")
    print("4. 自定义批量测试")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        test_single_sample()
    elif choice == "2":
        batch_test(100)
    elif choice == "3":
        batch_test(1000)
    elif choice == "4":
        try:
            num = int(input("请输入测试样本数: "))
            batch_test(num)
        except ValueError:
            print("输入无效，使用默认值100")
            batch_test(100)
    else:
        print("选择无效，运行单样本测试")
        test_single_sample()

if __name__ == "__main__":
    main()
