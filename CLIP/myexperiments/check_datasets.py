# ======== 检查白色连衣裙数据集标签分配 ========
from datasets import load_from_disk
import matplotlib.pyplot as plt

def check_white_dress_labels():
    """检查white_dress数据集的标签分配"""
    
    print("=== 检查白色连衣裙数据集标签 ===")
    
    try:
        # 加载数据集
        dataset = load_from_disk('CLIP/myexperiments/data/white_dress')
        train_data = dataset['train']
        
        print(f"✅ 数据集加载成功")
        print(f"📊 总样本数: {len(train_data)}")
        
        # 检查标签分布
        labels = train_data['labels']
        filenames = train_data['filename']
        categories = train_data['category']
        
        unique_labels = set(labels)
        print(f"🏷️ 唯一标签: {sorted(unique_labels)}")
        
        # 统计各类别数量
        label_counts = {}
        category_counts = {}
        
        for i, label in enumerate(labels):
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            
            cat = categories[i]
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1
        
        print(f"\n📈 标签统计:")
        for label, count in sorted(label_counts.items()):
            label_name = "white_dress" if label == 0 else "white_wedding_dress" if label == 1 else f"unknown({label})"
            print(f"   标签 {label} ({label_name}): {count} 个")
        
        print(f"\n📂 类别统计:")
        for cat, count in category_counts.items():
            print(f"   {cat}: {count} 个")
        
        # 检查前几个样本的详细信息
        print(f"\n🔍 前6个样本详细信息:")
        for i in range(min(6, len(train_data))):
            filename = filenames[i]
            label = labels[i]
            category = categories[i]
            
            expected_label = 0 if 'dress' in filename.lower() else 1 if 'wedding' in filename.lower() else -1
            
            label_name = "white_dress" if label == 0 else "white_wedding_dress" if label == 1 else f"unknown({label})"
            expected_name = "white_dress" if expected_label == 0 else "white_wedding_dress" if expected_label == 1 else "unknown"
            
            status = "✅" if label == expected_label else "❌"
            match_info = f"{expected_name}={expected_label}" if label == expected_label else f"{expected_name}={expected_label} → {label_name}={label}"
            
            print(f"   {i+1}. {filename}")
            print(f"      文件名暗示: {match_info}")
            print(f"      实际标签: {status}")
        
        # 验证标签一致性
        print(f"\n🔧 标签一致性检查:")
        
        dress_files = [f for f in filenames if 'dress' in f.lower()]
        wedding_files = [f for f in filenames if 'wedding' in f.lower()]
        
        dress_labels = [labels[i] for i, f in enumerate(filenames) if 'dress' in f.lower()]
        wedding_labels = [labels[i] for i, f in enumerate(filenames) if 'wedding' in f.lower()]
        
        dress_correct = all(label == 0 for label in dress_labels)
        wedding_correct = all(label == 1 for label in wedding_labels)
        
        print(f"   连衣裙文件数: {len(dress_files)}")
        print(f"   连衣裙标签正确: {dress_correct} ({'全部正确' if dress_correct else '存在问题'})")
        print(f"   婚纱文件数: {len(wedding_files)}")
        print(f"   婚纱标签正确: {wedding_correct} ({'全部正确' if wedding_correct else '存在问题'})")
        
        # 总体评估
        all_correct = dress_correct and wedding_correct
        print(f"\n🎯 总体评估: {'✅ 标签分配完全正确' if all_correct else '❌ 标签分配存在问题'}")
        
        if not all_correct:
            print(f"\n⚠️ 发现问题:")
            if not dress_correct:
                wrong_dress_indices = [i for i, f in enumerate(filenames) if 'dress' in f.lower() and labels[i] != 0]
                print(f"   连衣裙标签错误: {[filenames[i] for i in wrong_dress_indices[:3]]} (显示前3个)")
            
            if not wedding_correct:
                wrong_wedding_indices = [i for i, f in enumerate(filenames) if 'wedding' in f.lower() and labels[i] != 1]
                print(f"   婚纱标签错误: {[filenames[i] for i in wrong_wedding_indices[:3]]} (显示前3个)")
        
        # 可视化标签分布
        plt.figure(figsize=(10, 6))
        
        # 标签分布
        plt.subplot(1, 2, 1)
        label_names = ['white_dress' if l == 0 else 'white_wedding_dress' for l in sorted(unique_labels)]
        label_counts_list = [label_counts[l] for l in sorted(unique_labels)]
        
        bars = plt.bar(label_names, label_counts_list, color=['lightblue', 'lightpink'])
        plt.title('Label Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 在柱子上显示数值
        for i, (name, count) in enumerate(zip(label_names, label_counts_list)):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 样本分布
        plt.subplot(1, 2, 2)
        cat_names = list(category_counts.keys())
        cat_counts_list = list(category_counts.values())
        
        bars = plt.bar(cat_names, cat_counts_list, color=['lightblue', 'lightpink'])
        plt.title('Category Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 在柱子上显示数值
        for i, (name, count) in enumerate(zip(cat_names, cat_counts_list)):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.suptitle('White Dress Dataset Label Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return all_correct
        
    except Exception as e:
        print(f"❌ 数据集检查失败: {e}")
        return False

# 运行检查
if __name__ == "__main__":
    is_correct = check_white_dress_labels()
    
    print(f"\n📋 检查总结:")
    print(f"   数据集路径: CLIP/myexperiments/data/white_dress")
    print(f"   标签状态: {'✅ 正确' if is_correct else '❌ 需要修复'}")
    
    if not is_correct:
        print(f"\n💡 建议:")
        print(f"   1. 重新运行数据集创建脚本")
        print(f"   2. 确保文件夹名称正确映射到标签")
        print(f"   3. 检查图片文件命名规范")