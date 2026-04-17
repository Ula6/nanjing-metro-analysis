import pandas as pd
import requests
import json
import time
import os
import chardet
from tqdm import tqdm
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 配置路径
input_csv = r"D:\weibo-search\结果文件\南京地铁\text.csv"
output_csv = r"D:\weibo-search\结果文件\南京地铁\classified_results.csv"
ground_truth_csv = r"D:\weibo-search\结果文件\南京地铁\ground_truth.csv"  # 预设结果文件
log_file = r"D:\weibo-search\结果文件\南京地铁\processing_log.txt"

# 初始化OpenAI客户端
load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)

# 分类体系
CATEGORIES = ["基础层", "保障层", "舒适层", "尊重层", "自我实现层", "其他"]


def robust_read_csv(file_path, max_sample_size=100000):
    """
    增强版CSV读取函数，支持多种编码和错误恢复
    """
    # 尝试的编码列表（按优先级排序）
    encodings = ['utf-8-sig', 'gb18030', 'gbk', 'utf-8', 'latin1']

    # 首先尝试检测文件编码
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(min(os.path.getsize(file_path), max_sample_size))
            result = chardet.detect(rawdata)
            if result['confidence'] > 0.9:
                encodings.insert(0, result['encoding'])
    except:
        pass

    # 尝试各种编码
    last_exception = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if not df.empty and '微博正文' in df.columns:
                print(f"成功用 {encoding} 编码读取文件")
                return df

            # 如果列名有编码问题，尝试修复
            if df.empty or '微博正文' not in df.columns:
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = [col.encode('latin1').decode('gb18030', errors='ignore')
                              if isinstance(col, str) else col for col in df.columns]
                if '微博正文' in df.columns:
                    return df
        except Exception as e:
            last_exception = e
            continue

    # 如果所有编码都失败，尝试错误恢复模式
    try:
        print("尝试错误恢复模式读取")
        with open(file_path, 'rb') as f:
            content = f.read()
            for enc in ['gb18030', 'gbk', 'utf-8']:
                try:
                    decoded = content.decode(enc, errors='replace')
                    df = pd.read_csv(StringIO(decoded))
                    if '微博正文' in df.columns:
                        return df
                except:
                    continue

        # 最后尝试latin1编码
        df = pd.read_csv(file_path, encoding='latin1')
        # 尝试修复中文列名
        df.columns = [col.encode('latin1').decode('gb18030', errors='ignore')
                      if isinstance(col, str) else col for col in df.columns]
        if '微博正文' in df.columns:
            return df

        # 尝试修复内容
        for col in df.select_dtypes(include=['object']):
            try:
                df[col] = df[col].str.encode('latin1').str.decode('gb18030', errors='ignore')
            except:
                pass
        return df
    except Exception as e:
        last_exception = e

    raise ValueError(f"无法读取文件 {file_path}，所有编码尝试均失败。最后错误: {str(last_exception)}")

def call_openai_api(text):
    """使用OpenAI API进行分类"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文本分类器"},
                {"role": "user", "content": f"""
                请严格判断微博内容是否反映用户需求，参考定义和示例辅助理解：

                【层次定义与示例参考】
                仅当内容反映乘客实际需求时，选择对应层级：
                1. 基础层（安全与时效）：
                - 定义：保障乘客基本出行安全和时间可靠性的需求
                - 示例参考：安检问题、设备故障、运行延误、急刹等

                2. 保障层（基本生存资料供给）：
                - 定义：确保出行基本条件满足的需求  
                - 示例参考：设施便利性、电梯扶梯、线路可达性、支付保障、乘客诉求、动线复杂等

                3. 舒适层（物理环境和生理舒适度）：
                - 定义：提升乘车物理环境舒适度的需求
                - 示例参考：环境卫生、拥挤度、空调、冷热、座位等

                4. 尊重层（人性服务）：
                - 定义：获得尊重和人性化服务的需求，强调人与人之间的尊重与关怀
                - 示例参考：工作人员、服务态度、特殊关怀、隐私保护、乘客间互动等

                5. 自我实现层（情感共鸣与价值归属）：
                - 定义：超越交通功能，严格反映乘客对南京地铁的认同，有正面倾向
                - 示例参考：文化体验、归属感、城市认同、节日、毕业季、超话等

                【分类原则】
                1. 核心判断依据是层次定义而非具体示例
                2. 示例仅用于辅助理解层次内涵
                3. 当内容反映的需求符合某层次定义时即归类，不符合上述五类的归为"其他"
                4. 不要求出现示例中的具体关键词
                5. 模糊或未反映乘客情感需求的中性表述一律归为"其他"
                6. 必须归为"其他"的情况：
                   - 官方通告/新闻报道
                   - 政策法规说明
                   - 客观事实陈述
                   - 营销宣传内容
                   - 与地铁服务中用户需求无关的内容

                【输出格式】
                直接返回最符合的层次名称：
                基础层/保障层/舒适层/尊重层/自我实现层/其他

                待分析内容："{text[:500]}"
                """}
            ],
            temperature=0.3,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        return result if result in CATEGORIES else "其他"
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return "其他"

# 保留 evaluate_classification 和 process_csv 函数不变
def evaluate_classification(true_labels, predicted_labels):
    """
    自主实现的分类评价函数（不依赖scikit-learn）
    """
    # 获取所有类别
    unique_labels = sorted(set(true_labels + predicted_labels))
    num_classes = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # 初始化统计
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    class_stats = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in unique_labels}

    # 计算混淆矩阵和各类统计
    for true, pred in zip(true_labels, predicted_labels):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        confusion_matrix[true_idx][pred_idx] += 1

        if true == pred:
            class_stats[true]['tp'] += 1
        else:
            class_stats[true]['fn'] += 1
            class_stats[pred]['fp'] += 1

    # 计算指标
    metrics = {
        'accuracy': sum(stats['tp'] for stats in class_stats.values()) / len(true_labels),
        'class_metrics': {},
        'confusion_matrix': confusion_matrix
    }

    for label in unique_labels:
        tp = class_stats[label]['tp']
        fp = class_stats[label]['fp']
        fn = class_stats[label]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics['class_metrics'][label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': tp + fn
        }

    # 打印报告
    print("\n=== 分类性能报告 ===")
    print(f"{'类别':<10}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'支持数':<10}")
    for label in unique_labels:
        stats = metrics['class_metrics'][label]
        print(f"{label:<10}{stats['precision']:<10.3f}"
              f"{stats['recall']:<10.3f}"
              f"{stats['f1']:<10.3f}"
              f"{stats['support']:<10}")

    print(f"\n总准确率: {metrics['accuracy']:.4f}")

    # 设置全局字体（解决不匹配问题）
    plt.rcParams['font.family'] = 'SimHei'  # 中文用黑体
    plt.rcParams['font.size'] = 10  # 统一基础字号

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        linewidths=0.5,
        annot_kws={'size': 10}  # 标注字号与全局一致
    )

    # 设置标题和标签（自动继承全局字体设置）
    plt.title('混淆矩阵 - 真实 vs 预测')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 调整刻度标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 保存图像
    plt.tight_layout()
    confusion_matrix_path = os.path.join(os.path.dirname(output_csv), 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    return metrics


def process_csv(input_path, output_path, ground_truth_path=None):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 初始化日志
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"分类处理开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 读取数据（使用增强版函数）
        print("正在尝试读取CSV文件...")
        df = robust_read_csv(input_path)
        print(f"成功读取 {len(df)} 条记录")
        df['微博正文'] = df['微博正文'].fillna('').astype(str)

        # 读取预设结果（如果有）
        true_labels = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                gt_df = robust_read_csv(ground_truth_path)
                if '需求层次' in gt_df.columns:
                    true_labels = gt_df['需求层次'].tolist()
                    print(f"已加载 {len(true_labels)} 条预设分类结果")
            except Exception as e:
                print(f"读取预设结果失败: {str(e)}")

        # 初始化统计
        stats = {cat: 0 for cat in CATEGORIES}
        results = []

        # 处理数据
        print("开始需求层次分类...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row['微博正文'].strip()
            if not text:
                results.append("其他")
                continue

            category = call_openai_api(text)
            results.append(category)
            stats[category] += 1

            # 每1000条保存进度
            if (idx + 1) % 1000 == 0:
                temp_df = pd.DataFrame({
                    '微博原文': df['微博正文'].iloc[:idx + 1],
                    '需求层次': results
                })
                temp_df.to_csv(output_path.replace('.csv', '_temp.csv'),
                               index=False, encoding='utf_8_sig')

            time.sleep(1.2)  # API限速

        # 保存最终结果
        result_df = pd.DataFrame({
            '微博原文': df['微博正文'],
            '需求层次': results
        })
        result_df.to_csv(output_path, index=False, encoding='utf_8_sig')

        # 生成统计报告 - 修改后的部分
        stats_df = pd.DataFrame({
            '需求层次': CATEGORIES,
            '数量': [stats[cat] for cat in CATEGORIES],
            '占比': [stats[cat] / len(df) for cat in CATEGORIES]
        })

        # 排序并格式化
        stats_df = stats_df.sort_values('数量', ascending=False)
        stats_df['占比'] = stats_df['占比'].apply(lambda x: f"{x:.2%}")

        # 确保列名是简体中文
        stats_df.columns = ['需求层次', '数量', '占比']

        # 保存统计结果
        stats_df.to_csv(output_path.replace('.csv', '_stats.csv'),
                        index=False,
                        encoding='utf_8_sig')

        # 如果有预设结果，进行模型评价
        if true_labels and len(true_labels) == len(results):
            print("\n=== 开始模型评价 ===")
            evaluation = evaluate_classification(true_labels, results)

            # 保存评价结果
            eval_results = []
            for label, metrics in evaluation['class_metrics'].items():
                eval_results.append({
                    '类别': label,
                    '精确率': metrics['precision'],
                    '召回率': metrics['recall'],
                    'F1分数': metrics['f1'],
                    '支持数': metrics['support']
                })
            eval_results.append({
                '类别': '总体',
                '精确率': '',
                '召回率': '',
                'F1分数': '',
                '支持数': len(results),
                '准确率': evaluation['accuracy']
            })

            eval_df = pd.DataFrame(eval_results)
            eval_path = output_path.replace('.csv', '_evaluation.csv')
            eval_df.to_csv(eval_path, index=False, encoding='utf_8_sig')
            print(f"评价结果已保存至: {eval_path}")

        # 记录完成日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理完成 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总计处理: {len(df)} 条\n")
            f.write("分类统计:\n" + stats_df.to_string(index=False))
            if true_labels:
                f.write("\n\n模型评价结果:\n")
                f.write(eval_df.to_string(index=False))

        print(f"\n处理完成！结果已保存至: {output_path}")
        print("分类统计:")
        print(stats_df.to_string(index=False))

    except Exception as e:
        print(f"处理失败: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        process_csv(input_csv, output_csv, ground_truth_csv)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n用户中断操作")
    except Exception as e:
        print(f"\n程序错误: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n程序错误: {str(e)}")