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
from dotenv import load_dotenv

# 配置路径
input_csv = r"D:\weibo-search\结果文件\南京地铁\二版.csv"
output_csv = r"D:\weibo-search\结果文件\南京地铁\sentiment_results.csv"  # 修改输出文件名
ground_truth_csv = r"D:\weibo-search\结果文件\南京地铁\sentiment_ground_truth.csv"  # 修改预设结果文件
log_file = r"D:\weibo-search\结果文件\南京地铁\sentiment_log.txt"  # 修改日志文件名

# DeepSeek API配置
load_dotenv()
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "DEEPSEEK_API_KEY"  # 替换为实际API密钥

# 情感分类体系
SENTIMENT_CATEGORIES = ["正面", "中性", "负面"]
SENTIMENT_MAPPING = {"正面": 1, "中性": 0, "负面": -1}


def robust_read_csv(file_path, max_sample_size=100000):
    """增强版CSV读取函数，支持多种编码和错误恢复"""
    encodings = ['utf-8-sig', 'gb18030', 'gbk', 'utf-8', 'latin1']

    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(min(os.path.getsize(file_path), max_sample_size))
            result = chardet.detect(rawdata)
            if result['confidence'] > 0.9:
                encodings.insert(0, result['encoding'])
    except:
        pass

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if not df.empty and '微博正文' in df.columns:
                print(f"成功用 {encoding} 编码读取文件")
                return df

            if df.empty or '微博正文' not in df.columns:
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                df.columns = [col.encode('latin1').decode('gb18030', errors='ignore')
                              if isinstance(col, str) else col for col in df.columns]
                if '微博正文' in df.columns:
                    return df
        except Exception as e:
            continue

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

        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = [col.encode('latin1').decode('gb18030', errors='ignore')
                      if isinstance(col, str) else col for col in df.columns]
        if '微博正文' in df.columns:
            return df

        for col in df.select_dtypes(include=['object']):
            try:
                df[col] = df[col].str.encode('latin1').str.decode('gb18030', errors='ignore')
            except:
                pass
        return df
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}，所有编码尝试均失败。最后错误: {str(e)}")


def analyze_sentiment(text):
    """分析文本情感极性"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    请根据乘客对南京地铁的直接态度和感受进行情感分类：

    【分类标准】
    1. 正面(1): 直接表达对南京地铁的认可、满意、怀念、向往、自豪、纪念或赞赏等
       - 服务表扬: "南京地铁工作人员服务态度很热情"
       - 设施称赞: "车站的指引标识非常清晰易懂"
       - 体验好评: "乘车环境干净舒适，体验很好"
       - 归属感： "熟悉的南京地铁味"

    2. 负面(-1): 直接表达对南京地铁的不满、批评或抱怨
       - 服务批评: "安检效率太低，耽误时间"
       - 设施问题: "车厢空调温度太低，很不舒服"
       - 体验差评: "高峰期拥挤得让人窒息"

    3. 中性(0): 单纯描述事实或信息，无明显情感倾向
       - 客观陈述: "南京地铁3号线有29个车站"
       - 信息查询: "请问首班车是几点发车？"
       - 官方通告: "明日南京地铁将调整运营时间"

    【判断重点】
    - 核心关注乘客对南京地铁的直接感受和态度表达
    - 判断依据文本中明确表达的情感倾向
    - 反讽按字面意思判断
    - 混合情感以对南京地铁的情感态度为主
    - 疑问句根据隐含倾向判断

    【特殊情形处理】
    1. 对比表达: 
       - "比以前好多了" → 正面
       - "不如上海地铁" → 负面
       - "不如南京地铁" → 正面

    2. 建议类:
       - "希望改进服务" → 负面（隐含不满）
       - "建议增加班次" → 根据语气判断

    3. 体验描述:
       - "夏天车厢很凉快" → 正面
       - "冬天车厢太冷了" → 负面

    请直接返回最符合的分类：正面/中性/负面

    待分析内容："{text[:500]}"
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        # 标准化输出结果
        if "正面" in result:
            return "正面"
        elif "负面" in result:
            return "负面"
        else:
            return "中性"
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return "中性"


def evaluate_sentiment(true_labels, predicted_labels):
    """情感分析评价函数"""
    # 确保标签格式一致
    true_numeric = [SENTIMENT_MAPPING.get(label, 0) for label in true_labels]
    pred_numeric = [SENTIMENT_MAPPING.get(label, 0) for label in predicted_labels]

    # 计算混淆矩阵
    confusion_matrix = pd.crosstab(
        pd.Series(true_labels, name='真实'),
        pd.Series(predicted_labels, name='预测'),
        rownames=['真实'],
        colnames=['预测']
    )

    # 计算各类指标
    report = {}
    for sentiment in SENTIMENT_CATEGORIES:
        tp = sum((true == sentiment) & (pred == sentiment) for true, pred in zip(true_labels, predicted_labels))
        fp = sum((true != sentiment) & (pred == sentiment) for true, pred in zip(true_labels, predicted_labels))
        fn = sum((true == sentiment) & (pred != sentiment) for true, pred in zip(true_labels, predicted_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report[sentiment] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }

    # 计算总体准确率
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)

    # 打印报告
    print("\n=== 情感分析报告 ===")
    print(f"{'情感':<8}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'支持数':<10}")
    for sentiment in SENTIMENT_CATEGORIES:
        stats = report[sentiment]
        print(f"{sentiment:<8}{stats['precision']:<10.3f}"
              f"{stats['recall']:<10.3f}"
              f"{stats['f1']:<10.3f}"
              f"{stats['support']:<10}")

    print(f"\n总准确率: {accuracy:.4f}")

    # 设置全局字体（解决不匹配问题）
    plt.rcParams['font.family'] = 'SimHei'  # 中文用黑体
    plt.rcParams['font.size'] = 10  # 统一基础字号

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=SENTIMENT_CATEGORIES,
        yticklabels=SENTIMENT_CATEGORIES,
        linewidths = 0.5,
        annot_kws = {'size': 10}  # 标注字号与全局一致
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
    confusion_matrix_path = os.path.join(os.path.dirname(output_csv), 'sentiment_confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': confusion_matrix
    }


def process_sentiment_analysis(input_path, output_path, ground_truth_path=None):
    """处理情感分析流程"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 初始化日志
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"情感分析开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 读取数据
        print("正在读取CSV文件...")
        df = robust_read_csv(input_path)
        print(f"成功读取 {len(df)} 条记录")
        df['微博正文'] = df['微博正文'].fillna('').astype(str)

        # 读取预设结果（如果有）
        true_labels = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                gt_df = robust_read_csv(ground_truth_path)
                if '情感标签' in gt_df.columns:
                    true_labels = gt_df['情感标签'].tolist()
                    print(f"已加载 {len(true_labels)} 条预设情感标签")
            except Exception as e:
                print(f"读取预设结果失败: {str(e)}")

        # 初始化统计
        stats = {cat: 0 for cat in SENTIMENT_CATEGORIES}
        results = []

        # 处理数据
        print("开始情感分析...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row['微博正文'].strip()
            if not text:
                results.append("中性")
                continue

            sentiment = analyze_sentiment(text)
            results.append(sentiment)
            stats[sentiment] += 1

            # 每1000条保存进度
            if (idx + 1) % 1000 == 0:
                temp_df = pd.DataFrame({
                    '微博原文': df['微博正文'].iloc[:idx + 1],
                    '需求层次': df['需求层次'].iloc[:idx + 1],
                    '情感标签': results,
                    '情感数值': [SENTIMENT_MAPPING[s] for s in results]
                })
                temp_df.to_csv(output_path.replace('.csv', '_temp.csv'),
                               index=False, encoding='utf_8_sig')

            time.sleep(1.2)  # API限速

        # 保存最终结果
        result_df = pd.DataFrame({
            '微博原文': df['微博正文'],
            '需求层次': df['需求层次'],
            '情感标签': results,
            '情感数值': [SENTIMENT_MAPPING[s] for s in results]
        })
        result_df.to_csv(output_path, index=False, encoding='utf_8_sig')

        # 生成统计报告
        stats_df = pd.DataFrame({
            '情感类别': SENTIMENT_CATEGORIES,
            '数量': [stats[cat] for cat in SENTIMENT_CATEGORIES],
            '占比': [stats[cat] / len(df) for cat in SENTIMENT_CATEGORIES]
        })
        stats_df['占比'] = stats_df['占比'].apply(lambda x: f"{x:.2%}")
        stats_df.to_csv(output_path.replace('.csv', '_stats.csv'),
                        index=False, encoding='utf_8_sig')

        # 如果有预设结果，进行评价
        if true_labels and len(true_labels) == len(results):
            print("\n=== 开始模型评价 ===")
            evaluation = evaluate_sentiment(true_labels, results)

            # 保存评价结果
            eval_results = []
            for sentiment in SENTIMENT_CATEGORIES:
                eval_results.append({
                    '情感类别': sentiment,
                    '精确率': evaluation['report'][sentiment]['precision'],
                    '召回率': evaluation['report'][sentiment]['recall'],
                    'F1分数': evaluation['report'][sentiment]['f1'],
                    '支持数': evaluation['report'][sentiment]['support']
                })
            eval_results.append({
                '情感类别': '总体',
                '精确率': '',
                '召回率': '',
                'F1分数': '',
                '支持数': len(results),
                '准确率': evaluation['accuracy']
            })

            eval_df = pd.DataFrame(eval_results)
            eval_path = output_path.replace('.csv', '_evaluation.csv')
            eval_df.to_csv(eval_path, index=False, encoding='utf_8_sig')

        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理完成 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总计处理: {len(df)} 条\n")
            f.write("情感分布:\n" + stats_df.to_string(index=False))
            if true_labels:
                f.write("\n\n模型评价结果:\n")
                f.write(eval_df.to_string(index=False))

        print(f"\n处理完成！结果已保存至: {output_path}")
        print("情感分布统计:")
        print(stats_df.to_string(index=False))

    except Exception as e:
        print(f"处理失败: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        process_sentiment_analysis(input_csv, output_csv, ground_truth_csv)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n用户中断操作")
    except Exception as e:
        print(f"\n程序错误: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n程序错误: {str(e)}")