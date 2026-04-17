import pandas as pd
import time
import os
from tqdm import tqdm
from collections import defaultdict
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置路径
input_csv = r"D:\weibo-search\结果文件\南京地铁\三版.csv"
output_dir = r"D:\weibo-search\结果文件\南京地铁\情感因素分析"
os.makedirs(output_dir, exist_ok=True)

# 初始化OpenAI客户端
my_api_key = "OPENAI_API_KEY"  # 请替换为您的实际API密钥
client = OpenAI(api_key=my_api_key)

# 分类体系
DEMAND_LEVELS = ["基础层", "保障层", "舒适层", "尊重层", "共鸣层"]
SENTIMENT_CATEGORIES = ["正面", "负面"]


def load_data():
    """加载已分类数据"""
    print("正在加载数据...")
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print(f"成功加载 {len(df)} 条数据")
    return df


def analyze_factors_with_openai(text, demand_level, sentiment):
    """使用OpenAI API分析情感因素（优化版prompt结构）"""
    messages = [
        {
            "role": "system",
            "content": """你作为南京地铁服务分析专家，请严格从乘客反馈中提取南京地铁的具体服务特征要素。请遵守：

            核心原则：
            1. 所有输出关键词必须直接指向南京地铁的具体属性/服务/特征,必须是可以被南京地铁管理部门直接处理的具体问题
            2. 表面现象→本质问题：将乘客的具体遭遇映射到南京地铁的可改进点
            3. 采用最精简的名词短语（3-5个汉字）
            4. 禁止任何修饰词和情感词

            输出要求:
            - 必须包含"地铁"或能明确对应其服务的词汇，不能只输出“南京地铁”
            - 每个要素3-7个汉字
            - 用中文逗号分隔，不要解释"""
        },
        {
            "role": "user",
            "content": f"""请提取南京地铁的具体服务要素：

            【乘客反馈原文】
            {text}

            【分类信息】
            需求层次：{demand_level}
            情感倾向：{sentiment}

            【转化示例】
             输入："坐南京地铁十几年了真有感情"
             → 输出：地铁历史

             输入："毕业季的车厢装饰太浪漫了"
             → 输出：毕业季装饰

             输入："列车空调冷得像冰窖"
             → 输出：列车空调

             输入："安检员帮我找回了钱包"
             → 输出：安检服务

            【当前任务】
            请提取1-3个南京地铁具体服务要素："""
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip()
        factors = [
            f.strip()
            for f in result.replace("，", ",").split(",")
            if f.strip() and 2 <= len(f.strip()) <= 5
        ]
        return factors if factors else ["无法提取"]
    except Exception as e:
        print(f"\nAPI处理出错（文本前30字：'{text[:30]}'）: {str(e)}")
        return ["API错误"]


def process_factors_analysis(df):
    """处理情感因素分析"""
    factors_results = []
    factors_stats = defaultdict(lambda: defaultdict(int))
    total_rows = len(df)

    print(f"\n开始分析 {total_rows} 条数据...")

    with tqdm(total=total_rows, desc="处理进度") as pbar:
        for idx, row in df.iterrows():
            try:
                factors = analyze_factors_with_openai(
                    row['微博原文'],
                    row['需求层次'],
                    row['情感标签']
                )

                record = {
                    '序号': idx + 1,
                    '需求层次': row['需求层次'],
                    '情感标签': row['情感标签'],
                    '微博原文': row['微博原文'][:100] + "..." if len(row['微博原文']) > 100 else row['微博原文'],
                    '情感因素': "，".join(factors),
                    '因素数量': len(factors)
                }
                factors_results.append(record)

                if factors[0] not in ["无法提取", "API错误"]:
                    for factor in factors:
                        factors_stats[f"{row['需求层次']}-{row['情感标签']}"][factor] += 1

            except Exception as e:
                print(f"\n处理第 {idx + 1} 行时出错: {str(e)}")
                factors_results.append({
                    '序号': idx + 1,
                    '错误信息': str(e)
                })

            pbar.update(1)
            time.sleep(1.2)

    # 保存结果
    results_df = pd.DataFrame(factors_results)
    results_df.to_csv(
        os.path.join(output_dir, "主题生成结果.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    stats_output = {
        group: {
            "总记录数": sum(stats.values()),
            "高频因素": dict(sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        for group, stats in factors_stats.items()
    }

    with open(os.path.join(output_dir, "统计结果.json"), "w", encoding='utf-8') as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)

    return results_df, stats_output


def main():
    df = load_data()
    results_df, stats = process_factors_analysis(df)

    print("\n==== 分析摘要 ====")
    print(f"总处理行数: {len(results_df)}")
    print(f"成功提取行数: {len(results_df[results_df['因素数量'] > 0])}")

    print("\n各分组统计：")
    for group, data in stats.items():
        print(f"\n【{group}】")
        print(f"有效因素总数: {data['总记录数']}")
        print("高频因素TOP5:")
        for factor, count in list(data['高频因素'].items())[:5]:
            print(f"  {factor}: {count}次")

    print(f"\n分析完成！结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()