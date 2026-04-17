import pandas as pd
import requests
import json
import time
import os
import chardet
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# 配置路径
input_csv = r"D:\weibo-search\结果文件\南京地铁\一版.csv"
output_csv = r"D:\weibo-search\结果文件\南京地铁\classified_results.csv"
log_file = r"D:\weibo-search\结果文件\南京地铁\processing_log.txt"

# DeepSeek API配置
load_dotenv()
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 分类体系
CATEGORIES = ["基础层", "保障层", "舒适层", "尊重层", "共鸣层", "其他"]


def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(100000))['encoding']


def robust_read_csv(file_path):
    for encoding in ['gb18030', 'gbk', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if not df.empty and '微博正文' in df.columns:
                return df
        except:
            continue
    try:
        return pd.read_csv(file_path, encoding=detect_file_encoding(file_path))
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")


def call_deepseek_api(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = """
    请严格判断微博内容是否反映用户需求，参考定义和示例辅助理解：

        【层次定义与示例参考】
        仅当内容反映乘客实际需求时，选择对应层级：
        1. 基础层（安全与时效）：
        - 定义：保障乘客基本出行安全和时间可靠性的需求
        - 严格限定：直接反映南京地铁基础运维质量
        - 必须包含：乘客对行车安全/运营时效的直接评价
        - 示例参考：安检问题、设备故障、运行延误、急刹等

        2. 保障层（基本生存资料供给）：
        - 定义：确保出行基本条件满足的需求  
        - 严格限定：与南京地铁提供的核心和便利性服务功能直接相关
        - 必须包含：乘客对设施/服务/通车运营的功能性评价  
        - 示例参考：设施便利性、电梯扶梯、线路可达性（直达）、支付保障、乘客诉求、动线复杂、通车运营等

        3. 舒适层（环境体验）：
        - 定义：乘客对南京地铁物理及信息环境的舒适度感受
        - 严格限定：南京地铁创造的环境体验
        - 必须包含：乘客对乘车环境的直接感受
        - 包含维度：
            (1) 物理环境：环境卫生、温湿度、空调、冷热、空气质量、座椅设施等
            (2) 信息环境：提示音音量、报站清晰度、电子看板指引等
            (3) 空间体验：拥挤度、无障碍设计等

       4. 尊重层（人性化服务）：
        - 定义：乘客获得尊重和人性化服务的需求，强调工作人员对乘客的尊重与关怀，强调南京地铁主动提供的个性化服务
        - 严格限定：
            √ 必须是由南京地铁主动提供的文化服务（如：毕业季广播、特色主题车站、盖章等）
            √ 必须与南京地铁提供的个性化服务直接相关
        - 必须包含：乘客与地铁人员/地铁主动服务的互动评价
        - 示例参考：工作人员、服务态度、特殊关怀、隐私保护、文化体验、节日活动、毕业季等

        5. 共鸣层（情感共鸣与价值归属）：
        - 定义：超越交通功能，严格反映乘客对南京地铁自发产生的情感联结
        - 严格限定：必须表达乘客自发产生的情感联结（如：怀念、向往、自豪、纪念）
        - 示例参考：归属感等

    【分类原则】
    1. 核心判断依据是层次定义而非具体示例
    2. 示例仅用于辅助理解层次内涵
    3. 当内容反映的需求符合某层次定义时即归类，不符合上述五类的归为“其他”
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
    基础层/保障层/舒适层/尊重层/共鸣层/其他

    待分析内容："{}"
    """.format(text[:500])  # 限制输入长度

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # 保持适度灵活性
        "max_tokens": 10
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        return result if result in CATEGORIES else "其他"
    except:
        return "其他"


def process_csv(input_path, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 初始化日志
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"分类处理开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 读取数据
        print("正在读取CSV文件...")
        df = robust_read_csv(input_path)
        print(f"成功读取 {len(df)} 条记录")
        df['微博正文'] = df['微博正文'].fillna('')

        # 初始化统计
        stats = {cat: 0 for cat in CATEGORIES}
        results = []

        # 处理数据
        print("开始需求层次分类...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row['微博正文']).strip()
            if not text:
                results.append("其他")
                continue

            category = call_deepseek_api(text)
            results.append(category)
            stats[category] += 1

            # 每1000条保存进度
            if (idx + 1) % 1000 == 0:
                pd.DataFrame({
                    '原文': df['微博正文'].iloc[:idx + 1],
                    '需求层次': results
                }).to_csv(output_path.replace('.csv', '_temp.csv'), index=False, encoding='utf_8_sig')

            time.sleep(1.2)  # API限速

        # 保存最终结果
        result_df = pd.DataFrame({
            '微博原文': df['微博正文'],
            '需求层次': results
        })
        result_df.to_csv(output_path, index=False, encoding='utf_8_sig')

        # 生成统计报告
        stats_df = pd.DataFrame({
            '需求层次': CATEGORIES,
            '数量': [stats[cat] for cat in CATEGORIES],
            '占比': [f"{stats[cat] / len(df):.1%}" for cat in CATEGORIES]
        }).sort_values('数量', ascending=False)

        stats_df.to_csv(output_path.replace('.csv', '_stats.csv'), index=False)

        # 排序并格式化
        stats_df = stats_df.sort_values('数量', ascending=False)
        stats_df['占比'] = stats_df['占比'].apply(lambda x: f"{x:.2%}")

        # 确保列名是简体中文
        stats_df.columns = ['需求层次', '数量', '占比']

        # 保存统计结果
        stats_df.to_csv(output_path.replace('.csv', '_stats.csv'),
                        index=False,
                        encoding='utf_8_sig')

        # 记录完成日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理完成 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总计处理: {len(df)} 条\n")
            f.write("分类统计:\n" + stats_df.to_string(index=False))

        print(f"\n处理完成！结果已保存至: {output_path}")
        print("分类统计:")
        print(stats_df.to_string(index=False))

    except Exception as e:
        print(f"处理失败: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n处理失败: {str(e)}")


if __name__ == "__main__":
    try:
        process_csv(input_csv, output_csv)
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n程序错误: {str(e)}")