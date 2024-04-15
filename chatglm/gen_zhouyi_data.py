import csv
import datetime
import os
import json
import time
from configs import conf
from openai import OpenAI
client = OpenAI()
client.api_key =  conf.get("api_key")

system_parse_content = """
你是中国古典哲学大师，尤其擅长周易的哲学解读。
接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容。
返回结果采用json格式,content字段是具体的卦名,summary字段是该卦的具体内容。

示例输入：

师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

期待结果:

{"content":"师卦",
"summary":"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"}
"""
system_make_question = """
你是中国古典哲学大师。对于周易中的64卦，请根据输入的卦名，构建可能的用户提问方式。要遵守下面的规则：1.这些提问方式可以是疑问或者陈述。2.一个回答中只包含一个提问。3.不要涉及其中的具体卦词及爻词。4.不要问各个爻的变动情况及其影响。
"""

#使问题正常，
#1.将'这个卦'替换为卦名
#2.如果整个问题中没有出现卦名，则将卦名加到问题的前面。
def make_question_normal(question, gua_name):
    gua_name = gua_name[0:1] + '卦'
    m = question.replace('这个卦', gua_name)
    if m.find(gua_name) < 0:
        m = gua_name + '，' + m
    return m
    
def make_questions(gua_name, count):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
        {"role": "system", "content": system_make_question},
        {"role": "user", "content": gua_name}
        ],
        temperature=0.2,
        stream=False,
        n=count,
    )

    questions = [choice.message.content for choice in completion.choices]
    return map(lambda x : make_question_normal(x, gua_name), questions)

#print(list(make_questions('乾')))
def ai_parse_raw_content(raw_content):
    while(True):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
            {"role": "system", "content": system_parse_content},
            {"role": "user", "content": raw_content}
            ],
            stream=False,
            temperature=1,
            n=1,
        )
        response = completion.choices[0].message.content
        print('response', response)
        try:
            t = json.loads(response)
            return t['content'], t['summary']
        except Exception as e:
            print('Exception', e)
        time.sleep(1)
"""    
raw_content = "蒙卦是教育启蒙的智慧，艮为山，坎为泉，山下出泉。泉水始流出山，则必将渐汇成江河,正如蒙稚渐启，又山下有险，因为有险停止不前，所以蒙昧不明。事物发展的初期阶段，必然蒙昧，所以教育是当务之急，养学生纯正无邪的品质，是治蒙之道。\n蒙卦，这个卦是异卦相叠，下卦为坎，上卦为艮。艮是山的形象，喻止；坎是水的形象，喻险。卦形为山下有险，仍不停止前进，是为蒙昧，故称蒙卦。但因把握时机，行动切合时宜;因此，具有启蒙和通达的卦象。\n《蒙》卦是《屯》卦这个始生卦之后的第二卦。《序卦》中说：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”物之幼稚阶段，有如蒙昧未开的状态，在人则是指童蒙。\n《象》中这样解释蒙卦：山下出泉，蒙；君子以果行育德。"
m = parse_raw_content(raw_content)
print(m)
"""

def generate_question_summary_pairs(gua_name, summary):
    questions = make_questions(gua_name, 20)
    pairs = [(question, summary) for question in questions]
    return pairs
    
def main():
    # 确保 data 目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 解析 data/raw_data.txt 得到 raw_content_data 列表
    raw_content_data = []
    with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        data_samples = content.split('\n\n')
        for sample in data_samples:
            cleaned_sample = sample.strip()
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    # 创建带有时间戳的CSV文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/zhouyi_dataset_{timestamp}.csv"

    # 创建CSV文件并写入标题行
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for raw_content in raw_content_data:
            # 调用 ai_parse_raw_content 方法得到 content和summary
            content, summary = ai_parse_raw_content(raw_content)
            
            print("Content:", content)
            print("Summary:", summary)

            # 调用 generate_question_summary_pairs 得到20组 pairs
            pairs = generate_question_summary_pairs(content, summary)

            # 将 pairs 写入 csv 文件
            for pair in pairs:
                writer.writerow(pair)

main()