import itertools
import random
import json

# 定义句子
sentences = {
    "short": "Berlin is rich in culture, history, and art.",
    "short1": "Munich celebrates rich traditions, architecture, and a love for the arts.",
    "short2": "Lisbon shines with its historic charm and a lively maritime heritage.",
    "medium": "Vienna, located in the heart of Europe, combines old-world charm with a vibrant cultural scene, attracting tourists from around the world.",
    "medium1": "Prague, nestled in the heart of Europe, showcases medieval architecture mixed with a bustling modern lifestyle, appealing to travelers worldwide.",
    "medium2": "Amsterdam, famous for its canals and bicycles, offers a unique blend of historic sites and a progressive social scene, captivating visitors from every corner.",
    "long": "San Francisco, recognized globally for its iconic Golden Gate Bridge and vibrant culture, serves as a major hub for tourism and technology, offering a rich tapestry of arts, diverse communities, and breathtaking landscapes, attracting millions of visitors each year.",
    "long1": "Melbourne, known for its eclectic architecture and vibrant cultural scene, acts as a beacon for arts and education in Australia, hosting numerous festivals that attract global audiences each year.",
    "long2": "Vancouver, with its stunning mountain backdrop and thriving tech scene, serves as a cultural and economic powerhouse, celebrated for its diversity and environmental innovations.",
    "very_long": "Tokyo, the bustling capital of Japan, boasts a unique blend of traditional and modern influences, housing over thirteen million residents. Its diverse districts offer everything from historic temples and serene gardens to cutting-edge technology and fashion hubs. Tokyo's culinary scene is globally renowned, ranging from Michelin-starred restaurants to local izakayas, providing an unparalleled gastronomic experience. The city also thrives on its cultural festivals, state-of-the-art transportation, and a dynamic business environment that draws entrepreneurs and tourists alike.",
    "very_long1": "Seoul, the dynamic capital of South Korea, merges ancient palaces with skyscrapers, offering a spectrum from traditional markets to digital innovation hubs. Its food scene, highlighted by street vendors and fine dining, along with popular cultural exports like music and cinema, makes it a magnet for international tourists and business investors.",
    "very_long2": "Singapore, a bustling metropolis at the crossroads of Asia, stands out with its futuristic architecture and multicultural population. It hosts a myriad of attractions including lush green parks, world-class zoos, and a vibrant culinary landscape, making it a top destination for both leisure and business pursuits."
}

# 定义符号
symbols = ['.', '\n', '.\n']

# 生成所有符号组合的个数（0到4个每种符号）
symbol_combinations = [(0, 0, 0)]  # 添加无符号的原始句子情况
for i in range(5):
    for j in range(5):
        for k in range(5):
            if i + j + k > 0:  # 确保至少有一个符号
                symbol_combinations.append((i, j, k))

# 插入符号的函数
def insert_symbols(base_sentence, dot_count, newline_count, dot_newline_count):
    edits = ['.'] * dot_count + ['\n'] * newline_count + ['.\n'] * dot_newline_count
    positions = list(range(1, len(base_sentence.split())))  # 可插入的位置
    random.shuffle(positions)
    positions = positions[:len(edits)]  # 随机选取足够的位置
    words = base_sentence.split()
    for edit, pos in sorted(zip(edits, positions), key=lambda x: x[1], reverse=True):
        # 确保插入点后的所有位置索引增加，避免由于插入影响后续索引
        offset = sum(1 for p in positions if p <= pos)
        words.insert(pos + offset, edit)
    return ' '.join(words)

# 为每种组合生成三个示例
results = {}
for length, sentence in sentences.items():
    results[length] = {}
    for combination in symbol_combinations:
        dot_count, newline_count, dot_newline_count = combination
        combination_key = f"{dot_count},{newline_count},{dot_newline_count}"
        results[length][combination_key] = []
        for _ in range(16):  # 生成八个示例
            variation = insert_symbols(sentence, dot_count, newline_count, dot_newline_count)
            results[length][combination_key].append(variation)

# 将结果保存为JSON文件
with open('sentence_variations.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 打印确认消息
print("Sentences with symbol variations have been saved to 'sentence_variations.json'.")

