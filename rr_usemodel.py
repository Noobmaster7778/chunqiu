#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 1. 定义模型（与训练时一致）
# ---------------------------
class EnhancedRelationModel(tf.keras.Model):
    def __init__(self, char_vocab_size, pos_vocab_size, num_relations):
        super(EnhancedRelationModel, self).__init__()
        self.char_embed = tf.keras.layers.Embedding(char_vocab_size, 128, mask_zero=True)
        self.subj_pos_embed = tf.keras.layers.Embedding(pos_vocab_size, 32)
        self.obj_pos_embed = tf.keras.layers.Embedding(pos_vocab_size, 32)

        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2
        ))
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(num_relations, activation='softmax')

    def call(self, inputs):
        char_input = inputs["char_input"]
        subj_pos = inputs["subj_pos_input"]
        obj_pos = inputs["obj_pos_input"]

        char_emb = self.char_embed(char_input)
        subj_emb = self.subj_pos_embed(subj_pos)
        obj_emb = self.obj_pos_embed(obj_pos)

        combined = tf.concat([char_emb, subj_emb, obj_emb], axis=-1)
        lstm_out = self.bilstm(combined)
        attn_out = self.attention([lstm_out, lstm_out])
        pooled = tf.reduce_mean(attn_out, axis=1)
        x = self.dense1(pooled)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

# ---------------------------
# 2. 参数与辅助函数
# ---------------------------
max_seq_length = 100
pos_bins = 20
vocab_size = 75
num_relations = 14

# 加载字符词典（训练时保存的），否则使用默认
try:
    with open("char2idx.json", "r", encoding="utf-8") as f:
        char2idx = json.load(f)
except Exception as e:
    print("未能加载 char2idx.json，使用默认映射")
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    ascii_chars = [chr(i) for i in range(32, 32 + 73)]
    for i, c in enumerate(ascii_chars, start=2):
        char2idx[c] = i

def position_bucket(entity_pos, seq_len):
    positions = []
    for i in range(seq_len):
        distance = i - entity_pos
        if distance < -10:
            bucket = 0
        elif distance > 10:
            bucket = pos_bins - 1
        else:
            bucket = (distance + 10) // 2 + 1
        positions.append(bucket)
    return positions

def safe_char_index(c):
    idx = char2idx.get(c, char2idx.get("<UNK>", 1))
    if idx >= vocab_size:
        return char2idx.get("<UNK>", 1)
    return idx

def extract_features(sentence, subj_text, obj_text):
    tokens = list(sentence)
    seq_length = len(tokens)
    char_ids = [safe_char_index(c) for c in tokens]
    char_ids = pad_sequences([char_ids], maxlen=max_seq_length, padding="post", truncating="post")[0]

    subj_index = sentence.find(subj_text)
    obj_index = sentence.find(obj_text)
    if subj_index == -1:
        subj_index = 0
    if obj_index == -1:
        obj_index = 0

    subj_pos = position_bucket(subj_index, seq_length)
    obj_pos = position_bucket(obj_index, seq_length)
    subj_pos = pad_sequences([subj_pos], maxlen=max_seq_length, padding="post", truncating="post")[0]
    obj_pos = pad_sequences([obj_pos], maxlen=max_seq_length, padding="post", truncating="post")[0]

    features = {
        "char_input": np.expand_dims(char_ids, axis=0),
        "subj_pos_input": np.expand_dims(subj_pos, axis=0),
        "obj_pos_input": np.expand_dims(obj_pos, axis=0)
    }
    return features

# ---------------------------
# 3. 定义过滤规则与触发词
# ---------------------------
# 全局排除实体类型：Time, Weather（事件除外）
global_exclude_types = {"Time", "Weather"}

# 定义每种关系允许的实体类型组合
RELATION_TYPE_RULES = {
    "Marriage": {("Person", "Person")},
    "War": {("Person", "Person"), ("Person", "Location"), ("Location", "Person"), ("Location", "Location")},
    "Alliance": {("Person", "Person"), ("Person", "Location"), ("Location", "Person")},
    "Diplomatic Visit": {("Person", "Person"), ("Person", "Location")},
    "Death": {("Person", "Event"), ("Event", "Person")},
    "Crowning": {("Person", "Person")},
    "Dispatch": {("Person", "Person"), ("Person", "Location")},
    "Gift": {("Person", "Person")},
    "Occupation": {("Person", "Location"), ("Location", "Person")},
    "Meeting": {("Person", "Person")},
    "Move": {("Person", "Location")},
    # Natural Disaster 如需要，可添加，但如果Time/Weather参与则全局过滤
    "无关系": {("Any", "Any")}
}

# 定义每种关系的触发词列表（可根据实际情况扩充）
TRIGGER_WORDS = {
    "Marriage": {"婚", "嫁", "娶"},
    "War": {"伐", "战", "侵", "败", "杀", "灭", "攻"},
    "Alliance": {"盟", "结盟", "会盟"},
    "Diplomatic Visit": {"使", "来聘", "朝", "访"},
    "Death": {"卒", "崩", "薨", "死"},
    "Crowning": {"即位", "加冕", "立位", "封"},
    "Dispatch": {"使", "派", "遣"},
    "Gift": {"赠", "馈赠"},
    "Occupation": {"占领", "侵占"},
    "Meeting": {"会见", "见面"},
    "Move": {"入", "迁", "逃", "奔", "降"},
    # Natural Disaster 如果需要
}

# 为每种关系设置一个置信度下限（可以针对关系类型调整）
REL_CONF_THRESHOLDS = {
    "Marriage": 0.6,
    "War": 0.7,
    "Alliance": 0.65,
    "Diplomatic Visit": 0.7,
    "Death": 0.7,
    "Crowning": 0.7,
    "Dispatch": 0.6,
    "Gift": 0.6,
    "Occupation": 0.7,
    "Meeting": 0.6,
    "Move": 0.55,
    "无关系": 0.0,
    # 如果需要 Natural Disaster
}

def is_valid_pair(internal_rel, e1, e2, sentence):
    # 若任一实体为Time或Weather，则过滤
    if e1["type"] in global_exclude_types or e2["type"] in global_exclude_types:
        return False

    # Marriage要求实体不能是同一个
    if internal_rel == "Marriage" and e1["text"] == e2["text"]:
        return False

    # 检查实体类型组合是否符合规则
    allowed = RELATION_TYPE_RULES.get(internal_rel, {("Any", "Any")})
    pair_types = (e1["type"], e2["type"])
    if pair_types not in allowed and (e2["type"], e1["type"]) not in allowed:
        return False

    # 触发词检查（如果该关系有触发词要求，检查句子中是否至少出现一个触发词）
    triggers = TRIGGER_WORDS.get(internal_rel, set())
    if triggers:
        if not any(t in sentence for t in triggers):
            return False

    return True

# ---------------------------
# 4. 读取推理输入文件，并转换样本格式
# ---------------------------
input_ner_file = "ner_output.txt"
with open(input_ner_file, "r", encoding="utf-8") as f:
    ner_text = f.read()

pattern = re.compile(r"原始句子：(.*?)\n识别的实体：(.*?)\n\n", re.DOTALL)
matches = pattern.findall(ner_text)

def transform_inference_sample(sentence_raw, entities_raw):
    # 如果句子包含【...】头部，则移除头部部分，不用于实体定位
    header_match = re.match(r"【(.*?)】", sentence_raw)
    if header_match:
        sentence_body = sentence_raw[len(header_match.group(0)):].strip()
    else:
        sentence_body = sentence_raw.strip()

    transformed = []
    for (text, tag) in entities_raw:
        if tag == "O":
            continue
        start = sentence_body.find(text)
        if start == -1:
            continue
        end = start + len(text)
        transformed.append({
            "text": text,
            "type": tag,
            "start": start,
            "end": end
        })
    return {"sentence": sentence_body, "entities": transformed}

samples = []
for sentence_raw, entities_str in matches:
    try:
        entities_raw = eval(entities_str)
    except Exception as e:
        entities_raw = []
    sample = transform_inference_sample(sentence_raw, entities_raw)
    if not sample["sentence"] or len(sample["entities"]) < 2:
        continue
    samples.append(sample)

print(f"共转换出 {len(samples)} 个样本。")

def generate_entity_pairs(entities):
    pairs = []
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            # 如果两个实体完全相同则跳过
            if entities[i]["start"] == entities[j]["start"] and entities[i]["end"] == entities[j]["end"]:
                continue
            pairs.append((entities[i], entities[j]))
    return pairs

# ---------------------------
# 5. 构造模型并加载权重
# ---------------------------
inference_model = EnhancedRelationModel(
    char_vocab_size=vocab_size,
    pos_vocab_size=pos_bins,
    num_relations=num_relations
)
dummy_features = {
    "char_input": tf.zeros((1, max_seq_length), dtype=tf.int32),
    "subj_pos_input": tf.zeros((1, max_seq_length), dtype=tf.int32),
    "obj_pos_input": tf.zeros((1, max_seq_length), dtype=tf.int32),
}
_ = inference_model(dummy_features)
inference_model.load_weights("optimized_relation_model.weights.h5")
print("✅ 模型权重加载成功！")

# 假设模型输出的内部标签顺序如下（必须与训练时一致）
label_list = [
    "无关系", "Dispatch", "Gift", "Alliance", "War", "Marriage",
    "Occupation", "Crowning", "Natural Disaster", "Death",
    "Meeting", "Move", "Diplomatic Visit", "Extra"
]

# ---------------------------
# 6. 推理、过滤与选择最佳候选（每个句子仅输出一个最佳关系）
# ---------------------------
label_mapping = {
    "无关系": "无关系",
    "Dispatch": "Dispatch",
    "Gift": "Gift",
    "Alliance": "Alliance",
    "War": "War",
    "Marriage": "Marriage",
    "Occupation": "Occupation",
    "Crowning": "Crowning",
    "Natural Disaster": "Natural Disaster",
    "Death": "Death",
    "Meeting": "Meeting",
    "Move": "Move",
    "Diplomatic Visit": "Diplomatic Visit",
    "Extra": "Extra"  # 如果训练集中有额外类别
}
relation_results = []
for idx, sample in enumerate(samples):
    sentence = sample["sentence"]
    entities = sample["entities"]
    pairs = generate_entity_pairs(entities)

    best_score = -1.0
    best_pred = None

    for pair in pairs:
        e1, e2 = pair
        # 全局排除 Time / Weather
        if e1["type"] in global_exclude_types or e2["type"] in global_exclude_types:
            continue

        features = extract_features(sentence, e1["text"], e2["text"])
        preds = inference_model.predict(features, verbose=0)
        score = float(np.max(preds))
        rel_idx = int(np.argmax(preds))
        internal_rel = label_list[rel_idx]

        # 如果预测为 "无关系" 则跳过
        if internal_rel == "无关系":
            continue

        # 检查实体类型组合和触发词是否符合该关系要求
        if not is_valid_pair(internal_rel, e1, e2, sentence):
            continue

        # 根据关系类型取对应的置信度阈值
        conf_thresh = REL_CONF_THRESHOLDS.get(internal_rel, 0.55)
        if score < conf_thresh:
            continue

        # 如果当前候选比之前更高分，则更新最佳候选
        if score > best_score:
            best_score = score
            final_rel = label_mapping.get(internal_rel, internal_rel)
            best_pred = {
                "sentence": sentence,
                "entity_pair": (e1, e2),
                "relation": final_rel,
                "score": score
            }
    if best_pred:
        relation_results.append(best_pred)
    if (idx + 1) % 100 == 0:
        print(f"处理 {idx+1}/{len(samples)} 个样本...")

print(f"✅ 最终输出 {len(relation_results)} 条关系结果。")

# ---------------------------
# 7. 保存结果
# ---------------------------
with open("relation_output.txt", "w", encoding="utf-8") as f_out:
    for res in relation_results:
        f_out.write(f"句子：{res['sentence']}\n")
        f_out.write(f"实体对：{res['entity_pair']}\n")
        f_out.write(f"关系：{res['relation']} (置信度：{res['score']:.4f})\n\n")

print("✅ 关系识别结果已保存至 relation_output.txt")
