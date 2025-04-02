# -*- coding: utf-8 -*-
"""
用於處理討論串數據的模組
"""
import pandas as pd
import torch
from itertools import chain
from collections import Counter
from config import SEQ_LENGTH, GENERATE_NUM, DEVICE, PATH

def read_discussion_csv(file='ptt_real.csv', pad_token='PAD', num=None):
    """從CSV文件讀取討論串數據"""
    try:
        data = pd.read_csv(PATH + file)
        
        if num is not None:
            num = min(num, len(data))
            data = data[:num]
        
        # 按照相同的post分組為討論串
        threads = {}
        current_post = None
        responses = []
        
        for i, row in data.iterrows():
            post = row['post']
            response = row['response']
            
            if current_post != post:
                if current_post is not None:
                    threads[current_post] = responses
                current_post = post
                responses = [response]
            else:
                responses.append(response)
        
        # 添加最後一個討論串
        if current_post is not None:
            threads[current_post] = responses
        
        # 準備詞彙表
        all_text = []
        for post, resps in threads.items():
            all_text.append(post)
            all_text.extend(resps)
        
        characters = set(chain.from_iterable([list(text) for text in all_text]))
        
        # 創建詞彙映射
        vocabulary = dict([(y, x+1) for x, y in enumerate(characters)])
        reverse_vocab = dict([(x+1, y) for x, y in enumerate(characters)])
        
        # 添加特殊標記
        vocabulary['START'] = 0
        reverse_vocab[0] = 'START'
        if pad_token not in vocabulary:
            vocabulary[pad_token] = len(vocabulary)
            reverse_vocab[len(vocabulary)-1] = pad_token
        vocabulary['END'] = len(vocabulary)
        reverse_vocab[len(vocabulary)-1] = 'END'
        
        # 準備訓練數據：上下文(帖子+先前回應) -> 下一個回應
        training_data = []
        context_lengths = []
        
        for post, responses in threads.items():
            if len(responses) < 2:  # 需要至少2個回應才能形成上下文
                continue
                
            for i in range(1, len(responses)):
                # 上下文是帖子+先前的回應
                context = post + " " + " ".join(responses[:i])
                context = ['START'] + list(context)
                
                # 如果上下文太長，截斷
                if len(context) > SEQ_LENGTH - 1:
                    context = context[:SEQ_LENGTH - 1]
                
                # 記錄填充前的長度
                context_length = len(context)
                
                # 填充到固定長度
                if len(context) < SEQ_LENGTH:
                    context.extend([pad_token] * (SEQ_LENGTH - len(context)))
                
                # 將標記轉換為ID
                context_ids = [vocabulary.get(token, vocabulary.get(pad_token)) for token in context]
                
                # 加入訓練數據
                training_data.append(context_ids)
                context_lengths.append(context_length)
        
        # 轉換為張量
        x = torch.tensor(training_data, device=DEVICE).view(-1, SEQ_LENGTH)
        
        return x.int(), vocabulary, reverse_vocab, context_lengths
    
    except Exception as e:
        print(f"處理討論CSV時出錯: {e}")
        return None, None, None, None

def gen_discussion_data(num=GENERATE_NUM, vocab_size=10):
    """生成用於測試的合成討論數據"""
    # 生成隨機序列數據
    data = torch.rand(num, SEQ_LENGTH-1, device=DEVICE)
    data = torch.abs(data * (vocab_size-2)).int()+1
    data = torch.cat([torch.zeros([num,1],device=DEVICE).int(), data], dim=1)
    return data

def gen_label(num=GENERATE_NUM, target_space=2, fixed_value=None):
    """生成標籤"""
    if fixed_value is None:
        return torch.randint(low=0, high=target_space, size=(num,), device=DEVICE).long()
    else:
        assert fixed_value < target_space
        return torch.randint(low=fixed_value, high=fixed_value+1, size=(num,), device=DEVICE).long()

def decode_thread(token_tbl, reverse_vocab, log=None):
    """將標記ID解碼回文本"""
    texts_all = []
    for n in token_tbl:
        tokens = [reverse_vocab[int(l)] for l in n]
        text = ''.join(tokens[1:])  # 跳過START標記
        texts_all.append(text)
        if log is not None:
            log.write(text + '\n')
    return texts_all