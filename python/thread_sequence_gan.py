# -*- coding: utf-8 -*-
"""
基於討論串的SeqGAN主程序
"""
import sys
from datetime import datetime
import torch
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, PATH, NrGPU, openLog
from data_processing_threads import read_discussion_csv, decode_thread, gen_label
from thread_lstm_generator import pretrain_thread_generator
from thread_generator import ThreadGenerator, train_thread_generator
from discriminator import train_discriminator
from rollout import Rollout, getReward

def main(batch_size=1, input_file='ptt_real.csv', num=None):
    """主程序"""
    if batch_size is None:
        batch_size = 1
        
    # 讀取討論資料
    log = openLog()
    log.write("###### 讀取討論資料: {}\n".format(datetime.now()))
    log.close()
    
    # 從CSV檔案讀取討論串資料
    x, vocabulary, reverse_vocab, sentence_lengths = read_discussion_csv(file=input_file, num=num)
    
    if x is None or len(x) == 0:
        print("錯誤: 無法讀取討論資料或資料為空")
        return
        
    # 確保批次大小合適
    if batch_size > len(x):
        batch_size = len(x)
        print(f"警告: 批次大小已調整為 {batch_size} 以適應數據大小")
        
    # 取得特殊標記ID
    start_token = vocabulary['START']
    end_token = vocabulary['END']
    pad_token = vocabulary['PAD']
    ignored_tokens = [start_token, end_token, pad_token]
    vocab_size = len(vocabulary)
    
    # 預訓練生成器
    log = openLog()
    log.write("###### 開始預訓練生成器: {}\n".format(datetime.now()))
    log.close()
    
    lstm_model, _ = pretrain_thread_generator(
        train_x=x,
        sentence_lengths=torch.tensor(sentence_lengths, device=DEVICE).long(),
        batch_size=batch_size,
        end_token=end_token,
        vocab_size=vocab_size
    )
    
    # 創建生成器模型
    generator = ThreadGenerator(
        pretrain_model=lstm_model,
        start_token=start_token,
        ignored_tokens=ignored_tokens
    )
    generator.to(DEVICE)
    
    # 生成樣本用於訓練判別器
    log = openLog()
    log.write("###### 生成虛假樣本: {}\n".format(datetime.now()))
    log.close()
    
    x_gen = generator.generate(
        start_token=start_token,
        ignored_tokens=ignored_tokens,
        batch_size=len(x)
    )
    
    # 預訓練判別器
    log = openLog()
    log.write("###### 開始預訓練判別器: {}\n".format(datetime.now()))
    log.close()
    
    # 準備標籤：1為真實，0為生成
    y_real = gen_label(len(x), fixed_value=1)
    y_fake = gen_label(len(x_gen), fixed_value=0)
    
    # 合併訓練數據
    x_train = torch.cat([x.int(), x_gen.int()], dim=0)
    y_train = torch.cat([y_real, y_fake], dim=0)
    
    # 訓練判別器
    discriminator = train_discriminator(x_train, y_train, batch_size, vocab_size)
    
    # 初始化rollout
    log = openLog()
    log.write("###### 初始化rollout模型: {}\n".format(datetime.now()))
    log.close()
    
    rollout = Rollout(generator, r_update_rate=0.8)
    rollout = torch.nn.DataParallel(rollout)
    rollout.to(DEVICE)
    
    # 對抗訓練
    log = openLog()
    log.write("###### 開始對抗訓練: {}\n".format(datetime.now()))
    log.close()
    
    for total_batch in range(TOTAL_BATCH):
        log = openLog()
        log.write('批次: {} : {}\n'.format(total_batch, datetime.now()))
        print('批次: {} : {}\n'.format(total_batch, datetime.now()))
        log.close()
        
        # 生成樣本
        samples = generator.generate(
            start_token=start_token,
            ignored_tokens=ignored_tokens,
            batch_size=batch_size
        )
        
        # 透過rollout計算獎勵
        rewards = getReward(samples, rollout, discriminator)
        
        # 訓練生成器
        generator, _, _ = train_thread_generator(
            model=generator,
            x=samples,
            reward=rewards,
            iter_n_gen=1,
            batch_size=batch_size,
            sentence_lengths=sentence_lengths[:batch_size] if len(sentence_lengths) >= batch_size else None
        )
        
        # 更新rollout參數
        rollout.module.update_params(generator)
        
        # 訓練判別器
        for iter_n_dis in range(DIS_NUM_EPOCH):
            log = openLog()
            log.write('  判別器迭代: {} : {}\n'.format(iter_n_dis, datetime.now()))
            log.close()
            
            # 生成新的虛假樣本
            x_gen = generator.generate(
                start_token=start_token,
                ignored_tokens=ignored_tokens,
                batch_size=len(x)
            )
            
            # 準備新的訓練數據
            y_real = gen_label(len(x), fixed_value=1)
            y_fake = gen_label(len(x_gen), fixed_value=0)
            
            x_train = torch.cat([x.int(), x_gen.int()], dim=0)
            y_train = torch.cat([y_real, y_fake], dim=0)
            
            # 重新訓練判別器
            discriminator = train_discriminator(x_train, y_train, batch_size, vocab_size)
    
    # 訓練完成
    log = openLog()
    log.write('###### 訓練完成: {}\n'.format(datetime.now()))
    log.close()
    
    # 保存模型和詞彙表
    torch.save(reverse_vocab, PATH+'thread_reverse_vocab.pkl')
    try:
        torch.save(generator, PATH+'thread_generator.pkl')
        print('成功保存生成器模型.')
    except Exception as e:
        print(f'錯誤: 模型保存失敗! {e}')
    
    # 生成最終樣本
    log = openLog('thread_generated.txt')
    log.write("生成的討論回應樣本:\n\n")
    
    num_samples = min(10, batch_size)
    generated = generator.generate(
        start_token=start_token,
        ignored_tokens=ignored_tokens,
        batch_size=num_samples
    )
    
    decoded = decode_thread(generated, reverse_vocab, log)
    log.close()
    
    print("\n生成的討論回應樣本:")
    for i, text in enumerate(decoded):
        print(f"{i+1}. {text[:100]}...")
    
    return decoded

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
        
    try:
        input_file = sys.argv[2]
    except IndexError:
        input_file = 'ptt_real.csv'
        
    try:
        num = int(sys.argv[3])
    except IndexError:
        num = None
        
    # 如果可用的GPU數量大於0，確保批次大小至少與GPU數量相同
    if NrGPU > 0 and batch_size < NrGPU:
        batch_size = NrGPU
        print(f"批次大小已調整為 {batch_size} 以匹配GPU數量")
        
    results = main(batch_size, input_file, num)