# -*- coding: utf-8 -*-
"""
討論串生成器包裝類
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SEQ_LENGTH, DEVICE, GEN_NUM_EPOCH, MAXINT, openLog
from thread_lstm_generator import ThreadLSTMGenerator, pretrain_thread_generator

class ThreadGenerator(nn.Module):
    def __init__(self, pretrain_model=None, start_token=0, ignored_tokens=None):
        super().__init__()
        self.start_token = start_token
        self.ignored_tokens = ignored_tokens
        if pretrain_model is None:
            from data_processing_threads import read_discussion_csv
            x, vocabulary, reverse_vocab, _ = read_discussion_csv()
            self.pretrain_model, _ = pretrain_thread_generator(train_x=x, vocab_size=len(reverse_vocab))
            self.vocabulary = vocabulary
        else:
            self.pretrain_model = pretrain_model       
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden, rewards, ignored_tokens=None, sentence_lengths=None):
        '''前向傳播，變量可以反向傳播'''
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        y, tag_space = self.pretrain_model(x, hidden, sentence_lengths=sentence_lengths)
        y = y.data
        y_pred = self.ignoreTokens(tag_space, ignored_tokens)
        y_prob = self.softmax(y_pred)
        shape = (y_prob.shape[0], y_prob.shape[1])
        try:
            y_output = y_prob.view(-1,y_prob.shape[-1]).multinomial(num_samples=1).view(shape)
        except RuntimeError:
            print('使用multinomial時出錯，改用argmax')
            y_output = torch.argmax(y_prob.view(-1,y_prob.shape[-1]), dim=1).view(shape)
        if rewards is None:
            rewards = y_prob.sum(dim=2).data
        loss_variable = self.loss(y_prob, x, rewards)
        return y_output, y_prob, loss_variable

    def generate(self, start_token=None, ignored_tokens=None, batch_size=1, post_context=None):
        '''生成唯讀樣本，不會反向傳播'''
        if start_token is None:
            start_token = self.start_token
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        
        if post_context is not None:
            # 使用提供的上下文生成
            return self.generate_with_context(post_context, ignored_tokens, batch_size)
        else:
            # 從頭開始生成
            return self.generate_from_scratch(start_token, ignored_tokens, batch_size)
    
    def generate_from_scratch(self, start_token, ignored_tokens, batch_size=1):
        '''從頭開始生成樣本'''
        y = [start_token] * batch_size
        y_all_sample = torch.tensor(y, device=DEVICE).int().view(-1, 1)
        with torch.no_grad():
            hidden = self.pretrain_model.module.init_hidden(len(y))
            for i in range(SEQ_LENGTH-1):
                x = torch.tensor(y, device=DEVICE).view([-1, 1])
                y_pred, tag_space = self.pretrain_model(x, hidden, sentence_lengths=torch.tensor([1], device=DEVICE).long())
                # 基於概率分佈進行隨機選擇
                y_prob = F.softmax(self.ignoreTokens(tag_space, ignored_tokens), dim=2)
                shape = (y_prob.shape[0], y_prob.shape[1])
                try:
                    y = y_prob.view(-1, y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
                except RuntimeError:
                    print('multinomial出錯，改用argmax')
                    y = torch.argmax(y_prob.view(-1, y_prob.shape[-1]), dim=1).float().view(shape)
                y_all_sample = torch.cat([y_all_sample, y.int()], dim=1)
        return y_all_sample
    
    def generate_with_context(self, context_ids, ignored_tokens, batch_size=1):
        '''根據上下文生成樣本'''
        with torch.no_grad():
            # 如果沒有提供上下文，使用起始標記
            if context_ids is None:
                return self.generate_from_scratch(self.start_token, ignored_tokens, batch_size)
            
            # 確保上下文是正確的形狀
            if isinstance(context_ids, list):
                context_ids = torch.tensor(context_ids, device=DEVICE).int().view(1, -1)
            
            # 複製上下文以匹配批次大小
            if context_ids.size(0) < batch_size:
                context_ids = context_ids.repeat(batch_size, 1)
                
            # 從上下文生成
            hidden = self.pretrain_model.module.init_hidden(batch_size)
            context_length = torch.tensor([context_ids.size(1)] * batch_size, device=DEVICE).long()
            _, tag_space = self.pretrain_model(context_ids, hidden, sentence_lengths=context_length)
            
            # 生成下一個標記
            y_prob = F.softmax(self.ignoreTokens(tag_space[:, -1:, :], ignored_tokens), dim=2)
            shape = (y_prob.shape[0], y_prob.shape[1])
            try:
                y = y_prob.view(-1, y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
            except RuntimeError:
                print('multinomial出錯，改用argmax')
                y = torch.argmax(y_prob.view(-1, y_prob.shape[-1]), dim=1).float().view(shape)
            
            # 繼續生成其餘的序列
            y_all_sample = torch.cat([context_ids, y.int()], dim=1)
            current_y = y.view(-1).tolist()
            
            for i in range(SEQ_LENGTH - context_ids.size(1) - 1):
                x = torch.tensor(current_y, device=DEVICE).view([-1, 1])
                y_pred, tag_space = self.pretrain_model(x, hidden, sentence_lengths=torch.tensor([1], device=DEVICE).long())
                
                y_prob = F.softmax(self.ignoreTokens(tag_space, ignored_tokens), dim=2)
                shape = (y_prob.shape[0], y_prob.shape[1])
                try:
                    y = y_prob.view(-1, y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
                except RuntimeError:
                    print('multinomial出錯，改用argmax')
                    y = torch.argmax(y_prob.view(-1, y_prob.shape[-1]), dim=1).float().view(shape)
                
                y_all_sample = torch.cat([y_all_sample, y.int()], dim=1)
                current_y = y.view(-1).tolist()
                
            return y_all_sample
    
    def ignoreTokens(self, original, ignored_tokens):
        '''避免選擇'START'或'END'標記的概率'''
        if ignored_tokens is None:
            return original
        for token in ignored_tokens:
            if len(original.shape)==3:
                original[:,:,token] = -MAXINT
            else:
                original[:,token] = -MAXINT
        return original

    def loss(self, prediction, x, rewards):
        '''計算損失'''
        x1 = x.view(-1,1).long()
        pred1 = prediction.view(-1,prediction.shape[-1])
        
        # 創建one-hot編碼
        x2 = torch.zeros((x1.shape[0], pred1.shape[1]), device=DEVICE).scatter_(1, x1, 1)
        
        pred2 = torch.log(torch.clamp(pred1, min=1e-20, max=1.0))
        prod = torch.mul(x2, pred2)
        reduced_prod = torch.sum(prod, dim=1)
        rewards_prod = torch.mul(reduced_prod, rewards.view(-1))
        generator_loss = -torch.sum(rewards_prod)
        return generator_loss

def train_thread_generator(model, x, reward, iter_n_gen=None, batch_size=1, sentence_lengths=None):
    """訓練討論串生成器"""
    if len(x.shape) == 1:
        x = x.view(1, x.shape[0])
    rem = len(x) % batch_size
    if rem > 0:
        x = x[0:len(x)-rem]
    if sentence_lengths is None:
        sentence_lengths = [x.shape[1]] * len(x)
    if len(sentence_lengths) < len(x):
        sentence_lengths.extend([x.shape[1]] 
                                * (len(x)-len(sentence_lengths)))
    sentence_lengths = torch.tensor(sentence_lengths, device=DEVICE).long()
    if reward is None:
        reward = torch.tensor([1.0] * x.shape[0] * x.shape[1], device=DEVICE).view(x.shape)
    if iter_n_gen is None:
        iter_n_gen = GEN_NUM_EPOCH
        
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(params, lr=0.001)
    log = openLog()
    log.write('    training thread generator: {}\n'.format(datetime.now()))
    for epoch in range(iter_n_gen):
        pointer = 0
        y_prob_all = []
        y_output_all = []
        epoch_loss = []
        while pointer + batch_size <= len(x):
            x_batch = x[pointer:pointer+batch_size]
            r_batch = reward[pointer:pointer+batch_size]
            s_length = sentence_lengths[pointer:pointer+batch_size]
            hidden = model.pretrain_model.module.init_hidden(batch_size)
            y_output, y_prob, loss_var = model(x=x_batch, hidden=hidden, rewards=r_batch, sentence_lengths=s_length)
            optimizer.zero_grad()
            loss_var.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            y_prob_all.append(y_prob)
            y_output_all.append(y_output)  
            epoch_loss.append(loss_var.item())
            pointer = pointer + batch_size
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    log.close()
    return (model, torch.cat(y_prob_all), torch.cat(y_output_all).view(list(x.shape)))