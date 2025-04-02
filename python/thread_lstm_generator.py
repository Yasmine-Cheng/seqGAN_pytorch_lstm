# -*- coding: utf-8 -*-
"""
基於LSTM的討論串生成器
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SEQ_LENGTH, EMB_SIZE, DEVICE, GEN_HIDDEN_DIM, GEN_NUM_EPOCH_PRETRAIN, openLog
from data_processing_threads import gen_discussion_data

class ThreadLSTMGenerator(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE)
        self.lstm = nn.LSTM(EMB_SIZE, GEN_HIDDEN_DIM, batch_first=True)
        self.hidden2tag = nn.Linear(GEN_HIDDEN_DIM, vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=2)
    
    def init_hidden(self, batch_size=1):
        return (torch.empty(batch_size, 1, GEN_HIDDEN_DIM, device=DEVICE).normal_(),
                torch.empty(batch_size, 1, GEN_HIDDEN_DIM, device=DEVICE).normal_())
    
    def forward(self, sentence, hidden, sentence_lengths=None):
        if len(sentence.shape) == 1:
            sentence = sentence.view(1, sentence.shape[0])
        
        if sentence_lengths is None:
            sentence_lengths = torch.LongTensor([sentence.shape[1]] * len(sentence))
        
        sentence_lengths = sentence_lengths.type(torch.LongTensor)
        if len(sentence_lengths) < len(sentence):
            sentence_lengths = torch.cat([sentence_lengths, torch.LongTensor([sentence.shape[1]]
                                    * (len(sentence)-len(sentence_lengths)))])
                                    
        embeds = self.embedding(sentence.long())
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_lengths.to(torch.device('cpu')), batch_first=True)
        
        hidden0 = [x.permute(1,0,2).contiguous() for x in hidden]
        lstm_out, hidden0 = self.lstm(embeds, hidden0)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=sentence.shape[1])
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.logSoftmax(tag_space)
        
        return tag_scores, tag_space

def pretrain_thread_generator(train_x=None, sentence_lengths=None, batch_size=1, end_token=None, vocab_size=10):
    """預訓練討論串LSTM生成器"""
    if train_x is None:
        x = gen_discussion_data(vocab_size=vocab_size)
    else:
        x = train_x
        
    if len(x.shape) == 1:
        x = x.view(1, x.shape[0])
        
    if sentence_lengths is None:
        sentence_lengths = [x.shape[1]] * len(x)
        
    if len(sentence_lengths) < len(x):
        sentence_lengths.extend([x.shape[1]] * (len(x) - len(sentence_lengths)))
        
    if end_token is None:
        end_token = vocab_size - 1
    
    model = ThreadLSTMGenerator(vocab_size)
    model = nn.DataParallel(model)
    model.to(DEVICE)
    
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    y_pred_all = []
    
    log = openLog()
    log.write('    training thread LSTM generator: {}\n'.format(datetime.now()))
    
    for epoch in range(GEN_NUM_EPOCH_PRETRAIN):
        pointer = 0
        y_pred_all = []
        epoch_loss = []
        
        while pointer + batch_size <= len(x):
            x_batch = x[pointer:pointer+batch_size]
            x0_length = torch.tensor(sentence_lengths[pointer:pointer+batch_size]).to(device=DEVICE)
            
            # 目標是移位輸入（下一個標記預測）
            y = torch.cat((x_batch[:,1:],
                           torch.tensor([end_token]*x_batch.shape[0],device=DEVICE).int().view(x_batch.shape[0],1)),
                           dim=1)
            
            hidden = model.module.init_hidden(batch_size)
            y_pred, tag_space = model(x_batch, hidden, x0_length)
            
            loss = criterion(y_pred.view(-1,y_pred.shape[-1]), y.long().view(-1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            y_prob = F.softmax(tag_space, dim=2)
            y_pred_all.append(y_prob)
            epoch_loss.append(loss.item())
            pointer = pointer + batch_size
            
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    
    log.close()
    return model, torch.cat(y_pred_all)

def generate_thread_response(model, start_token=0, batch_size=1, post_context=None):
    """根據帖子和討論串上下文生成回應"""
    with torch.no_grad():
        # 以提供的帖子上下文或START標記開始
        if post_context is not None:
            y = post_context
        else:
            y = [start_token] * batch_size
            
        y_all = torch.tensor(y, device=DEVICE).int().view(-1, 1)
        hidden = model.module.init_hidden(batch_size)
        
        # 生成序列
        for i in range(SEQ_LENGTH-1):
            x = torch.tensor(y, device=DEVICE).view([-1, 1])
            y_pred, tag_space = model(x, hidden, sentence_lengths=torch.tensor([1], device=DEVICE).long())
            
            # 基於概率進行抽樣
            y_prob = F.softmax(tag_space, dim=2)
            shape = (y_prob.shape[0], y_prob.shape[1])
            try:
                y = y_prob.view(-1, y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
            except RuntimeError:
                print('multinomial出錯，改用argmax')
                y = torch.argmax(y_prob.view(-1, y_prob.shape[-1]), dim=1).float().view(shape)
            
            y_all = torch.cat([y_all, y.int()], dim=1)
            
        return y_all

def test_genMaxSample(model, start_token=0, batch_size=1):
    """測試生成器的生成功能"""
    log = openLog('test.txt')
    log.write('\n\nTest thread_lstm_generator.test_genMaxSample: {}'.format(datetime.now()))
    with torch.no_grad():
        y = [start_token] * batch_size
        y_all_max = torch.tensor(y, device=DEVICE).int().view(-1, 1)
        hidden = model.module.init_hidden(len(y))
        for i in range(SEQ_LENGTH-1):
            x = torch.tensor(y, device=DEVICE).view([-1, 1])
            y_pred, _ = model(x, hidden, sentence_lengths=torch.tensor([1], device=DEVICE).long())
            y_pred = y_pred.squeeze(dim=1)
            # 取最大值
            y = torch.argmax(y_pred, dim=1).float().view(-1, 1)
            y_all_max = torch.cat([y_all_max, y.int()], dim=1)
        
        y = [start_token] * batch_size
        y_all_sample = torch.tensor(y, device=DEVICE).int().view(-1, 1)
        hidden = model.module.init_hidden(len(y))
        for i in range(SEQ_LENGTH-1):
            x = torch.tensor(y, device=DEVICE).view([-1, 1])
            y_pred, tag_space = model(x, hidden, sentence_lengths=torch.tensor([1], device=DEVICE).long())
            # 基於概率分佈進行隨機選擇
            y_prob = F.softmax(tag_space, dim=2)
            shape = (y_prob.shape[0], y_prob.shape[1])
            try:
                y = y_prob.view(-1, y_prob.shape[-1]).multinomial(num_samples=1).float().view(shape)
            except RuntimeError:
                print('multinomial出錯，改用argmax')
                y = torch.argmax(y_prob.view(-1, y_prob.shape[-1]), dim=1).float().view(shape)
            y_all_sample = torch.cat([y_all_sample, y.int()], dim=1)
    log.write('\n  thread_lstm_generator.test_genMaxSample 成功: {}\n'.format(datetime.now()))
    log.close()
    return y_all_max, y_all_sample