# -*- coding: utf-8 -*-
"""
Rollout模組 - 用於SeqGAN的策略梯度計算
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (SEQ_LENGTH, EMB_SIZE, GEN_HIDDEN_DIM,
                    ROLLOUT_ITER, DEVICE, openLog)
import copy

class Rollout(nn.Module):
    def __init__(self, generator=None, r_update_rate=0.8, vocab_size=10):
        super().__init__()
        if generator is not None:
            self.ignored_tokens = generator.ignored_tokens
            self.embedding = generator.pretrain_model.module.embedding
            self.lstm = copy.deepcopy(generator.pretrain_model.module.lstm)
            self.hidden2tag = copy.deepcopy(generator.pretrain_model.module.hidden2tag)
            self.init_hidden = generator.pretrain_model.module.init_hidden
            self.logSoftmax = generator.pretrain_model.module.logSoftmax
            self.ignoreTokens = generator.ignoreTokens
            self.vocab_size = generator.pretrain_model.module.vocab_size
        else:
            from thread_lstm_generator import ThreadLSTMGenerator
            lstm = ThreadLSTMGenerator(vocab_size=vocab_size)
            self.ignored_tokens = None
            self.embedding = lstm.embedding
            self.lstm = lstm.lstm
            self.hidden2tag = lstm.hidden2tag
            self.init_hidden = lstm.init_hidden
            self.logSoftmax = lstm.logSoftmax
            self.vocab_size = vocab_size
            self.ignoreTokens = lambda x, y: x

        self.lstmCell = nn.LSTMCell(EMB_SIZE, GEN_HIDDEN_DIM)
        self.softmax = nn.Softmax(dim=-1)
        self.r_update_rate = r_update_rate

    def forward(self, sentence, hidden, given_num, ignored_tokens=None):
        assert given_num < sentence.shape[-1]
        if ignored_tokens is None:
            ignored_tokens = self.ignored_tokens
        if len(sentence.shape) == 1:
            sentence = sentence.view(1, sentence.shape[0])
        # 已知輸入部分（前given_num個標記）
        existing = sentence[:, 0:given_num]
        embeds_existing = self.embedding(existing.long())
        hidden0 = [x.permute(1, 0, 2).contiguous() for x in hidden]
        self.lstm_out, hidden0 = self.lstm(embeds_existing, hidden0)
        self.tag_space = self.hidden2tag(self.lstm_out)

        self.y_prob = self.softmax(self.ignoreTokens(self.tag_space, ignored_tokens))
        shape = (self.y_prob.shape[0], self.y_prob.shape[1])
        y_prob_existing_output = self.y_prob.view(-1, self.y_prob.shape[-1]).multinomial(num_samples=1).view(shape)
        y_pred_output = sentence[:, 0:given_num]

        # 未知部分要rollout，以估計總體獎勵
        x_t = y_prob_existing_output[:, -1].view(-1, 1)
        # 從前一個nn.LSTM中解包隱藏狀態和單元狀態，並為nn.LSTMCell重新整形:
        hidden_state, cell_state = hidden0
        hidden_state = hidden_state.view(-1, GEN_HIDDEN_DIM)
        cell_state = cell_state.view(-1, GEN_HIDDEN_DIM)
        hidden0 = (hidden_state, cell_state)
        for i in range(given_num, SEQ_LENGTH):
            embeds_rollout = self.embedding(x_t.long()).view(-1, EMB_SIZE)
            hidden0 = self.lstmCell(embeds_rollout, hidden0)
            tag_space_rollout = self.hidden2tag(hidden0[0])
            y_prob_rollout = self.softmax(self.ignoreTokens(tag_space_rollout, ignored_tokens))
            x_t = y_prob_rollout.multinomial(num_samples=1)
            self.lstm_out = torch.cat((self.lstm_out, hidden0[0].view(-1, 1, hidden0[0].shape[-1])), dim=1)
            self.tag_space = torch.cat((self.tag_space, tag_space_rollout.view(-1, 1, tag_space_rollout.shape[-1])), dim=1)
            self.y_prob = torch.cat((self.y_prob, y_prob_rollout.view(-1, 1, y_prob_rollout.shape[-1])), dim=1)
            y_pred_output = torch.cat((y_pred_output.int(), x_t.int()), dim=1)
        
        tag_scores = self.logSoftmax(self.tag_space)
        return tag_scores, y_pred_output

    def update_params(self, generator):
        for p, w in zip(self.lstm.parameters(), generator.pretrain_model.module.lstm.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data
        for p, w in zip(self.hidden2tag.parameters(), generator.pretrain_model.module.hidden2tag.parameters()):
            p.data = self.r_update_rate * p.data + (1-self.r_update_rate) * w.data

def getReward(gen_output, rollout, discriminator):
    with torch.no_grad():
        gen_output = gen_output.view(-1, SEQ_LENGTH)
        batch_size = len(gen_output)
        rewards = torch.zeros(batch_size, SEQ_LENGTH, device=DEVICE)
        # "ROLLOUT_ITER"確定rollout網絡重複運行的次數
        for i in range(ROLLOUT_ITER):
            # 前given_num個詞與輸入x相同
            # 最後(sequence_length-given_num)個詞由lstm網絡生成
            for given_num in range(1, SEQ_LENGTH):
                hidden = rollout.module.init_hidden(batch_size)
                tag_scores, rollout_output = rollout(sentence=gen_output, hidden=hidden, given_num=given_num)
                dis_output = discriminator(rollout_output)
                ypred = [item[1] for item in dis_output]
                # 每個given_num更新一列，每個i在ROLLOUT_ITER中更新整個表
                rewards[:, given_num-1] += torch.tensor(ypred, device=DEVICE)
            # 最後一個標記的獎勵
            dis_output = discriminator(gen_output)
            ypred = [item[1] for item in dis_output]
            rewards[:, SEQ_LENGTH-1] += torch.tensor(ypred, device=DEVICE)
        rewards = rewards / (1.0 * ROLLOUT_ITER)
    return rewards

def sanityCheck_rollout(batch_size=5):
    '''測試Rollout實例化'''
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_rollout: {}'.format(datetime.now()))
    from data_processing_threads import read_discussion_csv
    x, _, reverse_vocab, _ = read_discussion_csv()
    x0 = x[0:batch_size]
    try:
        model = Rollout(vocab_size=len(reverse_vocab))
        model = nn.DataParallel(model)
        model.to(DEVICE)
        hidden = model.module.init_hidden(len(x0))
        model(x0, hidden, given_num=3)
        log.write('\n  rollout.sanityCheck_rollout 成功: {}\n'.format(datetime.now()))
        log.close()
        return model
    except Exception as e:
        log.write('\n  rollout.sanityCheck_rollout 失敗! : {}\n'.format(datetime.now()))
        log.write(str(e) + '\n')
        log.close()
        return None

if __name__ == '__main__':
    rollout = sanityCheck_rollout(batch_size=5)