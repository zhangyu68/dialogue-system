from modules.entites import EntityTracker
from modules.bow import BoW_encoder
from  modules.lstm_net import LSTM_net
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data
import modules.util as util

import numpy as np
import sys


class Trainer():

    def __init__(self):
        import os
        #实体追踪
        et = EntityTracker()
        #词袋 word2vec
        self.bow_enc = BoW_encoder()
        #加载word2vec embedding
        self.emb = UtteranceEmbed()
        #将实体追踪器添加到动作追踪器中
        at = ActionTracker(et)
        #得到数据集和对话开始 结束行数
        self.dataset, dialog_indices = Data(et, at).trainset
        #划分数据集：200对做训练 50对做测试
        self.dialog_indices_tr = dialog_indices
        self.dialog_indices_dev = dialog_indices
        #obs_size 300维的词向量 + 85个袋中的词 + 4个槽位
        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features

        #话术模板
        self.action_templates = at.get_action_templates()
        #动作个数
        action_size = at.action_size
        #隐藏层神经元个数
        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)


    def train(self):

        print('\n:: training started\n')
        epochs = 25
        for j in range(epochs):
            # iterate through dialogs
            #训练集个数
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.
            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                # get start and end index
                start, end = dialog_idx['start'], dialog_idx['end']
                # train on dialogue
                loss += self.dialog_train(self.dataset[start:end])
                # print #iteration
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, num_tr_examples))

            print('\n\n:: {}.tr loss {}'.format(j+1, loss/num_tr_examples))
            # evaluate every epoch
            accuracy = self.evaluate()
            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))

            if accuracy > 0.4:
                self.net.save()
                continue

    #训练过程
    def dialog_train(self, dialog):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        loss = 0.
        # iterate through dialog
        #u 用户输入 r 对应的动作索引
        for (u,r) in dialog:
            #u_ent 分词后的字符串
            u_ent = et.extract_entities(u)
            #槽位填充情况 【0 0 0 0】
            u_ent_features = et.context_features()
            #word2vec
            u_emb = self.emb.encode(u)
            #multi-hot
            u_bow = self.bow_enc.encode(u)
            # concat features
            #300 + 85 + 4 = 389
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            # get action mask
            action_mask = at.action_mask()
            # forward propagation
            #  train step
            loss += self.net.train_step(features, r, action_mask)
        return loss/len(dialog)

    #评估acc
    def evaluate(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        dialog_accuracy = 0.
        #加载测试集
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net.reset_state()

            # iterate through dialog
            correct_examples = 0
            #对于每个dialog 提取出utterance 和 response
            for (u,r) in dialog:
                # encode utterance
                #提取出user中带有的实体
                u_ent = et.extract_entities(u)
                #提取当前槽位填充情况
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask 16维的multi-hot 向量
                action_mask = at.action_mask()
                # forward propagation
                #  train step
                prediction = self.net.forward(features, action_mask)
                correct_examples += int(prediction == r)
            # get dialog accuracy
            dialog_accuracy += correct_examples/len(dialog)

        return dialog_accuracy/num_dev_examples



if __name__ == '__main__':
    # setup trainer
    trainer = Trainer()
    # start training
    trainer.train()
