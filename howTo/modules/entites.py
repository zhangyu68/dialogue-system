'''
entites 主要功能槽值识别
'''

from enum import Enum
import numpy as np


class EntityTracker():

    def __init__(self):
        self.entities = {
                '<sub_intent>' : None,
                }
        self.num_features = 1 # tracking 4 entities
        self.rating = None

        # constants
        self.sub_intent = ['connect','turnoff']

        self.EntType = Enum('Entity Type', '<sub_intent>  <non_ent>')


    def ent_type(self, ent):
        if ent in self.sub_intent:
            print("entity recognize"+self.EntType['<sub_intent>'].name)
            return self.EntType['<sub_intent>'].name
        else:
            return ent

    #提取实体，判断类型 返回分词结果
    def extract_entities(self, utterance, update=True):
        tokenized = []
        for word in utterance.split(' '):
            entity = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = word

            tokenized.append(entity)

        return ' '.join(tokenized)


    def context_features(self):
       #1个槽位
       keys = list(set(self.entities.keys()))

       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys],
                                   dtype=np.float32 )
       return self.ctxt_features


    def action_mask(self):
        print('Not yet implemented. Need a list of action templates!')
