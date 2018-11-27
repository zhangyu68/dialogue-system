'''
actions.py 定义了系统可执行的动作
定义了动作模板
'''

import modules.util as util
import numpy as np

'''
    Action Templates
    1 'Are you referring to this question <sub_intent> ?'
    2 'Happy to help, do you have any other questions?'
    3 'Hi! I\\'m Moli, your virtual agent. How can I help you?'
    4 'I\\'m sorry,you can transfer a agent'
    5 'Is the solution helpful?'
    6 '[Answer]'
    7 bye
    8 'can you speak clearly?'
    9 'ok,you can ask me when you need'
    10 'you are welcome'
    
    

    [1] : sub_intent

'''


# 动作追踪
class ActionTracker():

    def __init__(self, ent_tracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = {
            '0': [1,2,3,4,7,8,9,10],
            '1': [1,2,3,4,5,6,7,8,9,10],
        }

    def action_mask(self):
        # get context features as string of ints (0/1)
        # 将当前槽值转为字符串
        ctxt_f = ''.join([str(flag) for flag in self.et.context_features().astype(np.int32)])

        # action mask 9维
        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index - 1] = 1.
            return self.am

        return construct_mask(ctxt_f)

    def get_action_templates(self):
        responses = list(set([self.et.extract_entities(response, update=False)
                              for response in util.get_responses()]))

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'resto_' in word:
                    if 'phone' in word:
                        template.append('<info_phone>')
                    elif 'address' in word:
                        template.append('<info_address>')
                    else:
                        template.append('<restaurant>')
                else:
                    template.append(word)
            return ' '.join(template)

        # extract restaurant entities
        return sorted(set([extract_(response) for response in responses]))
