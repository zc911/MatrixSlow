
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:12:15 CST 2019

@author: chenzhen
"""


class ClassMining(object):
    @classmethod
    def get_subclass_list(cls, model):
        subclass_list = []
        for subclass in model.__subclasses__():
            subclass_list.append(subclass)
            subclass_list.extend(cls.get_subclass_list(subclass))
        return subclass_list

    @classmethod
    def get_subclass_dict(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return {k: k.__name__ for k in subclass_list}

    @classmethod
    def get_subclass_names(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return [k.__name__ for k in subclass_list]

    @classmethod
    def get_instance_by_subclass_name(cls, model, name):
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.get_instance_by_subclass_name(subclass, name)
            if instance:
                return instance
