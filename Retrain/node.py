# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:33:13 2019

@author: dawun
"""

import tensorflow as tf
gf=tf.GraphDef()
gf.ParseFromString(open('C:\\tmp\\retrained_graph.pb','rb').read())
print([n.name + '=>' +  n.op for n in gf.node if n.op])
