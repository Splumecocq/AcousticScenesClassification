# -*- coding: utf-8 -*-
"""
Created on Apr 2019

@author: Plumecocq


"""
import os
import torch

def save_model(modele_name="CNN", file_name=""):
    save_path = "Result"
    result_path = save_path + "\\" + modele_name
    #test if the directory already exist else create this
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    state = {
        'epoch': epoch,
        'state_dict': cnn.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, result_path + "\\state_"+file_name+".pt")
    torch.save(cnn, result_path + "\\save_"+file_name+".pt")

'''
#to load :
result_path = "Result\\" + modele_name
state = torch.load(result_path + "\\state_"+file_name+".pt")
cnn.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])

#or:
cnn = torch.load(result_path + "\\save_"+file_name+".pt")
cnn = cnn.eval()
'''