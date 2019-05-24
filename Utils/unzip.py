# -*- coding: utf-8 -*-
"""
Created on Feb 2019

@author: Simon Plumecocq

Extract the zip file
"""

import os
import zipfile

main_path = "C:\\Users\\Plumecocq\\Documents\\Python Scripts\\ProjetAcousticSceneClassification"
os.chdir(main_path)

data_path = "data"
unzip_path = "unzip"

for index_file in range(1,14):
    print("Unzip file:"+str(index_file))
    zip_ref = zipfile.ZipFile(data_path+"\TUT-urban-acoustic-scenes-2018-development.audio."+str(index_file)+".zip", 'r')
    zip_ref.extractall(data_path+"\\"+unzip_path)
    zip_ref.close()
