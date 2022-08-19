import os
import shutil
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path2=dir_path.replace("/home","/n/scratch2")
if not os.path.isdir(dir_path2):
    os.makedirs(dir_path2,exist_ok=True)
pyfiles=glob.glob("./*.py")
shfiles=glob.glob("./*.sh")
allfiles=pyfiles+shfiles
for f in allfiles:
    shutil.copy(f,os.path.join(dir_path2,f))


