import glob,re,os,sys
import numpy as np
from functools import partial
import time,json
import itertools
sys.path.append('/home/rm335/repos/sharedposstpNov19/utilsGRF')
sys.path.append('../bin')
from PolAB_A_allpars import interface_GRF_PolAB_A_A as GRFA
import BoundaryFinder as BF

step=0.025
col_ar=np.arange(-5,10+step,step)
row_ar=np.arange(-5,10+step,step)
#col_ar=np.logspace(np.log10(0.1),np.log10(2**20),300)
#row_ar=col_ar

def return_fullparset(parset,case):
    if case=="any":
        parset2=parset.copy()
    elif case=="difAD_difbnp":
        parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[14:16],parset[14:16],parset[14:16]))
    elif case=="difADsbnp" or case=="difAD_samebnp":
        parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14]))
    elif case=="difAD_samebnp_step12":#ni,ia
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step13": #ni,an
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step23":#ia,an
        parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step11":
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[5:6],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step22":
        parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="sameAD_difbp":
        parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:20]))
    elif case=="sameAD_difbp_kuonly":
        kb=parset[8]
        ku1,ku2,ku3,ku4,ku5,ku6=parset[9:]
        bindingar=np.array([kb,ku1,kb,ku2,kb,ku3,kb,ku4,kb,ku5,kb,ku6])
        parset2=np.concatenate((parset[0:8],parset[4:8],bindingar))
    elif case=="sameAD_difbnp":
        parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:10],parset[8:10],parset[8:10],parset[10:12],parset[10:12],parset[10:12]))
    elif case=="empty":
         parset2=np.concatenate((parset[0:8],parset[0:4],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10]))
    else:
        print("unrecognised case, ", case)
        raise ValueError
    return parset2

def get_constraints_npars(case,fcd=0.01,fcu=100):
    if case=="any":
        npars=24
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
        
    elif case=="difAD_difbnp":
        npars=16
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="difADsbnp" or case=="difAD_samebnp":
        npars=14
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14]))
    elif case=="difAD_samebnp_step12":#ni,ia
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':0,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step13": #ni,an
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step23":#ia,an
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu}}
        npars=8
        #parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step11":
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[5:6],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step22":
        npars=8
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':0,'fcd':1,'fcu':fcu}}
    elif case=="sameAD_difbp":
        npars=20
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:20]))
    elif case=="sameAD_difbp_kuonly":
        #kb=parset[8]
        #ku1,ku2,ku3,ku4,ku5,ku6=parset[9:]
        #bindingar=np.array([kb,ku1,kb,ku2,kb,ku3,kb,ku4,kb,ku5,kb,ku6])
        #parset2=np.concatenate((parset[0:8],parset[4:8],bindingar))
        npars=15
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="sameAD_difbnp":
        npars=12
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:10],parset[8:10],parset[8:10],parset[10:12],parset[10:12],parset[10:12]))
    elif case=="empty":
        npars=10
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}} 
        #parset2=np.concatenate((parset[0:8],parset[0:4],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10]))
    else:
        print("unrecognised case, ", case)
        raise ValueError
    return [constraints,npars]


def compute_syn_2fc(parset,fc1=10,fc2=5,case=None):
    mstars=[]
    parset2=return_fullparset(parset,case)
    
    for i in range(4):
        if i==0:
            A=0.0
            B=0.0
        elif i==1:
            A=2.0
            B=0.0
        elif i==2:
            A=0.0
            B=2.0
        else:
            A=1.0
            B=1.0
        m=GRFA(parset2.copy(),np.array([B]),A) #in some cases I have observed weird behaviour if the array is passed multiple times so it is safest to copy. 
        mstars.append(m)
    if mstars[1]>mstars[2]:
        r=(mstars[1]/mstars[2])
    else:
        r=(mstars[2]/mstars[1])
    if mstars[1]/mstars[0]>fc1 or mstars[2]/mstars[0]>fc1 or r>fc2:
        result=[None,None]
    else:
        if mstars[1]>=mstars[2]:
            result=[np.log2(mstars[3]/mstars[1]),np.log2(mstars[3]/mstars[2])]
        else:#swap A and B
            result=[np.log2(mstars[3]/mstars[2]),np.log2(mstars[3]/mstars[1])]

    return result

jid=int(sys.argv[1])-1
cases=["difAD_difbnp","difAD_samebnp","difAD_samebnp_step12","difAD_samebnp_step13","difAD_samebnp_step23","difAD_samebnp_step11","difAD_samebnp_step22"]
#parslimit=[[-2,2],[-3,3],[-4,4]]
extremesu=[[-2,2],[-1.5,1.5],[-1,1]]
prob_par=[0.2,0.5]
prob_replace=[0.2,0.6]
foldchanges2f=[1]
foldchanges1=[10,5]
combination=list(itertools.product(cases,foldchanges1,foldchanges2f,extremesu,prob_par,prob_replace))[jid]
case,fc1,fc2f,extr_uniform,prob_par,prob_replace=combination
parslimit=[0,4]
fc2=fc1*fc2f
min_,max_=parslimit
fcd=0.01
fcu=100
constraints,npars=get_constraints_npars(case,fcd=fcd,fcu=fcu)
constraintsb={0:{"min":1,"max":100},1:{"min":1,"max":100},2:{"min":100,"max":10000},3:{"min":1,"max":100}}
constraints.update(constraintsb)
myfunc=partial(compute_syn_2fc,fc1=fc1,fc2=fc2,case=case)
myfunc.__name__="%s_fc1=%g_fc2=%g"%(case,fc1,fc2)
settings={'pars_limit':[10**min_,10**max_],
          'constraints':constraints, 
          'compute_x_y_f':myfunc,
          'npars':npars,
          'row_ar':row_ar,
          'col_ar':col_ar,
          'seed':jid,
         'mat':None, #np.load('2019_08_06_matallrev.npy'),
         'mat_pars':None} #np.load('2019_08_06_matparsallrev.npy')}





niters_conv=3000
niters=70000
L=10

name_save='%s_fcd=%g_fcu=%g_fc1=%d_fc2=%d'%(case,fcd,fcu,fc1,fc2)
#name_save="kinsyndifADsbnp"
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path2=dir_path.replace("/home","/n/scratch3/users/r")
outfolder=os.path.join(dir_path2,name_save+'_out_%d'%jid)

if not os.path.isdir(outfolder):
    os.mkdir(outfolder)


outfolder_final='final_results'

args={'niters':niters,
      'niters_conv':niters_conv,
      'niters_conv_points':1000,
      'niters_save':10,
      'folder_save':outfolder,
       'name_save':name_save, 
      'prob_par':prob_par,
      'prob_replace':prob_replace,
      'extr_uniform':extr_uniform,
      'L_project':L,
      'plotting':False,
      'verbose':False,
     'dopulltangents':True} #

def function_tostring(x):
    if isinstance(x, np.ndarray):
        return ','.join(map(str,x))
    else:
        return x.__name__

outfnames=[os.path.join(outfolder,name_save+'_%d.sett'%jid),os.path.join(outfolder_final,name_save+'_%d.sett'%jid)]
for fname in outfnames:
    outf=open(fname,'w')
    #outf.write(time.ctime()+'\n')
    #with open(outf, 'w') as file:
    json.dump(dict({'time':time.ctime()},**settings),outf,default=function_tostring) # use `json.loads` to do the reverse
    outf.close()

    outf=open(fname.replace('.sett','.args'),'w')
    #outf.write(time.ctime()+'\n')
    #with open(outf, 'w') as file:
    json.dump(dict({'time':time.ctime()},**args),outf) # use `json.loads` to do the reverse
    outf.close()

BE=BF.BoundaryExplorer(**settings)
if settings['mat'] is None:
    BE.get_initial_points(10)
ti=time.time()
BE.extend_boundary(**args)
name='%s_%d_last'%(name_save,jid)
np.save(os.path.join(outfolder_final,'mat_'+name+'.npy'),BE.mat)
np.save(os.path.join(outfolder_final,'mat_pars_'+name+'.npy'),BE.mat_pars)
te=time.time()
print('time difference',te-ti)
print(BE.converged)
