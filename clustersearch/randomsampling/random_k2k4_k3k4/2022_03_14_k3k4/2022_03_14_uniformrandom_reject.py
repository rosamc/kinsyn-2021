import numpy as np
import pandas as pd
import sys, os, time
import json
sys.path.append('/home/rm335/repos/sharedposstpNov19/utilsGRF')
sys.path.append("/home/rm335/repos/myrepos/funcsbPcycle")
sys.path.append("../../bin")
import basic
import PolAB_A_allpars
import itertools

npts=10*10**6

printpoints=[int(x) for x in np.logspace(3,np.log10(npts),20)]

SABar=np.arange(-5,5,0.025)
SBAar=np.arange(-5,5,0.025)
grid=np.zeros((len(SBAar),len(SABar)))

jid=int(sys.argv[1])-1

constraints=[0,4,3]
fchanges=[10,10]
seed=jid

print(constraints)
np.random.seed(seed)
minb,maxb,parfc=constraints
fc1,fc2=fchanges
outlist=[]
prevname=None
npt=0
ntrials=0
while npt<npts:
    ntrials+=1
    ktia0,ktan0,ktin0,ktni0,ktinA,ktanB=np.random.uniform(0,4,size=6)
    b_u=np.random.uniform(0,4,size=2)
    cont=True
    for par in [ktia0,ktan0,ktni0]:
        if par<minb or par>maxb:
            cont=False

    if cont:
        if ktin0<4-(maxb-minb):
            cont=False

    if cont:
        if parfc is not None:
            min_,max_=[ktia0,min(4,parfc+ktia0)]
            if ktanB<min_ or ktanB>max_:
                cont=False

            min_,max_=[max(0,ktin0-parfc),ktin0]
            if ktinA<min_ or ktinA>max_ :
                cont=False
        else:
            print("not prepared for parfc None")


    if cont:
        npt+=1
    
        parvals=np.concatenate((np.array([ktia0,ktan0,ktin0,ktni0,ktia0,ktan0,ktinA,ktni0,ktia0,ktanB,ktin0,ktni0]),np.tile(b_u,6)))
        parvaluesar=10**parvals
        out=basic.compute_synergy(parvaluesar,f=PolAB_A_allpars.interface_GRF_PolAB_A_A,ftype='crit',returnm=True,fcind=fc1,fcpair=fc2,anystronger=True) #I first run this without limiting this
        if out[0] is not None:
            m=out[-1]
            #print(m)
            m0,mA,mB,mAB=m
            if mB>mA:
                mA_=mB
                mB=mA
                mA=mA_
                SAB=out[1]
                SBA=out[0]
                #since A and B are each doing a particular thing, keep parameter set as such
                #parvaluesar=np.concatenate((parvaluesar[0:4],parvaluesar[8:12],parvaluesar[4:8],parvaluesar[12:]))
            else:
                SAB=out[0]
                SBA=out[1]
            if SAB>SABar[0] and SAB<SABar[-1] and SBA>SBAar[0] and SBA<SBAar[-1]:
                xc=np.where(SAB>SABar)[0][-1]
                yc=np.where(SBA>SBAar)[0][-1]
                if grid[yc,xc]<1:
                    grid[yc,xc]+=1
                    outlist.append([parvaluesar,SAB,SBA,m0,mA,mB,mAB])
        if npt in printpoints:
            if prevname is not None:
                os.remove(prevname)
            allvalues=np.zeros((len(outlist),31))
            for r,row in enumerate(outlist):
                allvalues[r,0:24]=row[0]
                allvalues[r,24:30]=np.array(row[1:])

            colnames=['p%d'%(i+1) for i in range(24)]
            df=pd.DataFrame(allvalues,columns=colnames+['SAB','SBA','m0','mA','mB','mAB',"quadrant"])
            quadrant=np.zeros(len(df))
            sabp=(df['SAB'].values>0)
            sbap=(df['SBA'].values>0)
            quadrant[sabp&sbap]=1
            quadrant[(~sabp)&sbap]=2
            quadrant[(~sabp)&(~sbap)]=3
            df["quadrant"]=pd.Series(quadrant,dtype='int')
            name='2022_03_14_randompoints_nit=%d_minb=%g_maxb=%g_parfc=%s_fc1=%s_fc2=%s_seed=%d.df'%(npt,minb,maxb,parfc,fc1,fc2,seed)
            name=os.path.join('final_results',name)
            df.to_csv(name,index=False)
            prevname=name

        if ntrials%1000==0:
            print(ntrials,npt)
            sys.stdout.flush()
        


