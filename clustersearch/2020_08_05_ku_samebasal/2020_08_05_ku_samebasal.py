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

npts1=100
npts2=10**3

tol=0.05

#[[0,2,2,None,None],[0,2,2,10,10],[0,2,2,10,5],[0,2,3,10,5],[0,4,None,10,5]]


jid=int(sys.argv[1])-1
np.random.seed(jid)
constraints=[0,4,3]
fchanges=[5,5]

allkufactors=np.logspace(-1,1,10)

minb,maxb,parfc=constraints
fc1,fc2=fchanges
outlist=[]
prevname=None
npt=0
ntrials=0
potential_classes=["nndd","pndd","ppdd","nndi","pndi","ppdi","nnid","npid","ppid","nnii","npii","ppii"]
outfile=open(os.path.join("results","20200813_npts1=%d_npts2=%d_minb=%d_maxb=%d_parfc=%d_fc1=%d_fc2=%d_%d.csv"%(npts1,npts2,minb,maxb,parfc,fc1,fc2,jid)),"w")
parnames=["p%d"%i for i in range(24)]
parnames=",".join(parnames)
synabnames=",".join(["synab_%g"%ku for ku in allkufactors])
synbanames=",".join(["synba_%g"%ku for ku in allkufactors])
outfile.write("npt1,npt2,class,%s,%s,%s\n"%(parnames,synabnames,synbanames))
for npt1 in range(npts1):
    ktia0,ktan0,ktin0,ktni0=np.random.uniform(0,4,size=4)
    b_u=np.random.uniform(1.5,3,size=2)
    npt2=0
    results=[]
    while npt2<npts2: #number of parameter sets in bounds and where I can change ku keeping A to be the strongest
        cont=True
        ktiaA,ktanA,ktinA,ktniA,ktiaB,ktanB,ktinB,ktniB=np.random.uniform(0,4,size=8)
        min_,max_=[ktia0,min(4,parfc+ktia0)]
        if ktiaA<min_ or ktiaA>max_:
            cont=False
        if ktiaB<min_ or ktiaB>max_:
            cont=False

        min_,max_=[ktan0,min(4,parfc+ktan0)]
        if ktanA<min_ or ktanA>max_:
            cont=False
        if ktanB<min_ or ktanB>max_:
            cont=False

        min_,max_=[ktni0,min(4,parfc+ktni0)]
        if ktniA<min_ or ktniA>max_:
            cont=False
        if ktniB<min_ or ktniB>max_:
            cont=False

        min_,max_=[max(0,ktin0-parfc),ktin0]
        if ktinA<min_ or ktinB<min_ or ktinA>max_ or ktinB>max_:
            cont=False
        if cont:
    
            parvals=np.concatenate((np.array([ktia0,ktan0,ktin0,ktni0,ktiaA,ktanA,ktinA,ktniA,ktiaB,ktanB,ktinB,ktniB]),np.tile(b_u,6)))
            fullpars=10**parvals

            synergies=[]
            cont2=True
            for i in range(len(allkufactors)):
                parset2=fullpars.copy()
                parset2[[13,15,17,19,21,23]]=parset2[[13,15,17,19,21,23]]*allkufactors[i]
                #print(parset2)
                syn0=basic.compute_synergy(parset2,f=PolAB_A_allpars.interface_GRF_PolAB_A_A,ftype='crit',fcind=5,fcpair=5,anystronger=False)
                if syn0[0]==None:
                    cont2=False
                    break
                else:
                    synergies.append(syn0)
             
            if cont2:
                npt2+=1
                synab=np.array([x[0] for x in synergies]) 
                synba=np.array([x[1] for x in synergies])
                
                if np.max(synab)-np.min(synab)>tol and np.max(synba)-np.min(synba)>tol:
                    SAB0=synab[0]
                    SAB1=synab[-1]
                    SBA0=synba[0]
                    SBA1=synba[-1]
        
                    class_=""

                    if SAB0>tol and SAB1>tol:
                        if SAB1-SAB0 >tol: 
                            class_="ppi"
                        elif SAB1-SAB0<-tol:
                            class_="ppd"
                    elif SAB0<-tol and SAB1>tol:
                            class_="npi"
                    elif SAB0>tol and SAB1<-tol:    
                            class_="pnd"
                    elif SAB0<-tol and SAB1<-tol:
                        if SAB1-SAB0 >tol: 
                            class_="nni"
                        elif SAB1-SAB0<-tol:
                            class_="nnd"
                    else:
                        class_="zero"

                        #print("unknown class", SAB0, SAB1)
                    if class_ != "":
                        if SBA1-SBA0 > tol:
                            class_+="i"
                        elif SBA1-SBA0<-tol:
                            class_+="d"
                        else:
                            class_+="s"
                    string="%d,%d,%s,%s,%s,%s\n"%(npt1,npt2,class_,",".join(map(str,fullpars)),",".join(map(str,synab)),",".join(map(str,synba)))
                    if class_ in potential_classes:           
                        outfile.write(string)
                        outfile.flush()
outfile.close()
                    
            
            
            
        


