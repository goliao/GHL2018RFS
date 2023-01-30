from pandas import *
from scipy.optimize import fsolve
from scipy import linalg
from math import *
from matplotlib.pyplot import *
from numpy.linalg import *
from numpy import *
import datetime
import pickle
import os.path

np.set_printoptions(precision=4,suppress=True,linewidth=140)
# cd /Users/gliao/Documents/git/python/2mkt
# locals().update(para)
def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]

def Testing(df_to_test,write=0,filename='goldstandard.pckl',show=0):
  if write==1:
    with open(filename,'w') as fp:
      pickle.dump(df_to_test,fp)
  else:
    with open(filename) as fp:
      df_gold=pickle.load(fp)
    visiblediff=all(abs(df_to_test-df_gold)<.01)
    print (visiblediff)
    if show and not visiblediff:
      print (df_to_test-df_gold)


def funSave(filename='shelve.out'):
  import shelve
  my_shelf = shelve.open(filename,'n') # 'n' for new
  for key in dir():
      try:
          my_shelf[key] = globals()[key]
      except TypeError:
          #
          # __builtins__, my_shelf, and imported modules can not be shelved.
          #
          print('ERROR shelving: {0}'.format(key))
  my_shelf.close()
def funLoad(filename='shelve.out'):
  import shelve
  my_shelf = shelve.open(filename)
  for key in my_shelf:
      globals()[key]=my_shelf[key]
  my_shelf.close()
def funvshort2vlong(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    # use environmental variables to construct missing elements from short v to longer solutions
    thetaA=1-(1/DA);thetaB=1-(1/DB);
    alphaAr=(1-thetaA)/(1-thetaA*rhor)
    alphaBr=(1-thetaB)/(1-thetaB*rhor)
    alphaAz=0
    alphaBz=(1-thetaB)/(1-thetaB*rhoz)*rhoz
    deltaAr=0; deltaAz=0; deltaBr=0; deltaBz=0
    # unpack the initial point into bare minimal
    #import pdb; pdb.set_trace()
    alphaA1_0, alphaB1_0, deltaA1_0, deltaB1_0=hsplit(v0,4)
    # repack into essential needed to construct full solution vectors 0f 2+2*k
    (alphaA1, alphaB1, deltaA1, deltaB1)=(hstack((alphaAr,alphaA1_0[0:k],alphaAz,alphaA1_0[-k:])), hstack((alphaBr,alphaB1_0[0:k],alphaBz,alphaB1_0[-k:])), hstack((deltaAr,deltaA1_0[0:k],deltaAz,deltaA1_0[-k:])), hstack((deltaBr,deltaB1_0[0:k],deltaBz,deltaB1_0[-k:])))
    return (alphaA1, alphaB1, deltaA1, deltaB1)

def funvshort2vlongDuffie(v0,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    # use environmental variables to construct missing elements from short v to longer solutions
    thetaA=1-(1/DA);thetaB=1-(1/DB);
    alphaAr=(1-thetaA)/(1-thetaA*rhor)
    alphaBr=(1-thetaB)/(1-thetaB*rhor)
    alphaAz=0
    alphaBz=(1-thetaB)/(1-thetaB*rhoz)*rhoz
    deltaAr=0; deltaAz=0; deltaBr=0; deltaBz=0
    # unpack the initial point into bare minimal
    #import pdb; pdb.set_trace()
    alphaA1_0, alphaB1_0, deltaA1_0, deltaB1_0=hsplit(v0,4)
    # repack into essential needed to construct full solution vectors 0f 2+2*k
    (alphaA1, alphaB1, deltaA1, deltaB1)=(hstack((alphaAr,alphaA1_0[0:k],alphaAz,alphaA1_0[-k:])), hstack((alphaBr,alphaB1_0[0:k],alphaBz,alphaB1_0[-k:])), hstack((deltaAr,deltaA1_0[0:k],deltaAz,deltaA1_0[-k:])), hstack((deltaBr,deltaB1_0[0:k],deltaBz,deltaB1_0[-k:])))
    return (alphaA1, alphaB1, deltaA1, deltaB1)

def funCommonCalc(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    if type(v0)==pandas.core.frame.DataFrame:
        (alphaA1, alphaB1, deltaA1, deltaB1)=(np.array(v0['alphaA'][1:]),np.array(v0['alphaB'][1:]),np.array(v0['deltaA'][1:]),np.array(v0['deltaB'][1:]))
    else:
        (alphaA1, alphaB1, deltaA1, deltaB1)=funvshort2vlong(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    thetaA=1-(1/DA);thetaB=1-(1/DB);
    L=2*(k-1)+4; I=eye((L))
    (er,esA,onesA,ez,esB,onesB,temp)=hsplit(I,[1,2,k-1+2,k-1+3,k-1+4,2*k-2+4])
    onesA=sum(onesA,axis=1);onesB=sum(onesB,axis=1)
    (er,esA,ez,esB)=(er.flatten(),esA.flatten(),ez.flatten(),esB.flatten())
    # deltaA1=deltaA1.flatten();alphaA1=alphaA1.flatten();deltaB1=deltaB1.flatten();alphaB1=alphaB1.flatten();
    # % L x L Gamma matrix of AR coefficients governing dynamics of the state vector
    if (k>1):
      Gamma=zeros_like(I);Gamma[0,0]=rhor;Gamma[1,1]=rhosA;Gamma[2,:]=deltaA1.T;Gamma[-(k-1),:]=deltaB1.T; Gamma[k+1,k+1]=rhoz;Gamma[k+2,k+2]=rhosB;t_subdiag=tri(L,L,-1)-tri(L,L,-2);t_subdiag[:,:2]=0;t_subdiag[:,k:k+3]=0;Gamma=Gamma+t_subdiag;
    else:
      Gamma=zeros_like(I);
      Gamma[0,0]=rhor;
      Gamma[1,1]=rhosA;
      Gamma[k+1,k+1]=rhoz;
      Gamma[k+2,k+2]=rhosB;
    # % L x L Sigma matrix summarizing correlations between exogenous shocks
    # % Hard wire the case where all exogenous shocks are mutually orthogonal
    Sigma=zeros_like(I); Sigma[0,0]=sigmar**2;Sigma[1,1]=sigmasA**2;Sigma[-(k+1),-(k+1)]=sigmaz**2;Sigma[-k,-k]=sigmasB**2;
    def funCMat(l,j):
        l=int(l)
        j=int(j)
        Cout=zeros_like((Gamma))
        for ind in range(1,(min(l,j)+1)):
            Cout=Cout+matrix_power(Gamma,l-ind).dot(Sigma).dot(matrix_power(Gamma,j-ind).T)
        return Cout
    Ckk=funCMat(k,k);
    Csum1=zeros_like((Gamma));
    for i in range(1,int(k)):
        Csum1=Csum1+funCMat(i,k)
    Csum2=zeros_like((Gamma));
    for i1 in range(1,int(k)):
        for i2 in range(1,int(k)):
            Csum2=Csum2+funCMat(i1,i2)
    VAk=(alphaA1-er).T.dot(Csum2).dot(alphaA1-er)+(thetaA/(1-thetaA))**2*alphaA1.T.dot(Ckk).dot(alphaA1)-2*(thetaA/(1-thetaA))*(alphaA1-er).T.dot(Csum1).dot(alphaA1)##good
    VBk=(alphaB1-er-ez).T.dot(Csum2).dot(alphaB1-er-ez)+(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)-2*(alphaB1-er-ez).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez) ##good
    #wrong: CABk=(alphaA1-er).T.dot(Csum2).dot(alphaB1-er-ez)-(alphaA1-er).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez)-thetaA/(1-thetaA)*alphaA1.T.dot(Csum1).dot(alphaB1-er-ez)+thetaA/(1-thetaA)*alphaA1.T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)## old
    CABk=(alphaA1-er).T.dot(Csum2).dot(alphaB1-er-ez)-(alphaA1-er).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez)-(alphaB1-er-ez).T.dot(Csum1).dot(thetaA/(1-thetaA)*alphaA1)+thetaA/(1-thetaA)*alphaA1.T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)## new slightly different!!! what's the difference?????? this is the error that sam noted in IA
    CAk=(alphaA1-er).T.dot(Csum2).dot(er)-thetaA/(1-thetaA)*alphaA1.T.dot(Csum1).dot(er)##
    CBk=(alphaB1-er-ez).T.dot(Csum2).dot(er)-(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Csum1).dot(er)##
    #import pdb; pdb.set_trace()
    return (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)


def funCommonCalcDuffie(v0,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    if type(v0)==pandas.core.frame.DataFrame:
        (alphaA1, alphaB1, deltaA1, deltaB1)=(np.array(v0['alphaA'][1:]),np.array(v0['alphaB'][1:]),np.array(v0['deltaA'][1:]),np.array(v0['deltaB'][1:]))
    else:
        (alphaA1, alphaB1, deltaA1, deltaB1)=funvshort2vlongDuffie(v0,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    thetaA=1-(1/DA);thetaB=1-(1/DB);
    L=2*(k-1)+4; I=eye((L))
    (er,esA,onesA,ez,esB,onesB,temp)=hsplit(I,[1,2,k-1+2,k-1+3,k-1+4,2*k-2+4])
    onesA=sum(onesA,axis=1);onesB=sum(onesB,axis=1)
    (er,esA,ez,esB)=(er.flatten(),esA.flatten(),ez.flatten(),esB.flatten())
    # deltaA1=deltaA1.flatten();alphaA1=alphaA1.flatten();deltaB1=deltaB1.flatten();alphaB1=alphaB1.flatten();
    # % L x L Gamma matrix of AR coefficients governing dynamics of the state vector
    if (k>1):
      Gamma=zeros_like(I);Gamma[0,0]=rhor;Gamma[1,1]=rhosA;Gamma[2,:]=deltaA1.T;Gamma[-(k-1),:]=deltaB1.T; Gamma[k+1,k+1]=rhoz;Gamma[k+2,k+2]=rhosB;t_subdiag=tri(L,L,-1)-tri(L,L,-2);t_subdiag[:,:2]=0;t_subdiag[:,k:k+3]=0;Gamma=Gamma+t_subdiag;
    else:
      Gamma=zeros_like(I);
      Gamma[0,0]=rhor;
      Gamma[1,1]=rhosA;
      Gamma[k+1,k+1]=rhoz;
      Gamma[k+2,k+2]=rhosB;
    # % L x L Sigma matrix summarizing correlations between exogenous shocks
    # % Hard wire the case where all exogenous shocks are mutually orthogonal
    Sigma=zeros_like(I); Sigma[0,0]=sigmar**2;Sigma[1,1]=sigmasA**2;Sigma[-(k+1),-(k+1)]=sigmaz**2;Sigma[-k,-k]=sigmasB**2;
    def funCMat(l,j):
        l=int(l)
        j=int(j)
        Cout=zeros_like((Gamma))
        for ind in range(1,(min(l,j)+1)):
            Cout=Cout+matrix_power(Gamma,l-ind).dot(Sigma).dot(matrix_power(Gamma,j-ind).T)
        return Cout
    Ckk=funCMat(k,k);
    Csum1=zeros_like((Gamma));
    for i in range(1,int(k)):
        Csum1=Csum1+funCMat(i,k)
    Csum2=zeros_like((Gamma));
    for i1 in range(1,int(k)):
        for i2 in range(1,int(k)):
            Csum2=Csum2+funCMat(i1,i2)
    VAk=(alphaA1-er).T.dot(Csum2).dot(alphaA1-er)+(thetaA/(1-thetaA))**2*alphaA1.T.dot(Ckk).dot(alphaA1)-2*(thetaA/(1-thetaA))*(alphaA1-er).T.dot(Csum1).dot(alphaA1)##good
    VBk=(alphaB1-er-ez).T.dot(Csum2).dot(alphaB1-er-ez)+(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)-2*(alphaB1-er-ez).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez) ##good
    #wrong: CABk=(alphaA1-er).T.dot(Csum2).dot(alphaB1-er-ez)-(alphaA1-er).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez)-thetaA/(1-thetaA)*alphaA1.T.dot(Csum1).dot(alphaB1-er-ez)+thetaA/(1-thetaA)*alphaA1.T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)## old
    CABk=(alphaA1-er).T.dot(Csum2).dot(alphaB1-er-ez)-(alphaA1-er).T.dot(Csum1).dot(thetaB/(1-thetaB)*alphaB1+ez)-(alphaB1-er-ez).T.dot(Csum1).dot(thetaA/(1-thetaA)*alphaA1)+thetaA/(1-thetaA)*alphaA1.T.dot(Ckk).dot(thetaB/(1-thetaB)*alphaB1+ez)## new slightly different!!! what's the difference?????? this is the error that sam noted in IA
    CAk=(alphaA1-er).T.dot(Csum2).dot(er)-thetaA/(1-thetaA)*alphaA1.T.dot(Csum1).dot(er)##
    CBk=(alphaB1-er-ez).T.dot(Csum2).dot(er)-(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Csum1).dot(er)##
    #import pdb; pdb.set_trace()
    return (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)
# the hardest part of the solution is finding alpha_S, delta_S and alpha_D and delta_D
# so we only need to initially solve for 2*k*4 elements
# we know that deltaA_r=deltaB_z=0 and we can caluate alphaA_r, alphaB_z
# we can calulate alpha0 and delta0 with a secondary solver (funEnrichSoln and funZeroConst)
def fnF(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalc(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    VA1=(thetaA/(1-thetaA))**2*alphaA1.T.dot(Sigma).dot(alphaA1) #new
    VB1=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez) #new
    ErxkACond=(alphaA1-er).T.dot(inv(I-Gamma))+thetaA/(1-thetaA)*alphaA1.T #new
    ErxkBCond=(alphaB1-er).T.dot(inv(I-Gamma))+thetaB/(1-thetaB)*alphaB1.T-ez.T.dot(inv(I-Gamma)).dot(Gamma) #new
    newalphaA1=(1-thetaA)/(1-thetaA*rhor)*er+VA1/(tau*pA)*((1-thetaA)/(1-thetaA*rhosA)*esA-(1-thetaA)*(1-pA-pB)/k*inv(I-thetaA*Gamma.T).dot(onesA+deltaA1)) #new
    newalphaB1=(1-thetaB)/(1-rhor*thetaB)*er+(1-thetaB)/(1-rhoz*thetaB)*rhoz*ez+VB1/(tau*pB)*((1-thetaB)/(1-thetaB*rhosB)*esB-(1-thetaB)*(1-pA-pB)/k*inv(I-thetaB*Gamma.T).dot(onesB+deltaB1)) #new
    newdeltaA1=tau/(VAk*VBk-CABk**2)*(VBk*ErxkACond-CABk*ErxkBCond).dot(I-matrix_power(Gamma,int(k))).T #new
    newdeltaB1=tau/(VAk*VBk-CABk**2)*(VAk*ErxkBCond-CABk*ErxkACond).dot(I-matrix_power(Gamma,int(k))).T #new
    v1=hstack((newalphaA1[1:k+1],newalphaA1[-k:],newalphaB1[1:k+1],newalphaB1[-k:],newdeltaA1[1:k+1],newdeltaA1[-k:],newdeltaB1[1:k+1],newdeltaB1[-k:]))
    return v1
def funZero(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalc(v0,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    VA1=(thetaA/(1-thetaA))**2*alphaA1.T.dot(Sigma).dot(alphaA1) #new
    VB1=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez) #new
    ErxkACond=(alphaA1-er).T.dot(inv(I-Gamma))+thetaA/(1-thetaA)*alphaA1.T #new
    ErxkBCond=(alphaB1-er).T.dot(inv(I-Gamma))+thetaB/(1-thetaB)*alphaB1.T-ez.T.dot(inv(I-Gamma)).dot(Gamma) #new
    newalphaA1=(1-thetaA)/(1-thetaA*rhor)*er+VA1/(tau*pA)*((1-thetaA)/(1-thetaA*rhosA)*esA-(1-thetaA)*(1-pA-pB)/k*inv(I-thetaA*Gamma.T).dot(onesA+deltaA1)) #new
    newalphaB1=(1-thetaB)/(1-rhor*thetaB)*er+(1-thetaB)/(1-rhoz*thetaB)*rhoz*ez+VB1/(tau*pB)*((1-thetaB)/(1-thetaB*rhosB)*esB-(1-thetaB)*(1-pA-pB)/k*inv(I-thetaB*Gamma.T).dot(onesB+deltaB1)) #new
    newdeltaA1=tau/(VAk*VBk-CABk**2)*(VBk*ErxkACond-CABk*ErxkBCond).dot(I-matrix_power(Gamma,int(k))).T #new
    newdeltaB1=tau/(VAk*VBk-CABk**2)*(VAk*ErxkBCond-CABk*ErxkACond).dot(I-matrix_power(Gamma,int(k))).T #new
    v1=hstack((newalphaA1[1:k+1],newalphaA1[-k:],newalphaB1[1:k+1],newalphaB1[-k:],newdeltaA1[1:k+1],newdeltaA1[-k:],newdeltaB1[1:k+1],newdeltaB1[-k:]))
    out=v1-v0
    return out

def funZeroDuffie(v0,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB):
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalcDuffie(v0,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    VA1=(thetaA/(1-thetaA))**2*alphaA1.T.dot(Sigma).dot(alphaA1) #new
    VB1=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez) #new
    ErxkACond=(alphaA1-er).T.dot(inv(I-Gamma))+thetaA/(1-thetaA)*alphaA1.T #new
    ErxkBCond=(alphaB1-er).T.dot(inv(I-Gamma))+thetaB/(1-thetaB)*alphaB1.T-ez.T.dot(inv(I-Gamma)).dot(Gamma) #new
    newalphaA1=(1-thetaA)/(1-thetaA*rhor)*er+VA1/(tau*q)*((1-thetaA)/(1-thetaA*rhosA)*esA-(1-thetaA)*(1-q)/k*inv(I-thetaA*Gamma.T).dot(onesA+deltaA1)) #new
    newalphaB1=(1-thetaB)/(1-rhor*thetaB)*er+(1-thetaB)/(1-rhoz*thetaB)*rhoz*ez+VB1/(tau*q)*((1-thetaB)/(1-thetaB*rhosB)*esB-(1-thetaB)*(1-q)/k*inv(I-thetaB*Gamma.T).dot(onesB+deltaB1)) #new
    newdeltaA1=tau/(VAk*VBk-CABk**2)*(VBk*ErxkACond-CABk*ErxkBCond).dot(I-matrix_power(Gamma,int(k))).T #new
    newdeltaB1=tau/(VAk*VBk-CABk**2)*(VAk*ErxkBCond-CABk*ErxkACond).dot(I-matrix_power(Gamma,int(k))).T #new
    v1=hstack((newalphaA1[1:k+1],newalphaA1[-k:],newalphaB1[1:k+1],newalphaB1[-k:],newdeltaA1[1:k+1],newdeltaA1[-k:],newdeltaB1[1:k+1],newdeltaB1[-k:]))
    out=v1-v0
    return out

def funZeroConst(w0,v,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB,cash_cov=0):
    (alphaA0,alphaB0,deltaA0,deltaB0)=w0
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalc(v,k,tau,sigmasA,sigmasB,pA,pB,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    VA1=(thetaA/(1-thetaA))**2*alphaA1.T.dot(Sigma).dot(alphaA1) #new
    VB1=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez) #new

    ErxkAUncond=k*(alphaA0-rm)
    ErxkBUncond=k*(alphaB0-rm-zm)
    newalphaB0=rm+zm+VB1/(tau*pB)*(smB-(1-pA-pB)*deltaB0)
    newdeltaB0=tau/(VAk*VBk-CABk**2)*(VAk*ErxkBUncond-CABk*ErxkAUncond)
    newalphaA0=rm+VA1/(tau*pA)*(smA-(1-pA-pB)*deltaA0)
    newdeltaA0=tau/(VAk*VBk-CABk**2)*(VBk*ErxkAUncond-CABk*ErxkBUncond)
    # if we want to have covariance between k period bond return and cash return over k periods:
    if cash_cov:
      #check before using
      newdeltaB0=newdeltaB0-1/(VAk*VBk-CABk**2)*(VAk*CBk-CABk*CAk)
      newdeltaA0=newdeltaA0-1/(VAk*VBk-CABk**2)*(VBk*CAk-CABk*CBk)
    w1=hstack((newalphaA0,newalphaB0,newdeltaA0,newdeltaB0))
    return (w1-w0)


def funZeroConstDuffie(w0,v,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB,cash_cov=0):
    (alphaA0,alphaB0,deltaA0,deltaB0)=w0
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalcDuffie(v,k,tau,sigmasA,sigmasB,q,smA,smB,rhosA,rhosB,rhor,rhoz,sigmar,sigmaz,rm,zm,DA,DB)
    VA1=(thetaA/(1-thetaA))**2*alphaA1.T.dot(Sigma).dot(alphaA1) #new
    VB1=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez) #new
    ErxkAUncond=k*(alphaA0-rm)
    ErxkBUncond=k*(alphaB0-rm-zm)
    newalphaB0=rm+zm+VB1/(tau*q)*(smB-(1-q)*deltaB0)
    newdeltaB0=tau/(VAk*VBk-CABk**2)*(VAk*ErxkBUncond-CABk*ErxkAUncond)
    newalphaA0=rm+VA1/(tau*q)*(smA-(1-q)*deltaA0)
    newdeltaA0=tau/(VAk*VBk-CABk**2)*(VBk*ErxkAUncond-CABk*ErxkBUncond)
    # if we want to have covariance between k period bond return and cash return over k periods:
    if cash_cov:
      #check before using
      newdeltaB0=newdeltaB0-1/(VAk*VBk-CABk**2)*(VAk*CBk-CABk*CAk)
      newdeltaA0=newdeltaA0-1/(VAk*VBk-CABk**2)*(VBk*CAk-CABk*CBk)
    w1=hstack((newalphaA0,newalphaB0,newdeltaA0,newdeltaB0))
    return (w1-w0)

def funEnrichSoln(v,**para):
    w0=zeros((4,))
    (alphaA0,alphaB0,deltaA0,deltaB0),out1_const,ex_const,exm_const = fsolve(lambda x:funZeroConst(x,v, **para),w0, full_output=1)#,maxfev=200*(8*k+1)
    (alphaA1, alphaB1, deltaA1, deltaB1)=funvshort2vlong(v,**para)
    if ex_const!=1:
        print('no const found')
    soln_array=(alphaA0,alphaB0,deltaA0,deltaB0,alphaA1, alphaB1, deltaA1, deltaB1)
    soln_dfs_tup=funSolnArray2Df(hstack((alphaA0,alphaB0,deltaA0,deltaB0,alphaA1, alphaB1, deltaA1, deltaB1)),**para)
    return (soln_array,soln_dfs_tup)

def funEnrichSolnDuffie(v,**para):
    w0=zeros((4,))
    (alphaA0,alphaB0,deltaA0,deltaB0),out1_const,ex_const,exm_const = fsolve(lambda x:funZeroConstDuffie(x,v, **para),w0, full_output=1)#,maxfev=200*(8*k+1)
    (alphaA1, alphaB1, deltaA1, deltaB1)=funvshort2vlongDuffie(v,**para)
    if ex_const!=1:
        print('no const found')
    soln_array=(alphaA0,alphaB0,deltaA0,deltaB0,alphaA1, alphaB1, deltaA1, deltaB1)
    soln_dfs_tup=funSolnArray2Df(hstack((alphaA0,alphaB0,deltaA0,deltaB0,alphaA1, alphaB1, deltaA1, deltaB1)),**para)
    return (soln_array,soln_dfs_tup)

def funGuessInitialVector(k,**para):
    v0=(random.rand(8*int(k),1)-.5)*random.rand()*20
    return v0


def funSolnArray2Df(soln,**para):
    k=para['k']
    if size(soln)!=4*(4+2*(k-1))+5: soln=hstack((0,soln))
    # import pdb; pdb.set_trace()
    (freq,alphaA0,alphaB0,deltaA0,deltaB0)=soln[:5] #solnfull[soln,:5]
    (alphaA1,alphaB1,deltaA1,deltaB1)=hsplit(soln[5:],4)#hsplit(solnfull[soln,5:],4)
    alphaA=hstack((alphaA0,alphaA1));alphaB=hstack((alphaB0,alphaB1));deltaA=hstack((deltaA0,deltaA1));deltaB=hstack((deltaB0,deltaB1));
    indexnames=['0','r','sA','z','sB']
    for ids in range(int(k)-1):
        indexnames.insert(3,'dA')
        indexnames.append('dB')
    df=DataFrame(data=np.vstack((alphaA,alphaB,deltaA,deltaB)).T,columns=['alphaA','alphaB','deltaA','deltaB'],index=indexnames)
    return (df,freq,para)

def funSolveUnique(para,niter=100):
    # niter=30
    k=para['k']; L=2*(k-1)+4;
    nsolnpass=0; nconverge=0; inisolve=0;
    for i in range(0,niter):
        print(i)
        if nconverge>6: break
        v0=funGuessInitialVector(**para)*.1
        if i==0: v0=0*v0
        v,out1,ex,exm = fsolve(lambda x:funZero(x, **para),v0, full_output=1)
        if ex!=1: continue # if soln didn't converge
        alphaA_short, alphaB_short, deltaA_short, deltaB_short=hsplit(v,4)
        indexnames=['sA','sB']
        for ids in range(int(k)-1):
            indexnames.insert(1,'dA')
            indexnames.append('dB')
        #import pdb; pdb.set_trace()
        df=DataFrame(data=np.vstack((alphaA_short,alphaB_short,deltaA_short,deltaB_short)).T,columns=['alphaA','alphaB','deltaA','deltaB'],index=indexnames)
        df=concat([df,DataFrame(data=[df[1:(1+int(k)-1)].sum(axis=0), df[(3+int(k)-2):].sum(axis=0)],columns=['alphaA','alphaB','deltaA','deltaB'],index=['sumdA','sumdB'])])

        # if (np.max(np.abs(np.linalg.eigvals(fnjcob(v,**para))))>1.):
        #     print('max abs eig of jacobian: %s' % (np.max(np.abs(np.linalg.eigvals(fnjcob(v,**para))))))
            #continue
        # first filter: filter by sum of deltaD's
        if df['deltaA']['sumdA']>0 or  df['deltaA']['sumdA']<-1 or df['deltaA']['sumdB']<-1 or  df['deltaB']['sumdA']<-1 or df['deltaB']['sumdB']>0 or  df['deltaB']['sumdB']<-1:
            continue # if sum of lagged demand coef. out of nice range
        # 2nd filter: filter by magnitude of sigmaS's
        if 'dfsave' not in locals():
            dfsave=df.copy()
            nconverge=1
            if i==0: inisolve=1
        elif all(around(df,4)==around(dfsave,4)):
            nconverge+=1
            continue
        elif df.alphaA.sA**2+df.alphaA.sB**2+df.alphaB.sA**2+df.alphaB.sB**2<dfsave.alphaA.sA**2+dfsave.alphaA.sB**2+dfsave.alphaB.sA**2+dfsave.alphaB.sB**2:
            dfsave=df.copy()
            nconverge=1
            inisolve=2
        else:
            nsolnpass+=1
            continue
    # print dfsave
    # print '# of convergence to this solution: %i out of %i' % (nconverge, i+1)
    # print '# of other plausible solutions passed: %i' % nsolnpass
    if inisolve==0:
        print ('initial 0 vector did not converge')
        return 0,0,0
    # if inisolve==1: print 'initial 0 vector converged'
    if inisolve==2: print ('obtained a solution that replaced the initial solution solved with zero vector! Check')
    if 'dfsave' not in locals():
        dfsave=None
        out_tup=(None,0,para)
        print('no solution found')
    else:
        v=dfsave[:-2].values.T.flatten()
        temp,out_tup=funEnrichSoln(v,**para);
    #if ~check_eig_gamma(out_tup[0],**para): print('error: max(abs(eigval(Gamma)))>=1')
    return out_tup, dfsave, v

def funSolveUniqueDuffie(para,niter=100):
    # niter=30
    k=para['k']; L=2*(k-1)+4;
    v0=funGuessInitialVector(**para)
    v0=0*v0
    import pdb; pdb.set_trace()

    v,out1,ex,exm = fsolve(lambda x:funZeroDuffie(x, **para),v0, full_output=1)
    if ex!=1: print( 'soln didnt converge')
    alphaA_short, alphaB_short, deltaA_short, deltaB_short=hsplit(v,4)
    indexnames=['sA','sB']
    for ids in range(int(k)-1):
        indexnames.insert(1,'dA')
        indexnames.append('dB')
    df=DataFrame(data=np.vstack((alphaA_short,alphaB_short,deltaA_short,deltaB_short)).T,columns=['alphaA','alphaB','deltaA','deltaB'],index=indexnames)
    df=concat([df,DataFrame(data=[df[1:(1+int(k)-1)].sum(axis=0), df[(3+int(k)-2):].sum(axis=0)],columns=['alphaA','alphaB','deltaA','deltaB'],index=['sumdA','sumdB'])])
    #v=dfsave[:-2].values.T.flatten()
    temp,out_tup=funEnrichSolnDuffie(v,**para);
    return out_tup, df

def check_eig_gamma(solin,**para):
    (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalc(solin,**para)
    return max(abs(eigvals(Gamma)))<1


def funCS(paravary,para,T=30,shock=5.0,t_announce=1,t_implement=1,**simkargs):
    # T=10;simniter=1e3
  sims=zeros((size(paravary),int(T),4))
  for ind_para in range(0, size(paravary)):
    para.update(paravary[ind_para])
    solntup, solncore,_=funSolveUnique(para,niter=1)
    if type(solncore)==pandas.core.frame.DataFrame:
      sims[ind_para]=funSimulateEV(solntup,T=T,shock=shock,t_announce=t_announce,t_implement=t_implement,**simkargs)[['yA','yB','ErxA','ErxB']]*4.
  condsims=sims-sims[:,0].repeat(T,axis=0).reshape(size(paravary),T,shape(sims)[2])
  index=[]
  for parasingle in paravary:
    index.append(str(parasingle).translate(None,"{}'"))
  df=DataFrame(index=index)
  ### cannot use this process for k vary
  (df['yA_short'],df['yB_short'],df['ErxA_short'],df['ErxB_short'])=condsims[:,t_announce,:].T
  (df['yA_long'],df['yB_long'],df['ErxA_long'],df['ErxB_long'])=condsims[:,t_announce+2*para['k'],:].T
  df['yA_preshock']=sims[:,0,0]
  df['yB_preshock']=sims[:,0,1]
  df['ErxA_preshock']=sims[:,0,2]
  df['ErxB_preshock']=sims[:,0,3]
  meanrevadj=1/(para['rhosA']**(2*para['k']))-1
  df['yA_overreact']=df['yA_short']/df['yA_long']-1-meanrevadj
  df['yB_overreact']=df['yB_short']/df['yB_long']-1-meanrevadj
  df['ErxA_overreact']=df['ErxA_short']/df['ErxA_long']-1-meanrevadj
  df['ErxB_overreact']=df['ErxB_short']/df['ErxB_long']-1-meanrevadj

  psims='nothing'#Panel(condsims,items=index,minor_axis=['yA','yB','ErxA','ErxB'])
  #df.to_clipboard()
  return df, psims


def funCS2(paravary,para,T=15.,shock=.5,t_announce=1,t_implement=1,**simkargs):
    # for use with figure 9 only to generate ...
# T=10;simniter=1e3
  sims=zeros((size(paravary),int(T),8))
  for ind_para in range(0, size(paravary)):
    para.update(paravary[ind_para])
    solntup, solncore,_=funSolveUnique(para,niter=1)
    if type(solncore)==pandas.core.frame.DataFrame:
      sims[ind_para]=funSimulateEV(solntup,T=T,shock=shock,t_announce=t_announce,t_implement=t_implement,**simkargs)[['yA','yB','ErxA','ErxB','bA','bB','dA','dB']]
  condsims=sims-sims[:,0].repeat(T,axis=0).reshape(size(paravary),T,shape(sims)[2])
  index=[]
  for parasingle in paravary:
    index.append(str(parasingle).translate(None,"{}'"))
  df=DataFrame(index=index)
  ### cannot use this process for k vary
  # (df['yA_short'],df['yB_short'],df['ErxA_short'],df['ErxB_short'])=condsims[:,t_announce,:].T
  # (df['yA_long'],df['yB_long'],df['ErxA_long'],df['ErxB_long'])=condsims[:,t_announce+2*para['k'],:].T
  # df['yA_preshock']=sims[:,0,0]
  # df['yB_preshock']=sims[:,0,1]
  df['ErxA_preshock']=sims[:,0,2]
  df['ErxB_preshock']=sims[:,0,3]
  # meanrevadj=1/(para['rhosA']**(2*para['k']))-1
  # df['yA_overreact']=df['yA_short']/df['yA_long']-1-meanrevadj
  # df['yB_overreact']=df['yB_short']/df['yB_long']-1-meanrevadj
  # df['ErxA_overreact']=df['ErxA_short']/df['ErxA_long']-1-meanrevadj
  # df['ErxB_overreact']=df['ErxB_short']/df['ErxB_long']-1-meanrevadj

  psims=Panel(condsims,items=index,minor_axis=['yA','yB','ErxA','ErxB','bA','bB','dA','dB'])
  # df.to_clipboard()
  return df, psims

def funSimulateEV(sol_tup, bshock=0, T=25, shock=.3,t_announce=10,t_implement=10, ycTenors=None,returnX=None,bse=0,seiter=1e4,decompose=0):
  # sol_tup=solntup; bshock=0; T=7; shock=1;t_announce=2;t_implement=2;ycTenors=np.array([1,2,5,10])
  para=sol_tup[2]
  k=para['k'];tau=para['tau'];sigmasA=para['sigmasA'];sigmasB=para['sigmasB'];pA=para['pA'];pB=para['pB'];smA=para['smA'];smB=para['smB'];
  rhosA=para['rhosA'];rhosB=para['rhosB'];rhor=para['rhor'];rhoz=para['rhoz'];sigmar=para['sigmar'];sigmaz=para['sigmaz'];rm=para['rm'];zm=para['zm'];
  DA=para['DA'];DB=para['DB'];

  df=sol_tup[0] 
  # import pdb; pdb.set_trace()

  (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalc(df,**sol_tup[2])
  (alphaA0, alphaB0, deltaA0, deltaB0)=(np.array(df['alphaA'][0]),np.array(df['alphaB'][0]),np.array(df['deltaA'][0]),np.array(df['deltaB'][0]))

  #define spot/fwd shock
  if t_announce==t_implement and size(shock)==1:
    shockvector=0;
  elif t_announce<t_implement and size(shock)==1: # Single fwd shock at t_implement
    jth=k-(t_implement-t_announce)-1;
    shockvector=zeros((k-1,));
    shockvector[jth]=shock*k/(1-pA-pB);
    if t_implement-t_announce>k-1:
      print('Error Implementation-Announcement > k-1 !!!')
      print (shockvector)
      print (jth)
    # import pdb; pdb.set_trace()
  elif size(shock)>1: # use the provided shock vector irregardless of t_implement
    if size(shock)==k-1:
      shockvector=shock*k/(1-pA-pB)
    else:
      print('shock vector of the wrong size')
      # return 0
  else: print ('error in anticipated vector/timing')
  ind_pastholdingsA=where(onesA==1)[0]
  ind_pastholdingsB=where(onesB==1)[0]
  ErxAuncond=(alphaA0-rm);
  ErxAcond=(1/(1-thetaA)*alphaA1-er).T-(thetaA/(1-thetaA)*alphaA1).T.dot(Gamma);
  VarA=(thetaA/(1-thetaA)*alphaA1).T.dot(Sigma).dot(thetaA/(1-thetaA)*alphaA1);


  ErxBuncond=alphaB0-rm-zm;
  ErxBcond=(1/(1-thetaB)*alphaB1-er).T-(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Gamma);
  VarB=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez);

  ti=arange(0,T); x=zeros((T,L))
  t=1
  for t in ti:
    if t>0:
      x[t,:]=Gamma.dot(x[t-1,:]) ######+SigmaSqrt.dot(randmat[ind_m,t])
    if t==t_announce:
      if bshock!=1: # % A shock
        x[t,ind_pastholdingsA]=x[t,ind_pastholdingsA]+shockvector;
        x[t,1]=x[t,1]+sum(shock);
      else: # % B shock:
        x[t,ind_pastholdingsB]=x[t,ind_pastholdingsB]+shockvector;
        x[t,k+2]=x[t,k+2]+sum(shock);
  r=x[:,0]+rm;
  z=x[:,1+k]+zm;
  #sA=x[:,1];
  #sB=x[:,2+k]
  ErxA=ErxAuncond+x.dot(ErxAcond)
  ErxB=ErxBuncond+x.dot(ErxBcond)
  bA=tau*ErxA/VarA
  bB=tau*ErxB/VarB

  dA=deltaA0+x.dot(deltaA1)
  yA=alphaA0+x.dot(alphaA1)
  sA=smA+x.dot(esA)

  dB=deltaB0+x.dot(deltaB1)
  yB=alphaB0+x.dot(alphaB1)
  sB=smB+x.dot(esB)

  pastholdingsA=(x[:,ind_pastholdingsA].sum(axis=1)+(k-1)*deltaA0)*(1-pA-pB)/k
  pastholdingsB=(x[:,ind_pastholdingsB].sum(axis=1)+(k-1)*deltaB0)*(1-pA-pB)/k

  demandA=(1-pA-pB)/k*dA+pA*bA;
  supplyA=sA-pastholdingsA;
  demandB=(1-pA-pB)/k*dB+pB*bB;
  supplyB=sB-pastholdingsB;


  if sum((demandA-supplyA)**2)>10e-4 or sum((demandB-supplyB)**2)>10e-4:
    print('supply demand mismatch')
    print('%s' % sum((demandB-supplyB)**2))
    print('%s' % sum((demandA-supplyA)**2))


  dct={'r':r, 'ErxA':ErxA,  'bA':bA, 'dA':dA, 'yA':yA, 'sA':sA, 'pastholdingsA':pastholdingsA, 'demandA':demandA, 'supplyA':supplyA,'ErxB':ErxB,  'bB':bB, 'dB':dB, 'yB':yB, 'sB':sB, 'pastholdingsB':pastholdingsB, 'demandB':demandB, 'supplyB':supplyB}
  df=DataFrame(data=dct,index=ti)

  if decompose==1:
    alphaA1_sA=alphaA1[1]
    alphaA1_sB=alphaA1[2+k]
    df['ErxA_r']=(sigmar**2/tau*(thetaA/(1-rhor*thetaA))**2)*bA
    df['ErxA_sA']=(sigmasA**2/tau*(thetaA/(1-thetaA)*alphaA1_sA)**2)*bA
    df['ErxA_sB']=(sigmasB**2/tau*(thetaA/(1-thetaA)*alphaA1_sB)**2)*bA
    alphaB1_sA=alphaB1[1]
    alphaB1_sB=alphaB1[2+k]
    df['ErxB_r']=(sigmar**2/tau*(thetaB/(1-rhor*thetaB))**2)*bB
    df['ErxB_z']=((sigmaz**2)/tau*(1/(1-rhoz*thetaB))**2)*bB
    df['ErxB_sA']=(sigmasA**2/tau*(thetaB/(1-thetaB)*alphaB1_sA)**2)*bB
    df['ErxB_sB']=(sigmasB**2/tau*(thetaB/(1-thetaB)*alphaB1_sB)**2)*bB

  if any(ycTenors):

    ind=0;
    df_curve=DataFrame(index=ti)
    for n in ycTenors:
      tempA=0;tempB=0;
      for i in range(0,n+1): # sume from i=0 to i=n-1
        if i==n: break
        # print i
        tempA+=rm+(1-thetaA)*(n-i)*((alphaA0-rm)+x.dot((er/((1-thetaA)*(n-i))+(alphaA1/(1-thetaA)-er)-(thetaA/(1-thetaA)*alphaA1).dot(Gamma)).dot(matrix_power(Gamma,n))))
        tempB+=rm+zm+(1-thetaB)*(n-i)*(alphaB0-rm-zm+x.dot(((alphaB1/(1-thetaB)-er+er/((1-thetaB)*(n-i)))-((thetaB*alphaB1/(1-thetaB)+ez-ez/(n-i)).dot(Gamma))).dot(matrix_power(Gamma,n))))
      df_curve['A'+str(n)]=tempA/float(n)
      df_curve['B'+str(n)]=tempB/float(n)
    df=concat([df,df_curve],axis=1)
    ## end curve
  # s.e.
  if bse:
    df['yA_se']=zeros_like(df.yA)
    df['yB_se']=zeros_like(df.yA)
    df['ErxA_se']=zeros_like(df.yA)
    df['ErxB_se']=zeros_like(df.yA)

    T2=T-t_announce
    se=zeros((T2,))
    for n in range(1,T2+1):
      aa=zeros_like(Sigma)
      for i in range(1, n+1):
        aa+=(matrix_power(Gamma,n-i).dot(Sigma).dot(matrix_power(Gamma,n-i).T))
      df.yA_se[t_announce+n-1]=sqrt(alphaA1.T.dot(aa).dot(alphaA1))
      df.yB_se[t_announce+n-1]=sqrt(alphaB1.T.dot(aa).dot(alphaB1))
      df.ErxA_se[t_announce+n-1]=sqrt(ErxAcond.T.dot(aa).dot(ErxAcond))
      df.ErxB_se[t_announce+n-1]=sqrt(ErxBcond.T.dot(aa).dot(ErxBcond))
  df['spread']=df.yB-df.yA

  if returnX: return df,x
  else: return df

def funSimulateEVDuffie(sol_tup, bshock=0, T=25, shock=.3,t_announce=10,t_implement=10, ycTenors=None,returnX=None,bse=0,seiter=1e4,decompose=0):
  # sol_tup=solntup; bshock=0; T=7; shock=1;t_announce=2;t_implement=2;ycTenors=np.array([1,2,5,10])
  para=sol_tup[2]
  k=para['k'];tau=para['tau'];sigmasA=para['sigmasA'];sigmasB=para['sigmasB'];q=para['q'];smA=para['smA'];smB=para['smB'];
  rhosA=para['rhosA'];rhosB=para['rhosB'];rhor=para['rhor'];rhoz=para['rhoz'];sigmar=para['sigmar'];sigmaz=para['sigmaz'];rm=para['rm'];zm=para['zm'];
  DA=para['DA'];DB=para['DB'];

  df=sol_tup[0]
  (thetaA,thetaB,DA,DB,L,I,er,esA,onesA,ez,esB,onesB,alphaA1,alphaB1,deltaA1,deltaB1,Gamma,Sigma,Ckk,Csum1,Csum2,VAk,VBk,CABk,CAk,CBk)=funCommonCalcDuffie(df,**sol_tup[2])
  (alphaA0, alphaB0, deltaA0, deltaB0)=(np.array(df['alphaA'][0]),np.array(df['alphaB'][0]),np.array(df['deltaA'][0]),np.array(df['deltaB'][0]))

  #define spot/fwd shock
  if t_announce==t_implement and size(shock)==1:
    shockvector=0;
  elif t_announce<t_implement and size(shock)==1: # Single fwd shock at t_implement
    jth=k-(t_implement-t_announce)-1;
    shockvector=zeros((k-1,));
    shockvector[jth]=shock*k/(1-q);
    if t_implement-t_announce>k-1:
      print('Error Implementation-Announcement > k-1 !!!')
      print (shockvector)
      print (jth)
    # import pdb; pdb.set_trace()
  elif size(shock)>1: # use the provided shock vector irregardless of t_implement
    if size(shock)==k-1:
      shockvector=shock*k/(1-q)
    else:
      print('shock vector of the wrong size')
      # return 0
  else: print ('error in anticipated vector/timing')
  ind_pastholdingsA=where(onesA==1)[0]
  ind_pastholdingsB=where(onesB==1)[0]
  ErxAuncond=(alphaA0-rm);
  ErxAcond=(1/(1-thetaA)*alphaA1-er).T-(thetaA/(1-thetaA)*alphaA1).T.dot(Gamma);
  VarA=(thetaA/(1-thetaA)*alphaA1).T.dot(Sigma).dot(thetaA/(1-thetaA)*alphaA1);


  ErxBuncond=alphaB0-rm-zm;
  ErxBcond=(1/(1-thetaB)*alphaB1-er).T-(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Gamma);
  VarB=(thetaB/(1-thetaB)*alphaB1+ez).T.dot(Sigma).dot(thetaB/(1-thetaB)*alphaB1+ez);

  ti=arange(0,T); x=zeros((T,L))
  t=1
  for t in ti:
    if t>0:
      x[t,:]=Gamma.dot(x[t-1,:]) ######+SigmaSqrt.dot(randmat[ind_m,t])
    if t==t_announce:
      if bshock!=1: # % A shock
        x[t,ind_pastholdingsA]=x[t,ind_pastholdingsA]+shockvector;
        x[t,1]=x[t,1]+sum(shock);
      else: # % B shock:
        x[t,ind_pastholdingsB]=x[t,ind_pastholdingsB]+shockvector;
        x[t,k+2]=x[t,k+2]+sum(shock);
  r=x[:,0]+rm;
  z=x[:,1+k]+zm;
  #sA=x[:,1];
  #sB=x[:,2+k]
  ErxA=ErxAuncond+x.dot(ErxAcond)
  ErxB=ErxBuncond+x.dot(ErxBcond)
  bA=tau*ErxA/VarA
  bB=tau*ErxB/VarB

  dA=deltaA0+x.dot(deltaA1)
  yA=alphaA0+x.dot(alphaA1)
  sA=smA+x.dot(esA)

  dB=deltaB0+x.dot(deltaB1)
  yB=alphaB0+x.dot(alphaB1)
  sB=smB+x.dot(esB)

  pastholdingsA=(x[:,ind_pastholdingsA].sum(axis=1)+(k-1)*deltaA0)*(1-q)/k
  pastholdingsB=(x[:,ind_pastholdingsB].sum(axis=1)+(k-1)*deltaB0)*(1-q)/k

  demandA=(1-q)/k*dA+q*bA;
  supplyA=sA-pastholdingsA;
  demandB=(1-q)/k*dB+q*bB;
  supplyB=sB-pastholdingsB;


  if sum((demandA-supplyA)**2)>10e-4 or sum((demandB-supplyB)**2)>10e-4:
    print('supply demand mismatch')
    print('%s' % sum((demandB-supplyB)**2))
    print('%s' % sum((demandA-supplyA)**2))


  dct={'r':r, 'ErxA':ErxA,  'bA':bA, 'dA':dA, 'yA':yA, 'sA':sA, 'pastholdingsA':pastholdingsA, 'demandA':demandA, 'supplyA':supplyA,'ErxB':ErxB,  'bB':bB, 'dB':dB, 'yB':yB, 'sB':sB, 'pastholdingsB':pastholdingsB, 'demandB':demandB, 'supplyB':supplyB}
  df=DataFrame(data=dct,index=ti)

  if decompose==1:
    alphaA1_sA=alphaA1[1]
    alphaA1_sB=alphaA1[2+k]
    df['ErxA_r']=(sigmar**2/tau*(thetaA/(1-rhor*thetaA))**2)*bA
    df['ErxA_sA']=(sigmasA**2/tau*(thetaA/(1-thetaA)*alphaA1_sA)**2)*bA
    df['ErxA_sB']=(sigmasB**2/tau*(thetaA/(1-thetaA)*alphaA1_sB)**2)*bA
    alphaB1_sA=alphaB1[1]
    alphaB1_sB=alphaB1[2+k]
    df['ErxB_r']=(sigmar**2/tau*(thetaB/(1-rhor*thetaB))**2)*bB
    df['ErxB_z']=((sigmaz**2)/tau*(1/(1-rhoz*thetaB))**2)*bB
    df['ErxB_sA']=(sigmasA**2/tau*(thetaB/(1-thetaB)*alphaB1_sA)**2)*bB
    df['ErxB_sB']=(sigmasB**2/tau*(thetaB/(1-thetaB)*alphaB1_sB)**2)*bB

  df['spread']=df.yB-df.yA

  if returnX: return df,x
  else: return df


def funPlotCurves(df, ycTenors,figsize=(8,8)):
  fig,axes=subplots(nrows=2, ncols=1,figsize=figsize,sharex='col',sharey='row')
  ind=0.; N=float(size(ycTenors))
  for itenor in ycTenors:
    ind+=1
    df['A'+str(itenor)].plot(ax=axes[0],title='yield curve A', color=cm.Greys(ind/N),legend=1)
    df['B'+str(itenor)].plot(ax=axes[1],title='yield curve B', color=cm.Greys(ind/N),legend=1)
  df.yA.plot(ax=axes[0],color='k',legend='yA', linestyle='-.')
  df.yB.plot(ax=axes[1],color='k',legend='yB', linestyle='-.')
  axes[0].legend(['$y_A^{(2)}$','$y_A^{(5)}$','$y_A^{(10)}$','$y_A$'],loc='best')
  axes[1].legend(['$y_B^{(2)}$','$y_B^{(5)}$','$y_B^{(10)}$','$y_B$'],loc='best')
  # for ax in fig.get_axes():
  #   setAxLinesBW(ax)

def funPlotSim4(df,savename=None,savename2=None, title=None,figsize=(5,3), bpastholding=0, bse=0):
  # figsize=(7,7); bpastholding=0; bse=0

  fig, axes = subplots(nrows=1, ncols=1,figsize=figsize) #
  fig2, axes2 = subplots(nrows=1, ncols=1,figsize=figsize) #
  fig3, axes3 = subplots(nrows=1, ncols=1,figsize=figsize) #
  fig4, axes4 = subplots(nrows=1, ncols=1,figsize=figsize) #
  # rcParams.update({'legend.fontsize': 10})
  df.loc[:,['ErxA','ErxB']].plot(ax=axes);
  df.loc[:,['yA','yB']].plot(ax=axes2)
  df.loc[:,['bA','dA','bB','dB']].plot(ax=axes3)
  df.loc[:,['supplyA','supplyB']].plot(ax=axes4);

  if bse:
    axes.fill_between(df.index,df.ErxA-1.96*df.ErxA_se,df.ErxA+1.96*df.ErxA_se,color='.85')
    axes2.fill_between(df.index,df.ErxB-1.96*df.ErxB_se,df.ErxB+1.96*df.ErxB_se,color='.75')
    axes3.fill_between(df.index,df.yA-1.96*df.yA_se,df.yA+1.96*df.yA_se,color='.85')
    axes4.fill_between(df.index,df.yB-1.96*df.yB_se,df.yB+1.96*df.yB_se,color='.75')
  if title: axes.set_title(title)

  # Legends
  axes.legend(['$Erx_A$','$Erx_B$'],loc='best', borderaxespad=0.)
  axes2.legend(['$y_A$','$y_B$'],loc='best', borderaxespad=0.)
  axes3.legend(['$b_A$','$d_A$','$b_B$','$d_B$'],loc='best', borderaxespad=0.)
  axes4.legend(['Active Supply A','Active Supply B'],loc='best', borderaxespad=0.)
  axes.set_xlabel("Time")
  axes2.set_xlabel("Time")
  axes3.set_xlabel("Time")
  axes4.set_xlabel("Time")

  axes.set_ylabel("Percent")
  axes2.set_ylabel("Percent")
  axes3.set_ylabel("Holdings")
  axes4.set_ylabel("Holdings")

  setAxLinesBW2(axes)
  setAxLinesBW2(axes2)
  setAxLinesBW2(axes3)
  setAxLinesBW2(axes4)
  if savename:
    fig.savefig(savename+'a.eps')#,bbox_inches='tight')
    fig2.savefig(savename+'b.eps')#,bbox_inches='tight')
    fig3.savefig(savename2+'a.eps')#,bbox_inches='tight')
    fig4.savefig(savename2+'b.eps')#,bbox_inches='tight')

def setAxLinesBW(ax):
  """
  Take each Line2D in the axes, ax, and convert the line style to be
  suitable for black and white viewing.
  """
  MARKERSIZE = 3

  COLORMAP = {
    'b': {'marker': None, 'dash': (None,None)},
    'g': {'marker': None, 'dash': [5,5]},
    'r': {'marker': None, 'dash': [1,3]},
    'c': {'marker': None, 'dash': [5,3,1,3]},
    'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
    'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
    'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
    }
  if ax.get_lines():
    for line in ax.get_lines() + ax.get_legend().get_lines():
      origColor = line.get_color()
      line.set_color('black')
      line.set_dashes(COLORMAP[origColor]['dash'])
      line.set_marker(COLORMAP[origColor]['marker'])
      line.set_markersize(MARKERSIZE)


def setAxLinesBW2(ax):
  """
  Take each Line2D in the axes, ax, and convert the line style to be
  suitable for black and white viewing.
  """
  MARKERSIZE = 3

  COLORMAP = {
    'b': {'marker': None, 'dash': (None,None)},
    'g': {'marker': None, 'dash': [5,5]},
    'r': {'marker': None, 'dash': [1,3]},
    'c': {'marker': None, 'dash': [5,3,1,3]},
    'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
    'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
    'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
    }
  dashes = [[1],[5,5],[1,3],[5,3,1,3],[5,2,5,2,5,10],[5,3,1,2,1,10]]
    
  if ax.get_lines():
    i=0
    for line in ax.get_lines(): #+ ax.get_legend().get_lines():
      i=i+1
      origColor = line.get_color()
      line.set_dashes(dashes[i])
    i=0
    for line in ax.get_legend().get_lines():
      i=i+1
      origColor = line.get_color()
      line.set_dashes(dashes[i])
      
def setAxLinesBW3(ax):
  """
  Take each Line2D in the axes, ax, and convert the line style to be
  suitable for black and white viewing.
  """
  MARKERSIZE = 3

  COLORMAP = {
    'b': {'marker': None, 'dash': (None,None)},
    'g': {'marker': None, 'dash': [5,5]},
    'r': {'marker': None, 'dash': [1,3]},
    'c': {'marker': None, 'dash': [5,3,1,3]},
    'm': {'marker': 'o', 'dash': [5, 1,1,10]},
    'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
    'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
    }
  if ax.get_lines():
    for line in ax.get_lines() + ax.get_legend().get_lines():
      origColor = line.get_color()
      # line.set_color('black')
      line.set_dashes(COLORMAP[origColor]['dash'])
      line.set_marker(COLORMAP[origColor]['marker'])
      line.set_markersize(MARKERSIZE)
