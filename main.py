from source import *

#conputation parameters
PRE_HEAT = 3000
TOTAL_STEP = 10000
RG_ITERATION = 4

Tc = J / 0.22165

'''
IsingGroup=np.empty((n,L,L),dtype=int)
IsingGroup0=np.empty((n,L,L),dtype=int)
IsingGroup1=np.empty((n,L//BLOCK,L//BLOCK),dtype=int)
IsingGroup2=np.empty((n,L//(BLOCK*BLOCK),L//(BLOCK*BLOCK)),dtype=int)
IsingGroup3=np.empty((n,L//(BLOCK*BLOCK*BLOCK),L//(BLOCK*BLOCK*BLOCK)),dtype=int)

for i in range(n):
    IsingGroup[i]=Initialq()
outputphi=[]
for m in range(nRG):
    outputphi.append(open('dataphi2_L='+str(L)+'_n='+str(n)+'_m='+str(m)+'.txt','w'))

for i in range(3000):#pre_heat
    for j in range(n):
        MCpass(IsingGroup[j],Tc)
print('pre_heat finish')

for j in range(n):#initialize IsingGroup0
    IsingGroup0[j]=IsingGroup[j]
    IsingGroup1[j]=BlockIsing(IsingGroup0[j],BLOCK)
    IsingGroup2[j]=BlockIsing(IsingGroup1[j],BLOCK)
    IsingGroup3[j]=BlockIsing(IsingGroup2[j],BLOCK)
IsingG0=[IsingGroup0,IsingGroup1,IsingGroup2,IsingGroup3]

for i in range(3000):
    for j in range(n):
        MCpass(IsingGroup[j],Tc)
    if (i+1)%STEP==0:
        #vE=np.zeros(4)
        vPhi=np.zeros(nRG)
        IsingTmp=[]
        for j in range(n):
            IsingTmp.append(IsingGroup[j])
        for m in range(nRG):
            if m>=1:
                for k in range(n):
                    IsingTmp[k]=BlockIsing(IsingTmp[k],BLOCK)
            vPhi[m] = funPhi(IsingTmp,IsingG0[m])
            outputphi[m].write(str(vPhi[m])+'\n')
            outputphi[m].flush()
'''

Ising = Initial()
S_alpha(Ising, 1)

for i in range(PRE_HEAT):  # pre_heat
    MCpass(Ising, Tc)
print('pre_heat finish')

IsingGroup = []
for step in range(TOTAL_STEP):
    tmp = Ising.copy()
    IsingGroup.append(tmp)
    MCpass(Ising, Tc)
print('MC finish')

outputfile = open('T3D_L=' + str(L) + '_b=' + str(BLOCK) + '_n=' + str(TOTAL_STEP) + '.txt', 'w')
for m in range(RG_ITERATION):
    A, B = matrixAB(IsingGroup)
    T = np.dot(np.linalg.inv(B), A)
    eigen, eigenv = np.linalg.eig(T)
    outputfile.write('T' + str(m + 1) + '=\n')
    outputfile.write(str(T) + '\n')
    outputfile.write('Eigen,Eigenv=\n')
    outputfile.write(str(eigen) + '\n')
    outputfile.write(str(eigenv) + '\n')
    for j in range(len(IsingGroup)):
        IsingGroup[j] = BlockIsing(IsingGroup[j], BLOCK)
