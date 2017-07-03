import pandas as pd
import numpy as np 
import scipy
from scipy.spatial import distance
import math
from sklearn.cluster import KMeans
import sklearn
import random
import math
import sys
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

a="/media/ryan/124C64CB4C64AB65/ACADEMICS/BKTRIPATHY/iris_data_set/iris.csv"
np.random.seed(42)
random.seed(1234)
df=pd.read_csv(a,sep=',')
data_array = np.array(df)
data_array=data_array[:,:4]
kk=KMeans(n_clusters=3)
kk.fit(data_array)
v=kk.cluster_centers_
v=np.array(v)
number_of_cluster=len(v)
Actions = np.array(range(len(v)))


e =2
m=2
epsa=0.1
wl=0.96
wb=0.04		
io=0		
results = []

sig=np.zeros((len(data_array)*len(data_array)))								
for i in data_array:
	for j in data_array:
		l=distance.euclidean(i,j)
		sig[io]=l**2
		io=io+1
sigma=sum(sig)/(2*(len(data_array)**2))

ind_data=np.zeros((len(data_array[0])))

def nebr(ind_data):        #finding the neighbouring points for ind_data
	z= np.zeros((len(data_array),len(data_array[0])))
	p= np.zeros((len(data_array)*len(v)))
	q=0
	o=0
	for l in v:
		p[q]=distance.euclidean(l,ind_data)
		q=q+1
	p=p[:q]
	thresh=min(p)/e   		#e is the input.Understand.
	#print thresh
	for i in data_array:
		if (distance.euclidean(i,ind_data)<=thresh and  distance.euclidean(i,ind_data)!=0):
			z[o]=i
			o=o+1
	z=z[:o]
	return z

def condProb(ind_data,cluster_number):    #Let P ( C i | x l ) be the conditional probability of x l being in C i.
	mn=0 											#condProb(ind_data,inn,v,m)
	prob=np.zeros((len(data_array)))
	for k in range(len(v)):
		mn=mn+((distance.euclidean(ind_data,v[cluster_number])/distance.euclidean(ind_data,v[k]))**(2/(m-1)))    #m is input.
	return 1/mn

def sim_cluster_sets(ind_data):
	t=0
	sim_clu=np.zeros((len(v)))
	for i in range(len(v)):
		if condProb(ind_data,i)>(1/float(len(v))):
			sim_clu[t]=i 
			t=t+1
	sim_clu=sim_clu[:t]

	return sim_clu

def lamda_aj_ci(i,j):
	if(i==Actions[j]):
		return 0
	else:
		return 1
                                                              ######UCI repository###########
def lamda_aj(j,sim_clu):
	if Actions[j] in sim_clu:
		return 0
	else:
		return 1

def beta(ind_data, nbr_ind_data):
	return math.exp(-(distance.euclidean(ind_data,nbr_ind_data))**2/(2*sigma))

def risk0(ind_data,j, sim_clu, kk,mm):    #risk associated with taking action  a j for x l   ##kk is nbrs   #j is ind cluster numbers #mm is label
	sec=np.zeros((len(data_array)))
	po=0
	#g=lamda_aj(j, sim_clu)
	for ii in kk:
		l=beta(ii, ind_data)       #ind_data is ii
		yy=sim_cluster_sets(ii)
		g=lamda_aj(j, yy)
		sec[po]=l*g
		po=po+1
	sec=sec[:po]
	secp=sum(sec)
	tt=lamda_aj_ci(mm, j)
	return (secp+tt)

def risk(ind_data, j,sim_clu, kk):
	g=np.zeros((len(data_array)))
	re=0
	for i in range(len(v)):
		l=condProb(ind_data, i)
        t=risk0(ind_data,j,sim_clu, kk, i)
        g[re]=l*t
        re=re+1
	g=g[:re]
	return sum(g)

def S(u):    #u is cluster number
	x=0
	q=0
	for i in g['low{0}'.format(u)]:
		x=x+distance.euclidean(i,v[u])
	return x/len(g['low{0}'.format(u)])

def dt(x,y):
	op=0
	dist=np.zeros(1000000)
	for i in x:
		for j in y:
			dist[op]=distance.euclidean(i,j)
			op=op+1
	dist=dist[:op]
	return min(dist)

def delta(x):
	op=0
	dist=np.zeros(1000000)
	for i in x:
		for j in x:
			if(set(i)!=set(j)):
				dist[op]=distance.euclidean(i,j)
				op=op+1
	dist=dist[:op]
	return max(dist)

g=globals()
ty=0
opo=0
DB0=np.zeros(100)
ioi=0
Dunn=np.zeros(100)
mo=0

for loop in range(0,10):
	print v
	
	for i in range(len(v)):
		g['low{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))
		g['low_x_prob{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))
		g['low_prob{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))

		g['up{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))
		g['up_x_prob{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))
		g['up_prob{0}'.format(i)]=np.zeros((len(data_array),len(data_array[0])))

		g['l{0}'.format(i)]=0
		g['u{0}'.format(i)]=0
		g['b{0}'.format(i)]=0

	for ll in range(len(data_array)):
		ind_data = data_array[ll]
		lll=0
		pro=np.zeros((len(v)))
		for inn in range(len(v)):
			pro[lll] = condProb(ind_data,inn)
			lll=lll+1
		pro=pro[:lll]
		nbrs = np.zeros((len(data_array),len(data_array[0])))
		nbrs = nebr(ind_data)
		#print nbrs
		sim_clstr = sim_cluster_sets(ind_data)
		j=0
		mk=0
		riskk=np.zeros((len(v)))
		for j in range(len(v)):
			riskk[j] = risk(ind_data,j, sim_clstr,nbrs)
		riskk=riskk[:len(v)]
		akk=np.argmin(riskk)  #index with min value
	
		g['up{0}'.format(akk)][g['u{0}'.format(akk)]]=ind_data
		g['up_prob{0}'.format(akk)][g['u{0}'.format(akk)]]= condProb(ind_data,akk)**2
		g['up_x_prob{0}'.format(akk)][g['u{0}'.format(akk)]]= (condProb(ind_data,akk)**2)*ind_data
		g['u{0}'.format(akk)]=g['u{0}'.format(akk)]+1

		JD=np.zeros((len(data_array)))
		n=np.zeros((len(data_array)), dtype=int)
		po=np.zeros((len(data_array)))
		p=0
		l=0
		for qq in riskk:
			l=l+1                   
			if qq!=riskk[akk] and qq/float(riskk[akk])<= (1+epsa):
				JD[p]=qq/float(riskk[akk])
				n[p]=l-1
				p=p+1
		
		JD=JD[:p]
		n=n[:p]	
	
		if p==0:
			g['low{0}'.format(akk)][g['l{0}'.format(akk)]]=ind_data
			g['low_prob{0}'.format(akk)][g['l{0}'.format(akk)]]= condProb(ind_data,akk)**2
			g['low_x_prob{0}'.format(akk)][g['l{0}'.format(akk)]]= (condProb(ind_data,akk)**2)*ind_data
			g['l{0}'.format(akk)]=g['l{0}'.format(akk)]+1

		else:
			for i in n:
				g['up{0}'.format(i)][g['u{0}'.format(i)]]=ind_data
				g['up_prob{0}'.format(i)][g['u{0}'.format(i)]]= condProb(ind_data,i)**2
				g['up_x_prob{0}'.format(i)][g['u{0}'.format(i)]]= (condProb(ind_data,i)**2)*ind_data
				g['u{0}'.format(i)]=g['u{0}'.format(i)]+1

	for ii in range(len(v)):
		g['low{0}'.format(ii)]=g['low{0}'.format(ii)][:g['l{0}'.format(ii)]]
		g['low_prob{0}'.format(ii)]=g['low_prob{0}'.format(ii)][:g['l{0}'.format(ii)]]
		g['low_x_prob{0}'.format(ii)]=g['low_x_prob{0}'.format(ii)][:g['l{0}'.format(ii)]]

		g['up{0}'.format(ii)]=g['up{0}'.format(ii)][:g['u{0}'.format(ii)]]
		g['up_prob{0}'.format(ii)]=g['up_prob{0}'.format(ii)][:g['u{0}'.format(ii)]]
		g['up_x_prob{0}'.format(ii)]=g['up_x_prob{0}'.format(ii)][:g['u{0}'.format(ii)]]

		g['bdr{0}'.format(ii)]=np.zeros((len(g['up{0}'.format(ii)]),len(data_array[0])))
		g['bdr_prob{0}'.format(ii)]=np.zeros((len(g['up{0}'.format(ii)]),len(data_array[0])))
		g['bdr_x_prob{0}'.format(ii)]=np.zeros((len(g['up{0}'.format(ii)]),len(data_array[0])))

		po=0
		for i in g['up{0}'.format(ii)]:
			a=0
			for j in g['low{0}'.format(ii)]:
				if set(i)==set(j):
					a=a+1
					break
			if a==0:
				g['bdr{0}'.format(ii)][po]=i
				g['bdr_prob{0}'.format(ii)][po]=condProb(i,ii)**2
				g['bdr_x_prob{0}'.format(ii)][po]=i*(condProb(i,ii)**2)
				po=po+1

		g['bdr{0}'.format(ii)]=g['bdr{0}'.format(ii)][:po]
		g['bdr_prob{0}'.format(ii)]=g['bdr_prob{0}'.format(ii)][:po]
		g['bdr_x_prob{0}'.format(ii)]=g['bdr_x_prob{0}'.format(ii)][:po]


	v_temp=np.zeros((len(v),len(data_array[0])))
	v_temp[:]=v[:]

	for ii in range(len(v)):
		if(len(g['low{0}'.format(ii)])!=0 and len(g['bdr{0}'.format(ii)])!=0):
			v[ii]=wl*(g['low{0}'.format(ii)].sum(axis=0))/len(g['low{0}'.format(ii)]) + wb*(g['bdr{0}'.format(ii)].sum(axis=0))/len(g['bdr{0}'.format(ii)])
		elif (len(g['low{0}'.format(ii)])==0 and len(g['bdr{0}'.format(ii)])!=0):
			v[ii]=(g['bdr{0}'.format(ii)].sum(axis=0))/(len(g['bdr{0}'.format(ii)]))
		else:
			v[ii]=(g['low{0}'.format(ii)].sum(axis=0))/(len(g['low{0}'.format(ii)]))


	if ((v==v_temp).all()):
		print "match"
		break



#db index
dbl=np.zeros(len(data_array))
DB=np.zeros(len(v))

rt=0
for i in range(len(v)-1):
	dd=0
	for j in range(i+1,len(v)):
		dbl[dd]=(S(i)+S(j))/float(distance.euclidean(v[i],v[j]))
		#print dbl[dd]
		dd=dd+1
	dbl=dbl[:dd]
	#print dbl
	DB[rt]=max(dbl)					#pavitra majhi , sk pal
	rt=rt+1
DB=DB[:rt]
#print sum(DB)
DB0[ioi]=float(sum(DB)/float(len(v)))
print DB0[ioi]
'''
#dunn index

dell=np.zeros(100000)
uio=0
for i in range(len(v)):
	dell[uio]=delta(g['low{0}'.format(i)])
	uio=uio+1

dell=dell[:uio]
#print dell
dell_max=max(dell)
l=np.argmax(dell)


lo=np.zeros(100000)
lof=np.zeros(100000)
yuyu=0

ipf=0
for i in range(len(v)):
	ip=0
	for j in range(len(v)):
		if(i==j):
			continue
		#print dt(g['low{0}'.format(i)],g['low{0}'.format(j)])
		lo[ip]=float(distance.euclidean(v[i],v[j]))/float(dell_max)
		ip=ip+1
		#print lo[ip-1]
	lo=lo[:ip]
	lof[ipf]=min(lo)
	ipf=ipf+1
	#print lo
lof=lof[:ipf]	
lo_min=min(lof)



Dunn[mo]=lo_min
mo=mo+1

Dunn=Dunn[:mo]

'''

dell=np.zeros(100000)
uio=0
for i in range(len(v)):
	dell[uio]=delta(g['low{0}'.format(i)])
	uio=uio+1

dell=dell[:uio]
#print dell
dell_max=max(dell)
l=np.argmax(dell)
#print dell_max_ind

#print dell_max

lo=np.zeros(100000)
lo_min=np.zeros(100000)
yuyu=0
for i in range(l+1):
	ip=0
	for j in range(l+1):
		#print dt(g['low{0}'.format(i)],g['low{0}'.format(j)])
		lo[ip]=dt(g['low{0}'.format(i)],g['low{0}'.format(j)])/float(dell_max)
		ip=ip+1
		#print lo[ip-1]
	lo=lo[:ip]
	#print lo
	lo_min[yuyu]=max(lo)
	yuyu=yuyu+1
lo_min=lo_min[:yuyu]
lo_min_min=max(lo_min)

lo=lo[:ip]
print lo
Dunn[mo]=lo_min_min
mo=mo+1

Dunn=Dunn[:mo]

print "Dunn = {}".format(Dunn)


#plot points

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

for i in range(len(v)):
	g['low{0}'.format(i)]=g['low{0}'.format(i)][:,:3]
        g['low{0}'.format(i)]=g['low{0}'.format(i)].T


X0=low0[0]
Y0=low0[1]
Z0=low0[2]

X1=low1[0]
Y1=low1[1]
Z1=low1[2]

X2=low2[0]
Y2=low2[1]
Z2=low2[2]

XV0=v[0][0]
YV0=v[0][1]
ZV0=v[0][2]

XV1=v[1][0]
YV1=v[1][1]
ZV1=v[1][2]

XV2=v[2][0]
YV2=v[2][1]
ZV2=v[2][2]


ax.scatter(XV0,YV0,ZV0,c='r',marker='^',s=200)		
ax.scatter(XV1,YV1,ZV1,c='b',marker='^',s=200)		
ax.scatter(XV2,YV2,ZV2,c='g',marker='^',s=200)

ax.scatter(X0,Y0,Z0,c='r',marker='o',)		
ax.scatter(X1,Y1,Z1,c='b',marker='o')		
ax.scatter(X2,Y2,Z2,c='g',marker='o')


ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')

plt.show()		


print v

print len(low0)
print len(low1)
print len(low2)
print len(up0)
print len(up1)
print len(up2)
print len(bdr0)
print len(bdr1)
print len(bdr2)

















