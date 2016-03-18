import numpy as np
import functools
#from matplotlib import pyplot as plt
import pickle

f = open('cello2.txt')
cello2_data = []
try:
    for line in f:
         cello2_data.append(float(line.strip('\n'))/1000) #data is in seconds
except ValueError as v:
    print(line)

cello2_data = np.array(cello2_data)

# parameters
#num_experts = 10
#trials = 5500
eta = 4.0
alpha = 0.08
sdc = 10

#Experts  spaced linearly, harmonically, or exponentially
#Use 100 experts with time out value from  0 to spin_down_cost
lin_experts = np.linspace(0,sdc,10)
exp_experts = np.logspace(0.01, 1, num=10, endpoint=True)

'''
% Calculate loss %

   master_pred = dot(wmat(:,t),exp_pred)/sum(wmat(:,t));

    % For master
    if idletimes(t) < master_pred
        loss_master(t) = idletimes(t);
    else
        loss_master(t) = master_pred + spin_down_cost;
    end

    % For optimal
    optimal = min(idletimes(t),spin_down_cost);
    loss_optimal(t) = optimal;
'''
def get_losses_master(idletimes,experts,setting):
    losses = []
    weights = pickle.load(open('plottables/weights_'+setting+'_100k.pkl'))
    for i in range(len(idletimes)):
        master_pred = np.dot(weights[i],experts)
        if idletimes[i] < master_pred:
            losses.append(idletimes[i])
        else:
            losses.append(master_pred + sdc)
    
    cum_losses = np.cumsum(losses)
    pickle.dump(cum_losses,open('plottables/cum_losses_master_'+setting+'_100k.pkl','w'))
    

def get_losses_optimal(idletimes):
    global sdc
    losses_optimal = []
    for idletime in idletimes:
        losses_optimal.append(min(idletime,sdc))

    cum_losses = np.cumsum(losses_optimal)
    pickle.dump(cum_losses,open('plottables/cum_losses_optimal_100k.pkl','w'))

    

def get_losses_static_expert(idletimes,experts,filename):
    num_experts = len(experts)
    w_init = np.ones(num_experts) * 1/num_experts
    w = np.array(w_init)
    losses = np.array(np.zeros(num_experts))
    i = 1
    
    for idletime in idletimes:
        loss = np.array([idletime if timeout >= idletime else timeout+sdc for timeout in experts])
        w = np.vstack((w,np.exp(-eta*loss)*w[i-1]))
        w[i] = w[i]/sum(w[i])
        losses = np.vstack((losses,loss))
        i += 1
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open(filename,'w'))
    pickle.dump(w,open(filename.replace('cum_losses','weights'),'w'))


def get_losses_fixed_share_to_start_vector(idletimes,experts,filename):
    num_experts = len(experts)
    w_init = np.ones(num_experts) * 1/num_experts
    w = np.array(w_init)
    losses = np.array(np.zeros(num_experts))
    i = 0
    
    for idletime in idletimes:
        loss = np.array([idletime if timeout >= idletime else timeout+sdc for timeout in experts])
        w_temp = np.exp(-eta*loss)*w[i]
        w_temp = (1-alpha)*w_temp/sum(w_temp) + alpha*(w_init)
	if not np.allclose(sum(w_temp), 1.0, rtol=1e-05, atol=1e-08):
	    print("weights may not sum to 1",i,sum(w_temp))
        w = np.vstack((w,w_temp))
        losses = np.vstack((losses,loss))
        i += 1
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open(filename,'w'))
    pickle.dump(w,open(filename.replace('cum_losses','weights'),'w'))


def get_losses_fixed_share_to_uniform_past(idletimes,experts,filename):
    num_experts = len(experts)
    w_init = np.ones(num_experts) * 1/num_experts
    w_sum = np.zeros(num_experts)
    w = np.array([w_init])
    losses = np.array(np.zeros(num_experts))
    i = 0
    
    for idletime in idletimes:
        loss = np.array([idletime if timeout >= idletime else timeout+sdc for timeout in experts])
        w_temp = np.exp(-eta*loss)*w[i]
        w_sum += w[i]
        i += 1
        w_temp = (1-alpha)*w_temp/sum(w_temp) + alpha*(w_sum)/i
	if not np.allclose(sum(w_temp), 1.0, rtol=1e-05, atol=1e-08):
	    print("weights may not sum to 1",i,sum(w_temp))
        w = np.vstack((w,w_temp))
        losses = np.vstack((losses,loss))
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open(filename,'w'))
    pickle.dump(w,open(filename.replace('cum_losses','weights'),'w'))

from pprint import pprint

def beta(t):
    global Z,alpha
    return np.array([[1.0/((t-q)*Z[t]) for q in range(t)]])
    
def get_losses_fixed_share_to_decaying_past(idletimes,experts,filename):
    num_experts = len(experts)
    w_init = np.ones(num_experts) * 1/num_experts
    w = np.array([w_init])
    losses = np.array(np.zeros(num_experts))
    t = 0
    
    for idletime in idletimes:
        loss = np.array([idletime if timeout >= idletime else timeout+sdc for timeout in experts])
        w_temp = np.exp(-eta*loss)*w[t]
        t += 1
        w_temp = (1-alpha)*w_temp/sum(w_temp) + alpha*np.dot(beta(t),w)
	#for some reason the line above returns an array of the form a = [[1,2,3]]
	if not np.allclose(sum(w_temp[0]), 1.0, rtol=1e-05, atol=1e-08):
	    print("weights may not sum to 1",t,sum(w_temp[0]))
        w = np.vstack((w,w_temp))
        losses = np.vstack((losses,loss))
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open(filename,'w'))
    pickle.dump(w,open(filename.replace('cum_losses','weights'),'w'))

num_data = 100000
'''
print('cum_losses_SE_linexp_100k')
get_losses_static_expert(cello2_data[:num_data],lin_experts,'cum_losses_SE_linexp_100k.pkl')
print('cum_losses_SE_expexp_100k')
get_losses_static_expert(cello2_data[:num_data],exp_experts,'cum_losses_SE_expexp_100k.pkl')
print('cum_losses_FSSV_linexp_100k')
get_losses_fixed_share_to_start_vector(cello2_data[:num_data],lin_experts,'cum_losses_FSSV_linexp_100k.pkl')
print('cum_losses_FSSV_linexp_100k')
get_losses_fixed_share_to_start_vector(cello2_data[:num_data],exp_experts,'cum_losses_FSSV_expexp_100k.pkl')
print('cum_losses_FSUP_linexp_100k')
get_losses_fixed_share_to_uniform_past(cello2_data[:num_data],lin_experts,'cum_losses_FSUP_linexp_100k.pkl')
print('cum_losses_FSUP_expexp_100k')
get_losses_fixed_share_to_uniform_past(cello2_data[:num_data],exp_experts,'cum_losses_FSUP_expexp_100k.pkl')
Z = pickle.load(open('Zvalues.py2.pkl'))
print('cum_losses_FSDP_linexp_100k')
get_losses_fixed_share_to_decaying_past(cello2_data[:num_data],lin_experts,'cum_losses_FSDP_linexp_100k.pkl')
print('cum_losses_FSDP_expexp_100k')
get_losses_fixed_share_to_decaying_past(cello2_data[:num_data],exp_experts,'cum_losses_FSDP_expexp_100k.pkl')

print('cum_master_losses_SE_linexp_100k')
get_losses_master(cello2_data[:num_data],lin_experts,'SE_linexp')
print('cum_master_losses_SE_expexp_100k')
get_losses_master(cello2_data[:num_data],exp_experts,'SE_expexp')
print('cum_master_losses_FSSV_linexp_100k')
get_losses_master(cello2_data[:num_data],lin_experts,'FSSV_linexp')
print('cum_master_losses_FSSV_expexp_100k')
get_losses_master(cello2_data[:num_data],exp_experts,'FSSV_expexp')
print('cum_master_losses_FSUP_linexp_100k')
get_losses_master(cello2_data[:num_data],lin_experts,'FSUP_linexp')
print('cum_master_losses_FSUP_expexp_100k')
get_losses_master(cello2_data[:num_data],exp_experts,'FSUP_expexp')
print('cum_master_losses_FSDP_linexp_100k')
get_losses_master(cello2_data[:num_data],lin_experts,'FSDP_linexp')
print('cum_master_losses_FSDP_expexp_100k')
get_losses_master(cello2_data[:num_data],exp_experts,'FSDP_expexp')
'''

print('cum_optimal_losses_SE_linexp_100k')
get_losses_optimal(cello2_data[:num_data])
