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
num_experts = 10
trials = 5500
eta = 4.0
alpha = 0.08
sdc = 10

#Experts  spaced linearly, harmonically, or exponentially
#Use 100 experts with time out value from  0 to spin_down_cost
lin_experts = np.linspace(0,sdc,10)
exp_experts = np.logspace(0.01, 1, num=10, endpoint=True)

w_temp = []
w = []
def plot_losses_fixed_share_to_uniform_past(idletimes,experts):
    global w_temp,w
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
        w = np.vstack((w,w_temp))
        losses = np.vstack((losses,loss))
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open('cum_losses_FSUP_expexp_100k.pkl','w'))
    #plt.figure(figsize=(20, 12), dpi=80)

    #for i in range(len(experts)):
    #    plt.plot(cum_losses[:,i],label='fixed timeout {p}'.format(p=experts[i]))
        
    #plt.legend(loc='best')
    #plt.savefig('losses_fixed_share_to_uniform_past.png')


Z = pickle.load(open('Zvalues.py2.pkl'))

def beta(t):
    global Z,alpha
    return np.array([[alpha/(t-q)*Z[t] for q in range(t)]])
    
w_temp = []
w = []
def plot_losses_fixed_share_to_decaying_past(idletimes,experts):
    global w_temp,w
    num_experts = len(experts)
    w_init = np.ones(num_experts) * 1/num_experts
    w = np.array([w_init])
    losses = np.array(np.zeros(num_experts))
    t = 0
    
    for idletime in idletimes:
        loss = np.array([idletime if timeout >= idletime else timeout+sdc for timeout in experts])
        w_temp = np.exp(-eta*loss)*w[t]
        t += 1
        w_temp = (1-alpha)*w_temp/sum(w_temp) + np.dot(beta(t),w)
        #alpha*sigma_q_0_to_t((1/(t-q))*(1/Z[t]))#(sum(w[:,:]))/i)
        w = np.vstack((w,w_temp))
        losses = np.vstack((losses,loss))
        
    cum_losses = np.cumsum(losses,axis=0)
    pickle.dump(cum_losses,open('cum_losses_FSDP_expexp_10k.pkl','w'))


#plot_losses_fixed_share_to_uniform_past(cello2_data[:100000],lin_experts)
#plot_losses_fixed_share_to_uniform_past(cello2_data[:100000],exp_experts)
plot_losses_fixed_share_to_decaying_past(cello2_data[:10000],lin_experts)
plot_losses_fixed_share_to_decaying_past(cello2_data[:10000],exp_experts)
