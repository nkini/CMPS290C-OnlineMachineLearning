import pickle
Z = pickle.load(open('Zvalues.pkl','rb'))

beta = []#to convert beta(t) into beta(t+1)
for t in range(200000):
    for q in range(t):
        b.append(1/Z[t]*(t-q))
    beta.append(b)

pickle.dump(beta,open('Beta_values.pkl','wb'),protocol=2)
