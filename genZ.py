def getZ(t):
    global Z
    #for q in range(0,t):
    z = Z[t-1]+1/t
    return z

Z = [0]
for i in range(1,200000):
    Z.append(getZ(i))

for i in range(10):
    print(Z[i])

import pickle
pickle.dump(Z,open('Zvalues.pkl','wb'))
