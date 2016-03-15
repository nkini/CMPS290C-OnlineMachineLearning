def getZ(t):
    z = 0      
    for q in range(0,t):
        z += 1/(t-q)
    return z

Z = []
for i in range(20):
    Z.append(getZ(i))

for i in range(10):
    print(Z[i])

import pickle
pickle.dump(Z,open('Zvalues.pkl','wb'))
