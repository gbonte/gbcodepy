
import numpy as np
import matplotlib.pyplot as plt
p=np.arange(0,1,0.001)

import matplotlib.pyplot as plt
plt.plot(p,p*(1-p),label='Variance')

plt.plot(p,-p*np.log2(p)-(1-p)*np.log2(1-p),label='Entropy')
plt.title('Binary 0/1 r.v.')
plt.xlabel('PROB(y=1)')
plt.legend()
plt.show()
