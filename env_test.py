import numpy as np
import tracking.quadenv as car
import time
import matplotlib.pyplot as plt
test = car.quad_env()

init = test.reset()
action = np.random.uniform(0,20,[4,])
t = time.time()
count = 0
while 1:
    predict1 = init + np.array(test.quad.next_state(init, action, 0) * test.quad.dt)
    predict2 = np.squeeze(test._predict_next_obs_uncertainty(init, action.reshape([1, 4, 1])))
    err1 = (np.square(predict1 - predict2))
    count += 1
    if max(err1) > 1:
        np.save('state.npy',init)
        np.save('action.npy',action)
        break
    init = predict1
    action = np.clip(np.random.uniform(0,100,[4,]),0,100)
print(count)
pass