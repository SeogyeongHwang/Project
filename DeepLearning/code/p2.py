import numpy as np
import matplotlib.pyplot as plt

X = np.load('canvas_causal_X.npy')
y = np.load('canvas_causal_y.npy')

# 1) Find the solution using the linear regression (minimum least squared) method.
# Add bias into weights
X_p = np.ones((X.shape[0], X.shape[1] + 1))
X_p[:, 0:-1] = X

# Training
W = np.linalg.inv(X_p.T @ X_p) @ X_p.T @ y
print(W)

l = np.mean((y - X_p @ W) ** 2)
print('l = ', l)

# 2) Find the solution using gradient descent method with for various step size.
n = X.shape[0]
step_size = 0.000001
W_1 = np.zeros((100001, X_p.shape[1]))
W_1[0, :] = np.random.rand(X_p.shape[1])
l_pre = np.mean((y - X_p @ W_1[0, :]) ** 2)
for i in range(100000):
    grad = -(2 / n) * (y - (X_p @ W_1[i, :])) @ X_p
    W_1[i+1, :] = W_1[i, :] - (step_size * grad)
    l = np.mean((y - X_p @ W_1[i+1, :]) ** 2)
    if l_pre - l < 0.0001:
        W_1[i+2:, :] = W_1[i+1, :]
        break
    l_pre = l
np.save("step_0.000001.npy", W_1)

# 3) Draw the estimated parameters for each step of gradient descent.
W0 = np.load('step_0.0000001.npy')
W1 = np.load('step_0.0000002.npy')
W2 = np.load('step_0.0000005.npy')
W3 = np.load('step_0.000001.npy')
W4 = np.load('step_0.000005.npy')

plt.xlabel('w_0')
plt.ylabel('w_1')
plt.plot(W4[:, 0], W4[:, 1], 'r', label='step=0.000005')
plt.plot(W3[:, 0], W3[:, 1], 'g',label='step=0.000001')
plt.plot(W2[:, 0], W2[:, 1], 'b', label='step=0.0000005')
plt.plot(W1[:, 0], W1[:, 1], 'c', label='step=0.0000002')
plt.plot(W0[:, 0], W0[:, 1], 'm', label='step=0.0000001')

plt.plot(W4[::1000, 0], W4[::1000, 1], 'r.')
plt.plot(W3[::1000, 0], W3[::1000, 1], 'g.')
plt.plot(W2[::1000, 0], W2[::1000, 1], 'b.')
plt.plot(W1[::1000, 0], W1[::1000, 1], 'c.')
plt.plot(W0[::1000, 0], W0[::1000, 1], 'm.')
plt.legend()
plt.show()

plt.xlabel('w_2')
plt.ylabel('w_3')
plt.plot(W4[:, 2], W4[:, 3], 'r', label='step=0.000005')
plt.plot(W3[:, 2], W3[:, 3], 'g',label='step=0.000001')
plt.plot(W2[:, 2], W2[:, 3], 'b', label='step=0.0000005')
plt.plot(W1[:, 2], W1[:, 3], 'c', label='step=0.0000002')
plt.plot(W0[:, 2], W0[:, 3], 'm', label='step=0.0000001')


plt.plot(W4[::1000, 2], W4[::1000, 3], 'r.')
plt.plot(W3[::1000, 2], W3[::1000, 3], 'g.')
plt.plot(W2[::1000, 2], W2[::1000, 3], 'b.')
plt.plot(W1[::1000, 2], W1[::1000, 3], 'c.')
plt.plot(W0[::1000, 2], W0[::1000, 3], 'm.')
plt.legend()
plt.show()

# 4) We can roughly find out the optimal step size for each iteration by attempting multiple stepsize
# simultaneously as the following figure. Find out the efficiency of gradient descent method for various step sizes.
n = X_p.shape[0]
W = np.random.rand(X_p.shape[1])
grad = -2 / n * (y - (X_p @ W)) @ X_p

step_size = np.linspace(0.0000001, 0.000015, 2000)

W_new = np.zeros((step_size.size, X_p.shape[1]))
loss = np.zeros(step_size.size)
for i in range(step_size.size):
    W_new[i, :] = W - step_size[i] * grad
    loss[i] = np.mean((y - X_p @ W_new[i, :]) ** 2)

plt.xlabel("step_size(x10^-6)")
plt.ylabel("loss")
plt.plot(step_size*10**6, loss, '-')
plt.show()

# 5) Build your own method by adjusting the step size to maximize the performance.
###########
#Causal
X = np.load('canvas_causal_X.npy')
y = np.load('canvas_causal_y.npy')

X_p = np.ones((X.shape[0], X.shape[1] + 1))
X_p[:, 0:-1] = X

n = X.shape[0]
W_1 = np.zeros((100001, X_p.shape[1]))
W_1[0, :] = np.random.rand(X_p.shape[1])
W_2 = np.zeros((3, X_p.shape[1]))
l = np.zeros(3)
l_pre = np.mean((y - X_p @ W_1[0, :]) ** 2)
step_size = np.array([0.000005, 0.000007, 0.000009])
for i in range(100000):
    grad = -(2 / n) * (y - (X_p @ W_1[i, :])) @ X_p
    for j in range(3):
        W_2[j, :] = W_1[i, :] - (step_size[j] * grad)
        l[j] = np.mean((y - X_p @ W_2[j, :]) ** 2)
    min = np.argmin(l)
    W_1[i+1, :] = W_2[min,:]
    if l_pre - l[min] < 0.0001:
        print(i+1)
        W_1[i+2:, :] = W_1[i+1, :]
        break
    l_pre = l[min]
np.save("my_step_causal.npy", W_1)

plt.xlabel('w_0')
plt.ylabel('w_1')
plt.plot(W_1[:, 0], W_1[:, 1], 'c', label='my_step')
plt.plot(W_1[::1000, 0], W_1[::1000, 1], 'c.')
plt.legend()
plt.show()

plt.xlabel('w_2')
plt.ylabel('w_3')
plt.plot(W_1[:, 2], W_1[:, 3], 'c', label='my step')
plt.plot(W_1[::1000, 2], W_1[::1000, 3], 'c.')
plt.legend()
plt.show()

####################
# Non-causal
X = np.load('canvas_non-causal_X.npy')
y = np.load('canvas_non-causal_y.npy')

n = X.shape[0]
W_1 = np.zeros((100001, X_p.shape[1]))
W_1[0, :] = np.random.rand(X_p.shape[1])
W_2 = np.zeros((3, X_p.shape[1]))
l = np.zeros(3)
l_pre = np.mean((y - X_p @ W_1[0, :]) ** 2)
step_size = np.array([0.000002, 0.000004, 0.000006])
for i in range(100000):
    grad = -(2 / n) * (y - (X_p @ W_1[i, :])) @ X_p
    for j in range(3):
        W_2[j, :] = W_1[i, :] - (step_size[j] * grad)
        l[j] = np.mean((y - X_p @ W_2[j, :]) ** 2)
    min = np.argmin(l)
    W_1[i+1, :] = W_2[min,:]
    if l_pre - l[min] < 0.0001:
        print(i+1)
        W_1[i+2:, :] = W_1[i+1, :]
        print(W_1[i+1, :])
        break
    l_pre = l[min]
np.save("my_step_non-causal.npy", W_1)

plt.xlabel('w_0')
plt.ylabel('w_1')
plt.plot(W_1[:, 0], W_1[:, 1], 'c', label='my_step')
plt.plot(W_1[::1000, 0], W_1[::1000, 1], 'c.')
plt.legend()
plt.show()

plt.xlabel('w_2')
plt.ylabel('w_3')
plt.plot(W_1[:, 2], W_1[:, 3], 'c', label='my step')
plt.plot(W_1[::1000, 2], W_1[::1000, 3], 'c.')
plt.legend()
plt.show()

plt.xlabel('w_4')
plt.ylabel('w_5')
plt.plot(W_1[:, 4], W_1[:, 5], 'c', label='my_step')
plt.plot(W_1[::1000, 4], W_1[::1000, 5], 'c.')
plt.legend()
plt.show()

plt.xlabel('w_6')
plt.ylabel('w_7')
plt.plot(W_1[:, 6], W_1[:, 7], 'c', label='my step')
plt.plot(W_1[::1000, 6], W_1[::1000, 7], 'c.')
plt.legend()
plt.show()
