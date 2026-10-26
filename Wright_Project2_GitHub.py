# %% [markdown]
# Kennamer Wright <br>
# 12113153

# %% [markdown]
# # 1.1 Exercise 1

# %% [markdown]
# **1.1.2**

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def rk4(fx, t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros((len(t_values), len(y0)))

    y_values[0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[i]

        k1 = fx(t, y, a, b)
        k2 = fx(t + h/2, y + (h/2)*k1, a, b)
        k3 = fx(t + h/2, y + (h/2)*k2, a, b)
        k4 = fx(t + h, y + h*k3, a, b)

        y_values[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t_values, y_values

# %%
def fx(t, y, a, b):
    x = y[0]
    y_val = y[1]

    dxdt = a*x - b*y_val
    dydt = b*x + a*y_val

    return np.array([dxdt, dydt])

# %%
t0 = 0
y0 = np.array([0.5, 0.8])

t_end = 10
h = 0.1

t_values, y_values = rk4(fx, t0, y0, t_end, h)

# %%
# test case 1
a = -5
b = 3
t1, y1 = rk4(fx, t0, y0, t_end, h)

# test case 2
a = 5
b = 3
t2, y2 = rk4(fx, t0, y0, t_end, h)

# test case 3
a = 0.2
b = 3
t3, y3 = rk4(fx, t0, y0, t_end, h)

# %%
# test case 1 graph
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(t1, y1[:, 0], label='Case 1')
axs[1].plot(t1, y1[:, 1], label='Case 1')

axs[0].set_title('x(t) - Case 1')
axs[1].set_title('y(t) - Case 1')

plt.tight_layout()
plt.show()

# %%
# test case 2 graph
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(t2, y2[:, 0], label='Case 2')
axs[1].plot(t2, y2[:, 1], label='Case 2')

axs[0].set_title('x(t) - Case 2')
axs[1].set_title('y(t) - Case 2')

plt.tight_layout()
plt.show()

# %%
# test case 3 graph
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(t3, y3[:, 0], label='Case 3')
axs[1].plot(t3, y3[:, 1], label='Case 3')

axs[0].set_title('x(t) - Case 3')
axs[1].set_title('y(t) - Case 3')

plt.tight_layout()
plt.show()

# %% [markdown]
# --------------------------

# %% [markdown]
# # 1.2 Exercise 2

# %% [markdown]
# **1.2.1**

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def f(X):
    x = X[0]
    y = X[1]

    a = 1.5
    b = 1.0
    c = 0.5
    d = 0.5

    return x**2 + y**2 - a*np.exp(-((x-1)**2 + y**2)/c) - b*np.exp(-((x+1)**2 + y**2)/d)

# %%
def g(X):
    x = X[0]
    y = X[1]

    a = 1
    b = 2

    return (a - x)**2 + b*(y - x**2)**2

# %%
def grad_f(X):
    x = X[0]
    y = X[1]

    a = 1.5
    b = 1.0
    c = 0.5
    d = 0.5

    df_dx = 2*x + (2*a*(x-1)/c)*np.exp(-((x-1)**2 + y**2)/c) + (2*b*(x+1)/d)*np.exp(-((x+1)**2 + y**2)/d)
    df_dy = 2*y + (2*a*y/c)*np.exp(-((x-1)**2 + y**2)/c) + (2*b*y/d)*np.exp(-((x+1)**2 + y**2)/d)

    return np.array([df_dx, df_dy])

# %%
def grad_g(X):
    x = X[0]
    y = X[1]

    a = 1
    b = 2

    dg_dx = -2*(a - x) - 4*b*x*(y - x**2)
    dg_dy = 2*b*(y - x**2)

    return np.array([dg_dx, dg_dy])

# %%
def grad_descent(grad_f, init_point, max_iter=100, alpha=0.05, tol=1e-6):
    convergent_points = [init_point]
    Xk = init_point

    for i in range(max_iter):
        grad = grad_f(Xk)

        Xk = Xk - alpha * grad
        convergent_points.append(Xk)
        last_norm = np.linalg.vector_norm(grad)

        if last_norm < tol:
            break
    
    return convergent_points, last_norm

# %%
def grad_descent_with_momentum(grad_f, init_point, max_iter=100, alpha=0.05, tol=1e-6, beta=0.9):
    convergent_points = [init_point]
    
    v = np.array([0, 0])
    Xk = init_point
    for i in range(max_iter):
        grad = grad_f(Xk)
        
        v = beta * v + (1 - beta) * grad
        Xk = Xk - alpha * v
        
        convergent_points.append(Xk)
        last_norm = np.linalg.vector_norm(grad)

        if last_norm < tol:
            break

    return convergent_points, last_norm

# %%
def adam_grad(grad_f, init_point, max_iter=100, alpha=0.05, beta1=0.9, beta2=0.999, tol=1e-6, epsilon=1e-8):
    convergent_points = [init_point]
    Xk = init_point

    m = np.array([0, 0])
    v = np.array([0, 0])

    for i in range(max_iter):
        grad = grad_f(Xk)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**(i+1))
        v_hat = v / (1 - beta2**(i+1))

        Xk = Xk - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        convergent_points.append(Xk)
        last_norm = np.linalg.vector_norm(grad)

        if last_norm < tol:
            break
        
    return convergent_points, last_norm

# %%
init_point = np.array([-2, 2])

gd_path, gd_norm = grad_descent(grad_f, init_point)
mom_path, mom_norm = grad_descent_with_momentum(grad_f, init_point)
adam_path, adam_norm = adam_grad(grad_f, init_point)

# %%
xs = np.linspace(-3, 3, 100)
ys = np.linspace(-3, 3, 100)

# %%
x_vals, y_vals = np.meshgrid(xs, ys)

# %%
f_vals = np.zeros_like(x_vals)

for i in range(len(xs)):
    for j in range(len(ys)):
        f_vals[j, i] = f(np.array([x_vals[j, i], y_vals[j, i]]))

# %%
gd_path = np.array(gd_path)
mom_path = np.array(mom_path)
adam_path = np.array(adam_path)

# %%
fig, ax = plt.subplots(figsize=(16, 9))

ax.contour(x_vals, y_vals, f_vals, levels=150, cmap='jet')

ax.plot(*zip(*gd_path), marker='o', label='Gradient Descent')
ax.plot(*zip(*mom_path), marker='o', label='Momentum')
ax.plot(*zip(*adam_path), marker='o', label='Adam')

ax.set_title(f'Last norms: GD={round(gd_norm, 3)}, MOM={round(mom_norm, 3)}, ADAM={round(adam_norm, 3)}')
ax.legend()

plt.show()

# %%
init_point = np.array([-2, 2])

gd_path_g, gd_norm_g = grad_descent(grad_g, init_point, max_iter=500, alpha=0.005)
mom_path_g, mom_norm_g = grad_descent_with_momentum(grad_g, init_point, max_iter=500, alpha=0.005)
adam_path_g, adam_norm_g = adam_grad(grad_g, init_point, max_iter=500, alpha=0.05)

# %%
xs = np.linspace(-3, 3, 100)
ys = np.linspace(-3, 3, 100)

# %%
x_vals, y_vals = np.meshgrid(xs, ys)

# %%
g_vals = np.zeros_like(x_vals)

for i in range(len(xs)):
    for j in range(len(ys)):
        g_vals[j, i] = g(np.array([x_vals[j, i], y_vals[j, i]]))

# %%
fig, ax = plt.subplots(figsize=(16, 9))

ax.contour(x_vals, y_vals, g_vals, levels=150, cmap='plasma')

ax.plot(*zip(*gd_path_g), marker='o', label='Gradient Descent')
ax.plot(*zip(*mom_path_g), marker='o', label='Momentum')
ax.plot(*zip(*adam_path_g), marker='o', label='Adam')

ax.set_title(f'Last norms: GD={round(gd_norm_g, 3)}, MOM={round(mom_norm_g, 3)}, ADAM={round(adam_norm_g, 3)}')
ax.legend()

plt.show()

# %% [markdown]
# **How fast does each method converge?** <br>
# 
# The Adam gradient convergences the fastest, followed by momentum, then gradient descent. The gradient descent needs more steps and moves slower while the momentum uses past updates which speeds it up. The Adam gradient is the fastest because it automatically adjusts its step size. <br>
# <br>
# 
# **Which method finds the minimum more accurately?** <br>
# 
# The Adam gradient finds the mimimum more accurately because it's able to adapt the learning rate and follow the shape of the function better. <br>
# <br>
# 
# **How does each algorithm behave around flat regions or saddle points?** <br>
# 
# In flat regions/saddle points, the gradient descent slows down and takes many steps to move. The momentum keeps its direction and moves faster through these areas. The Adam is the best with flat regions because it adjusts its step size in order to continue to move effeciently, no matter the size of the gradient.


