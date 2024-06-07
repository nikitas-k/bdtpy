import numpy as np
import matplotlib.pyplot as plt

from .base import BaseModel
from .solvers import sde_solver

class Oscillator1D(BaseModel):
    """

    1D Nonlinear oscillator of the form

        dy = 1 + epsilon*cos(y) + sigma*dW

    where
        y(t) is the dynamic variable
        epsilon is the damping term
        sigma > 0 is the volatility of the noise
    
    """
    def __init__(self, epsilon, sigma, T=10.0, dt=0.01):
        super().__init__(epsilon=epsilon, sigma=sigma, T=T, dt=dt)
        self.time_points = np.arange(0, self.T + self.dt, self.dt)

    def _drift(self, y, t):
        return 1 + self.epsilon*np.cos(y)

    def _diffusion(self, y, t):
        return self.sigma
    
    def solve(self):
        self.y0 = 1.0
        sol = sde_solver(self._drift, self._diffusion, self.y0, tspan=self.time_points)
        self.solution = sol

        return self.time_points, sol
    
    def plot_timeseries(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 5))
        plt.plot(self.time_points, self.solution, lw=1)
        
        plt.title('1D Nonlinear Oscillator')
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

class Oscillator2D(BaseModel):
    """
    
    Generate solutions to two coupled nonlinear oscillators 
    with independent noise terms of the form

        dy_1 = lambda*y_1 - omega*y_2 + sigma_1*dW_1
        dy_2 = lambda*y_2 + omega*y_1 + sigma_2*dW_2

    where
        lambda is the bias
        omega is the coupling constant
        sigma > 0 is the volatility of the noise
    
    """
    def __init__(self, lam, omega, sigma1, sigma2, T=5.0, dt=0.01):
        super().__init__(lam=lam, omega=omega, sigma1=sigma1, sigma2=sigma2, T=T, dt=dt)
        self.time_points = np.arange(0, self.T + self.dt, self.dt)
        self.name = '2D Nonlinear Oscillator'

    def _drift(self, y, t):
        ret = np.zeros(2)
        ret[0] = self.lam*y[0] - self.omega*y[1]
        ret[1] = self.lam*y[1] + self.omega*y[0]
        return ret
    
    def _diffusion(self, y, t):
        return np.diag([self.sigma1, self.sigma2])
    
    def solve(self):
        self.Y0 = np.array([1.0, 1.0])
        sol = sde_solver(self._drift, self._diffusion, self.Y0, self.time_points)
        self.solution = sol

        return self.time_points, sol
    
    def plot_timeseries(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 5))
        plt.plot(self.time_points, self.solution[:, 0], lw=0.5, alpha=1.0, label='1')
        plt.plot(self.time_points, self.solution[:, 1], lw=0.5, alpha=1.0, label='2')
        
        plt.title(self.name)
        plt.legend(loc=0)
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def plot_phase_portrait(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 5))
        plt.scatter(self.solution[:, 0], self.solution[:, 1], lw=1.0, alpha=1.0)
        
        plt.title('Phase portrait')
        plt.xlabel('Y1')
        plt.ylabel('Y2')
        plt.grid(True)
        plt.show()

class OrnsteinUhlenbeck(BaseModel):
    """ 
    Generate solutions to a system of n independent 
    Ornstein-Uhlenbeck processes

    dY_i = theta(mu - Y_i)dt + sigma*dW_i

    where
    Y_i(t) are the n state variables
    mu dictates the long-term mean of Y_i(t)
    theta > 0 is the rate of convergence to the mean
    sigma > 0 is the volatility of the noise

    Notes
    =====
    This system has n = 10
    """
    def __init__(self, theta, mu, sigma, n=10, T=2000, dt=1, solver='itoSRI2'):
        super().__init__(theta=theta, mu=mu, sigma=sigma, n=n, T=T, dt=dt, solver=solver)
        self.time_points = np.arange(0, self.T + self.dt, self.dt)
        self.name = 'OrnsteinUhlenbeck'
        
    def _drift(self, y, t):
        """ Drift function for the OU process """
        return self.theta * (self.mu - y)
    
    def _diffusion(self, y, t):
        """ Diffusion function for the OU process """
        return self.sigma * np.ones_like(y)[..., np.newaxis]
    
    def solve(self):
        y0 = np.zeros(self.n)
        sol = sde_solver(self._drift, self._diffusion, y0, tspan=self.time_points, solver=self.solver)
        self.solution = sol

        return self.time_points, sol
    
    def plot_timeseries(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 5))
        for i in range(self.n):
            plt.plot(self.time_points, self.solution[i], lw=0.5, alpha=0.05, color='k')
            plt.plot(self.time_points, self.solution[0], lw=0.5, color='k')
        
        plt.title(self.name)
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def plot_phase_portrait(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 6))
        plt.scatter(self.solution[0], self.solution[1], lw=1)
        plt.title('Phase Portrait')
        plt.xlabel('Y1')
        plt.ylabel('Y2')
        plt.grid(True)
        plt.show()

class GeometricBrownianMotion(BaseModel):
    """
    The Ito Stochastic Differential Equation is

        dY = mu*Y*dt + sigma*Y*dW_t

    where
        Y(t) is the dynamic variable
        mu is the drift parameter
        sigma is the diffusion parameter
    
    The dynamic variable Y(t) is normally distributed with mean Y_0*exp(mu*t)
    and variance Y^2_0*exp(2*mu*t)(exp(sigma^2*t) - 1)

    This simulation has n = 100 independent processes.

    """
    def __init__(self, mu, sigma, Y0, T=5.0, dt=0.001, n=100):
        super().__init__(mu=mu, sigma=sigma, Y0=Y0, T=T, dt=dt, n=n)
        self.time_points = np.arange(0, self.T + self.dt, self.dt)
    
    def solve(self):
        mu = self.mu
        sigma = self.sigma
        Y0 = self.Y0
        dt = self.dt
        n = self.n

        Y = np.zeros((len(self.time_points), n))
        Y[0, :] = Y0
        
        for i in range(1, len(self.time_points)):
            dW = np.random.normal(scale=np.sqrt(dt), size=n)
            Y_prev = Y[i-1, :]
            drift = mu * Y_prev * dt
            diffusion = sigma * Y_prev * dW
            Y[i, :] = Y_prev + drift + diffusion
        
        self.solution = Y
        return self.time_points, Y
    
    def plot_timeseries(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        for i in range(self.solution.shape[1]):
            plt.plot(self.time_points, self.solution[:, i], lw=1, alpha=0.05, color='k')
            plt.plot(self.time_points, self.solution[:, 0], lw=1, color='k')
        
        plt.title('Geometric Brownian Motion')
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.ylim([0, 20])
        plt.grid(True)
        plt.show()

    def plot_phase_portrait(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(8, 8))
        plt.plot(self.solution[:, 0], self.solution[:, 1], lw=1)
        plt.title('Phase Portrait')
        plt.xlabel('Y1')
        plt.ylabel('Y2')
        plt.grid(True)
        plt.show()
    
    
class KloedenPlaten(BaseModel):
    """
    Ito Stochastic Differential Equation (4.46) from Kloeden and Platen (1999)

        dy = -(a + yb^2)(1 - y^2)dt + b(1 - y^2)dW_t

    where
        y(t) is the dynamic variable
        a and b are scalar constants

    It has the explicit solution
        
        y = A/B
    
    where
        A = (1 + y_0) exp(-2at + 2bW_t) + y_0 - 1
        B = (1 + y_0) exp(-2at + 2bW_t) - y_0 + 1
    
    """
    def __init__(self, a, b, y0, T=5.0, dt=0.001):
        super().__init__(a=a, b=b, y0=y0, T=T, dt=dt)
        self.time_points = np.arange(0, self.T + self.dt, self.dt)
    
    def _drift(self, y):
        a = self.a
        b = self.b
        return -(a + y * b**2) * (1 - y**2)

    def _diffusion(self, y):
        b = self.b
        return b * (1 - y**2)
    
    def solve(self):
        y = np.zeros(len(self.time_points))
        y[0] = self.y0
        dt = self.dt
        
        for i in range(1, len(self.time_points)):
            t = self.time_points[i-1]
            y_prev = y[i-1]
            dW = np.random.normal(scale=np.sqrt(dt))
            
            drift = self._drift(y_prev) * dt
            diffusion = self._diffusion(y_prev) * dW
            
            y[i] = y_prev + drift + diffusion
        
        self.solution = y
        return self.time_points, y

    def plot_timeseries(self):
        if not hasattr(self, 'solution'):
            self.solve()
        
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(self.time_points, self.solution, lw=1)
        
        plt.title('Kloeden-Platen 1999 SDE')
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.ylim([-1, 1])
        plt.grid(True)
        plt.show()
    
