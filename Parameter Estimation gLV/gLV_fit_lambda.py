from itertools import repeat
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, minimize_scalar

from tqdm import tqdm
import time

### Basic usage ###

'''
# instantiate ODE fit
model = ODE(df, params, species)

# fit to data
params = model.fit()

where df has columns [Time, Treatments, S1, ..., SN]
'''

# augmented LV model
def system(x, r, Ax):
    # dx / dt = x (r + A @ x)
    return x*(r + Ax)

# derivative of sensitivity equations
def dZdt(Z, x, r, A, Ax):

    # compute Jacobian (gradient of model w.r.t. x)
    Jx = Jacobian(x, r, A, Ax)

    # compute gradient of model w.r.t. parameters
    Jp = Gradient(x, len(x))

    return Jx@Z + Jp

# compute gradient of system w.r.t. x
def Jacobian(x, r, A, Ax):
    J = np.einsum('i,ij->ij', x, A) + np.diag(r + Ax)
    return J

# compute gradients of system w.r.t. parameters
def Gradient(x, dimx):

    # gradient w.r.t. r, [grad_r(system)]ij = d x_i / d rj
    gradr = np.diag(x)

    # gradient w.r.t. A, [grad_A(system)]ijk = d x_i / d Ajk
    gradA = np.reshape(np.einsum('i,ij,k->ijk', x, np.eye(len(x)), x), [dimx, dimx**2])

    return np.concatenate((gradr, gradA), axis=-1)

# define function that returns model output and sensitivity vector
def runODEZ(t_eval, x, params):

    # reshape parameters to matrices
    dimx  = len(x)
    r  = params[:dimx]
    A  = np.reshape(params[dimx:], [dimx, dimx])

    # determine dimension of sensitivity vector
    n_params = len(params)
    dimz = dimx*n_params

    # xz will be 1-D
    def dXZ_dt(t, xz):
        # split up x and z
        x = xz[:dimx]

        # concatenate x and u
        Ax = A@x

        # compute derivative of sensitivity
        Z    = np.reshape(xz[dimx:], [dimx, n_params])
        dzdt = np.reshape(dZdt(Z, x, r, A, Ax), dimz)

        # concatentate x and z
        dXZdt = np.concatenate([system(x, r, Ax), dzdt])
        return dXZdt

    # set initial condition of z equal to zeros
    xz = np.concatenate((x, np.zeros(dimz)))

    # solve ODE model
    soln = solve_ivp(dXZ_dt, (0, t_eval[-1]), xz,
                     t_eval=t_eval, method='LSODA')
    t_solver, y = soln.t, soln.y.T
    return y

# define function that returns model output and sensitivity vector
def runODE(t_eval, x, params):
    # reshape parameters to matrices
    dimx  = len(x)
    r  = params[:dimx]
    A  = np.reshape(params[dimx:], [dimx, dimx])

    # x will be 1-D
    def dX_dt(t, x):
        # compute matrix vector products
        Ax = A@x
        return system(x, r, Ax)

    # solve ODE model
    soln = solve_ivp(dX_dt, (0, t_eval[-1]), x,
                     t_eval=t_eval, method='LSODA')
    t_solver, y = soln.t, soln.y.T
    return y

# compute gradients of cost function
def compute_gradient(t_eval, Y_species, model):

    # run model using current parameters
    output = runODEZ(t_eval, Y_species[0, :], model.params)
    if output.shape[0] < Y_species.shape[0]:
        print("Warning: could not compute full trajectory!")

    # collect gradients and reshape
    G = np.reshape(output[:, model.n_species:],
                  [output.shape[0], model.n_species, model.n_params])

    # compute Hessian
    H = np.einsum('tki,kl,tlj->ij', G, model.Beta, G)

    # determine SSE, gradient of SSE
    Y_predicted = output[:, :model.n_species] # dim = T x n_species
    Y_error = Y_predicted - Y_species[:output.shape[0],:]
    SSE = np.sum(Y_error**2)
    NLP = np.einsum('tk,kl,tl->', Y_error, model.Beta, Y_error)/2.

    # sum over time and outputs to get gradient w.r.t params
    grad_NLP = np.einsum('tk,kl,tli->i', Y_error, model.Beta, G)

    return H, SSE, NLP, grad_NLP

class ODE:
    def __init__(self, df, params, species, verbose=True, n_jobs=1):
        # store experimental data
        # columns: [Treatment name], [Time], [x_1], ..., [x_n]
        self.df = df
        # make sure params are 1-dimensional
        self.params = np.array(params).ravel()
        self.prior  = np.array(params).ravel()
        self.hyper_prior = np.array(params).ravel()
        # number of parameters
        self.n_params = len(params)
        # dimension of model output
        self.n_species = len(species)

        # runtime parameters
        self.verbose = verbose
        self.n_jobs = n_jobs

        # set parameters of precision hyper-priors
        self.a = 1e-3
        self.b = 1e-3
        self.gamma = 1e-3

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        # store treatment names
        self.all_treatments = df.Treatments.values
        self.unique_treatments = np.unique(self.all_treatments)
        self.species_names = species

        # store measured datasets for quick access
        self.data = []
        for treatment in self.unique_treatments:

            # pull community trajectory
            comm_inds = np.in1d(self.df['Treatments'].values, treatment)
            comm_data = self.df.iloc[comm_inds].copy()

            # make sure comm_data is sorted in chronological order
            comm_data.sort_values(by='Time', ascending=True, inplace=True)

            # pull evaluation times
            t_eval = np.array(comm_data['Time'].values, np.float16)

            # pull species data
            Y_species = np.array(comm_data[self.species_names].values, np.float16)

            # append t_eval and Y_species to data list
            self.data.append([t_eval, Y_species])

    # EM updates to alpha and Beta parameters
    def update_precision(self):
        print("Updating precision...")
        SSE = 0.
        self.N   = 0
        for t_eval, Y_species in tqdm(self.data):
            # N is the number of measured datapoints
            self.N   += np.sum(Y_species > 0.)

            # run model using current parameters
            output = runODEZ(t_eval, Y_species[0, :], self.params)
            if output.shape[0] < Y_species.shape[0]:
                print("Warning: could not compute full trajectory!")

            # collect gradients and reshape
            G = np.reshape(output[:, self.n_species:],
                          [output.shape[0], self.n_species, self.n_params])

            # determine SSE
            Y_predicted = output[:, :self.n_species] # dim = T x n_species
            Y_error = Y_predicted - Y_species[:output.shape[0],:]
            if self.Ainv is None:
                SSE += np.sum(Y_error**2)
            else:
                SSE += np.sum(Y_error**2) + np.einsum('tki,ij,tkj',G,self.Ainv,G)

        ### M step: update hyper-parameters ###
        if self.A is None:
            # initial guess
            # parameter precision
            self.Alpha = np.eye(self.n_params)
            # target precision
            self.beta  = self.N/(SSE + 2.*self.b)
            self.Beta = self.beta*np.eye(self.n_species)
        else:
            # maximize complete data log-likelihood w.r.t. alpha and beta
            # parameter precision
            self.alpha = self.n_params/(np.dot(self.params-self.prior, self.params-self.prior) + np.trace(self.Ainv) + 2.*self.a)
            self.Alpha = self.alpha*np.eye(self.n_params)
            # measurement precision
            self.beta  = self.N /(SSE + 2.*self.b)
            self.Beta = self.beta*np.eye(self.n_species)
            # regularization coefficient
            self.lmbda = self.alpha/self.beta
            print("Total samples: {}, Updated regularization: {:.2e}".format(self.N, self.lmbda))

    def update_evidence(self):
        # compute evidence
        self.evidence= self.N*np.sum(np.log(np.linalg.eigvalsh(self.Beta)))/2. + \
                  np.sum(np.log(np.diag(self.Alpha)))/2. - \
                  np.sum(np.log(np.linalg.eigvalsh(self.A)))/2. - self.NLP
        print("Evidence {:.3f}".format(self.evidence))

    def update_covariance(self):
        # update parameter covariance matrix given current parameter estimate
        self.A = np.copy(self.Alpha)

        # loop over each sample in dataset
        print("Updating parameter covariance...")
        for t_eval, Y_species in tqdm(self.data):

            # run model using current parameters
            output = runODEZ(t_eval, Y_species[0, :], self.params)
            if output.shape[0] < Y_species.shape[0]:
                print("Warning: could not compute full trajectory!")

            # collect gradients and reshape
            G = np.reshape(output[:, self.n_species:],
                          [output.shape[0], self.n_species, self.n_params])

            # compute Hessian
            self.A += np.einsum('tki,kl,tlj->ij', G, self.Beta, G)

        # make sure Hessian is symmetric
        self.A = (self.A + self.A.T)/2.

        # Compute posterior covariance
        self.Ainv = np.linalg.inv(self.A)

        # make sure Ainv is symmetric
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

    def objective(self, params):
        # quick function to compute negative log of posterior
        NLP = np.einsum('i,ij,j->',params-self.prior,self.Alpha,params-self.prior)
        for t_eval, Y_species in self.data:
            # run model using current parameters
            output = runODE(t_eval, Y_species[0, :], params)
            if output.shape[0] < Y_species.shape[0]:
                print("Warning: could not compute full trajectory!")
                return np.inf
            # determine SSE and gradient of SSE
            Y_error = output - Y_species[:output.shape[0],:]
            NLP += np.einsum('tk,kl,tl->',Y_error,self.Beta,Y_error)
        return NLP/2.

    def objective_grad(self, params):
        # compute cost, gradients, Hessian, and precision
        # Add parameter penalty to SSE and grad_NLP
        self.A = np.copy(self.Alpha)
        self.NLP = (params-self.prior)@self.Alpha@(params-self.prior)/2.
        self.grad_NLP = self.Alpha@(params-self.prior)

        # compute gradients in parallel
        res = Parallel(n_jobs=self.n_jobs)(delayed(compute_gradient)(t_eval, Y_species, self) for t_eval, Y_species in self.data)
        for H, SSE, NLP, grad_NLP in res:
            self.A += H
            self.NLP += NLP
            self.grad_NLP += grad_NLP

        # make sure Hessian is symmetric
        self.A = (self.A + self.A.T)/2.
        # Compute posterior covariance
        self.Ainv = np.linalg.inv(self.A)
        # make sure Ainv is symmetric
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

    def gradient_descent(self, eta, tol, maxpatience, maxiter, maxstep):
        # use gradient descent to determine parameters
        fit_NLP = np.inf
        epochs = 0
        patience = 0

        while True:
            # call objective function
            self.objective_grad(self.params)

            # determine descent direction
            gradient  = self.Ainv@self.grad_NLP

            # evaluate convergence criterion
            if ~np.isinf(fit_NLP):
                convergence = (fit_NLP - self.NLP) / np.max([1,np.abs(fit_NLP)])
            else:
                convergence = np.inf
            if self.verbose:
                print("Iteration: {}, Loss: {:.3e}, Convergence rate: {:.3e}".format(epochs, self.NLP, convergence))

            # if convergence less than tolerance, end
            if abs(convergence) < tol:
                eta *= 2.
                print("Step size set to {:.3e}".format(eta))
                patience += 1

            if patience > maxpatience or epochs > maxiter:
                return

            # if the SSE was worse, reset and lower learning rate
            if convergence<0 or np.isnan(convergence):
                eta /= 10.
                print("Step size set to {:.3e}".format(eta))
                self.params = best_params - eta*best_grad
                fit_NLP = np.inf
            else:
                best_params = np.copy(self.params)
                best_grad   = np.copy(gradient)
                '''if epochs > 1 and epochs%10==0:
                    eta = np.clip(eta*2.0, 0.0, maxstep)
                    print("Optimizing step size...")
                    bracket  = [eta/10., np.min([eta*2., maxstep])]
                    step_obj = lambda step: self.objective(self.params - step*gradient)
                    step_soln = minimize_scalar(step_obj, bracket, tol=tol)
                    eta = np.clip(step_soln.x, 0.0, maxstep)
                    print("Step size set to {:.3e}".format(eta))'''

                # update parameters
                self.params = self.params - eta*gradient

                # update previous error
                fit_NLP = np.copy(self.NLP)

            # update epoch and repeat
            epochs += 1

    def fit(self, eta=1e-2, evidence_tol=1e-2, beta_tol=1e-2, maxpatience=1, maxiter=100, maxstep=.1):
        # estimate parameters using gradient descent
        convergence = np.inf
        prev_evidence    = 0
        print("\nRunning gradient descent...")
        while convergence > evidence_tol:
            # update alpha and Beta hyper-parameters
            self.update_precision()
            # fit using updated alpha and Beta
            self.gradient_descent(eta, beta_tol, maxpatience, maxiter, maxstep)
            # check convergence on evidence
            self.update_evidence()
            convergence = np.abs(prev_evidence - self.evidence) / np.max([1,np.abs(prev_evidence)])
            prev_evidence = np.copy(self.evidence)

    def predict(self, x_test, teval):
        # check if precision has been computed
        if self.Ainv is None:
            self.update_covariance()

        # make predictions given initial conditions and evaluation times
        output = runODEZ(teval, x_test, self.params)

        # reshape gradient
        G = np.reshape(output[:, self.n_species:],
                       [output.shape[0], self.n_species, self.n_params])

        # compute Hessian over trajectory
        H = np.einsum('tki,kl,tlj->ij', G, self.Beta, G)

        # calculate variance of each output (dimension = [steps, outputs])
        variance = np.diag(np.linalg.inv(self.Beta)) + np.einsum('tki,ij,tkj->tk', G, self.Ainv, G)

        return output[:, :self.n_species], variance, H
