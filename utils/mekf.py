

import torch
from torch.optim import Optimizer


class MEKF_MA(Optimizer):
	"""
	Modified Extended Kalman Filter with generalized exponential Moving Average
	"""

	def __init__(self, params, dim_out, p0=1e-2, lbd=1, sigma_r=None, sigma_q=0, lr=1,
	             miu_v=0, miu_p=0, k_p=1,
	             R_decay=False,R_decay_step=1000000):

		if sigma_r is None:
			sigma_r = max(lbd,0)
		self._check_format(dim_out, p0, lbd, sigma_r, sigma_q, lr,miu_v,miu_p,k_p,R_decay,R_decay_step)
		defaults = dict(p0=p0, lbd=lbd, sigma_r=sigma_r, sigma_q=sigma_q,
		                lr=lr,miu_v=miu_v,miu_p=miu_p,k_p=k_p,
		                R_decay=R_decay,R_decay_step=R_decay_step)
		super(MEKF_MA, self).__init__(params, defaults)

		self.state['dim_out'] = dim_out
		with torch.no_grad():
			self._init_mekf_matrix()

	def _check_format(self, dim_out, p0, lbd, sigma_r, sigma_q, lr, miu_v, miu_p, k_p, R_decay, R_decay_step):
		if not isinstance(dim_out, int) and dim_out > 0:
			raise ValueError("Invalid output dimension: {}".format(dim_out))
		if not 0.0 < p0:
			raise ValueError("Invalid initial P value: {}".format(p0))
		if not 0.0 < lbd:
			raise ValueError("Invalid forgetting factor: {}".format(lbd))
		if not 0.0 < sigma_r:
			raise ValueError("Invalid covariance matrix value for R: {}".format(sigma_r))
		if not 0.0 <= sigma_q:
			raise ValueError("Invalid covariance matrix value for Q: {}".format(sigma_q))
		if not 0.0 < lr:
			raise ValueError("Invalid learning rate: {}".format(lr))

		if not 0.0 <= miu_v < 1.0:
			raise ValueError("Invalid EMA decaying factor for V matrix: {}".format(miu_v))
		if not 0.0 <= miu_p < 1.0:
			raise ValueError("Invalid EMA decaying factor for P matrix: {}".format(miu_p))
		if not isinstance(k_p, int) and k_p >= 0:
			raise ValueError("Invalid delayed step size of Lookahead P: {}".format(k_p))

		if not isinstance(R_decay, int) and not isinstance(R_decay, bool):
			raise ValueError("Invalid R decay flag: {}".format(R_decay))
		if not isinstance(R_decay_step, int):
			raise ValueError("Invalid max step for R decaying: {}".format(R_decay_step))

	def _init_mekf_matrix(self):
		self.state['step']=0
		self.state['mekf_groups']=[]
		dim_out = self.state['dim_out']
		for group in self.param_groups:
			mekf_mat=[]
			for p in group['params']:
				matrix = {}
				size = p.size()
				dim_w=1
				for dim in size:
					dim_w*=dim
				device= p.device
				matrix['P'] = group['p0']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['R'] = group['sigma_r']*torch.eye(dim_out,dtype=torch.float,device=device)
				matrix['Q'] = group['sigma_q'] * torch.eye(dim_w, dtype=torch.float, device=device)
				matrix['H'] = None
				mekf_mat.append(matrix)
			self.state['mekf_groups'].append(mekf_mat)

	def step(self,closure=None, H_groups=None, err=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
			H_groups: groups of gradient matrix
			err: error value
		example 1 (optimization step with closure):
			# optimizer -> MEKF_MA optimizer
			# y -> observed value, y_hat -> predicted value
			y = y.contiguous().view((-1, 1))   #  shape of (dim_out,1)
			y_hat = y_hat.contiguous().view((-1, 1)) #  shape of (dim_out,1)
			err = (y - y_hat).detach()

			def mekf_closure(index=0):
				optimizer.zero_grad()
				dim_out = optimizer.state['dim_out']
				retain = index < dim_out - 1
				y_hat[index].backward(retain_graph=retain)
				return err

			optimizer.step(mekf_closure)

		example 2 (optimization step with H_groups):
			# y -> observed value, y_hat -> predicted value
			# H -> gradient matrix that need to be specified
			y = y.contiguous().view((-1, 1))   #  shape of (dim_out,1)
			y_hat = y_hat.contiguous().view((-1, 1)) #  shape of (dim_out,1)
			err = (y - y_hat).detach()
			optimizer.step(H_groups=H_groups,err=err)
		"""
		self.state['step'] += 1

		if closure is not None:
			for y_ind in range(self.state['dim_out']):
				err = closure(y_ind)
				for group_ind in range(len(self.param_groups)):
					group = self.param_groups[group_ind]
					mekf_mat = self.state['mekf_groups'][group_ind]
					for ii, w in enumerate(group['params']):
						if w.grad is None:
							continue
						H_n = mekf_mat[ii]['H']
						grad = w.grad.data.detach()
						if len(w.size())>1:
							grad = grad.transpose(1, 0)
						grad = grad.contiguous().view((1,-1))
						if y_ind ==0:
							H_n=grad
						else:
							H_n = torch.cat([H_n,grad],dim=0)
						self.state['mekf_groups'][group_ind][ii]['H'] = H_n
		else:
			for group_ind in range(len(self.param_groups)):
				H_mats = H_groups[group_ind]
				for ii, H_n in enumerate(H_mats):
					self.state['mekf_groups'][group_ind][ii]['H'] = H_n

		err_T = err.transpose(0,1)

		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			mekf_mat = self.state['mekf_groups'][group_ind]

			miu_v = group['miu_v']
			miu_p = group['miu_p']
			k_p = group['k_p']
			lr = group['lr']
			lbd = group['lbd']

			for ii,w in enumerate(group['params']):
				if w.grad is None:
					continue

				P_n_1 = mekf_mat[ii]['P']
				R_n = mekf_mat[ii]['R']
				Q_n = mekf_mat[ii]['Q']
				H_n = mekf_mat[ii]['H']
				H_n_T = H_n.transpose(0, 1)

				if group['R_decay']:
					miu = 1.0 / min(self.state['step'],group['R_decay_step'])
					R_n = R_n + miu * (err.mm(err_T) + H_n.mm(P_n_1).mm(H_n_T) - R_n)
					self.state['mekf_groups'][group_ind][ii]['R']= R_n

				g_n = H_n.mm(P_n_1).mm(H_n_T) + R_n
				g_n = g_n.inverse()
				K_n = P_n_1.mm(H_n_T).mm(g_n)
				V_n = lr * K_n.mm(err)
				if len(w.size()) > 1:
					V_n = V_n.view((w.size(1),w.size(0))).transpose(1,0)
				else:
					V_n = V_n.view(w.size())
				if miu_v>0:
					param_state = self.state[w]
					if 'buffer_V' not in param_state:
						V_ema = param_state['buffer_V'] = torch.clone(V_n).detach()
					else:
						V_ema = param_state['buffer_V']
						V_ema.mul_(miu_v).add_(V_n.mul(1-miu_v).detach())
					V_n=V_ema
				w.data.add_(V_n)

				P_n = (1/lbd) * (P_n_1 - K_n.mm(H_n).mm(P_n_1) + Q_n)
				if miu_p>0 and k_p>0:
					if self.state['step'] % k_p==0:
						param_state = self.state[w]
						if 'buffer_P' not in param_state:
							P_ema = param_state['buffer_P'] = torch.clone(P_n).detach()
						else:
							P_ema = param_state['buffer_P']
							P_ema.mul_(miu_p).add_(P_n.mul(1 - miu_p).detach())
						P_n = P_ema
				self.state['mekf_groups'][group_ind][ii]['P'] =P_n

		return err

class MEKF(Optimizer):
	"""
	Modified Extended Kalman Filter
	"""

	def __init__(self, params, dim_out, p0=1e-2, lbd=1, sigma_r=None, sigma_q=0,
	             R_decay=False,R_decay_step=1000000):

		if sigma_r is None:
			sigma_r = max(lbd,0)
		self._check_format(dim_out, p0, lbd, sigma_r, sigma_q, R_decay,R_decay_step)
		defaults = dict(p0=p0, lbd=lbd, sigma_r=sigma_r, sigma_q=sigma_q,
		                R_decay=R_decay,R_decay_step=R_decay_step)
		super(MEKF, self).__init__(params, defaults)

		self.state['dim_out'] = dim_out
		with torch.no_grad():
			self._init_mekf_matrix()

	def _check_format(self, dim_out, p0, lbd, sigma_r, sigma_q, R_decay, R_decay_step):
		if not isinstance(dim_out, int) and dim_out > 0:
			raise ValueError("Invalid output dimension: {}".format(dim_out))
		if not 0.0 < p0:
			raise ValueError("Invalid initial P value: {}".format(p0))
		if not 0.0 < lbd:
			raise ValueError("Invalid forgetting factor: {}".format(lbd))
		if not 0.0 < sigma_r:
			raise ValueError("Invalid covariance matrix value for R: {}".format(sigma_r))
		if not 0.0 <= sigma_q:
			raise ValueError("Invalid covariance matrix value for Q: {}".format(sigma_q))

		if not isinstance(R_decay, int) and not isinstance(R_decay, bool):
			raise ValueError("Invalid R decay flag: {}".format(R_decay))
		if not isinstance(R_decay_step, int):
			raise ValueError("Invalid max step for R decaying: {}".format(R_decay_step))

	def _init_mekf_matrix(self):
		self.state['step']=0
		self.state['mekf_groups']=[]
		dim_out = self.state['dim_out']
		for group in self.param_groups:
			mekf_mat=[]
			for p in group['params']:
				matrix = {}
				size = p.size()
				dim_w=1
				for dim in size:
					dim_w*=dim
				device= p.device
				matrix['P'] = group['p0']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['R'] = group['sigma_r']*torch.eye(dim_out,dtype=torch.float,device=device)
				matrix['Q'] = group['sigma_q'] * torch.eye(dim_w, dtype=torch.float, device=device)
				matrix['H'] = None
				mekf_mat.append(matrix)
			self.state['mekf_groups'].append(mekf_mat)

	def step(self,closure=None, H_groups=None, err=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
			H_groups: groups of gradient matrix
			err: error value
		example 1 (optimization step with closure):
			# optimizer -> MEKF_MA optimizer
			# y -> observed value, y_hat -> predicted value
			y = y.contiguous().view((-1, 1))   #  shape of (dim_out,1)
			y_hat = y_hat.contiguous().view((-1, 1)) #  shape of (dim_out,1)
			err = (y - y_hat).detach()

			def mekf_closure(index=0):
				optimizer.zero_grad()
				dim_out = optimizer.state['dim_out']
				retain = index < dim_out - 1
				y_hat[index].backward(retain_graph=retain)
				return err

			optimizer.step(mekf_closure)

		example 2 (optimization step with H_groups):
			# y -> observed value, y_hat -> predicted value
			# H -> gradient matrix that need to be specified
			y = y.contiguous().view((-1, 1))   #  shape of (dim_out,1)
			y_hat = y_hat.contiguous().view((-1, 1)) #  shape of (dim_out,1)
			err = (y - y_hat).detach()
			optimizer.step(H_groups=H_groups,err=err)
		"""
		self.state['step'] += 1

		if closure is not None:
			for y_ind in range(self.state['dim_out']):
				err = closure(y_ind)
				for group_ind in range(len(self.param_groups)):
					group = self.param_groups[group_ind]
					mekf_mat = self.state['mekf_groups'][group_ind]
					for ii, w in enumerate(group['params']):
						if w.grad is None:
							continue
						H_n = mekf_mat[ii]['H']
						grad = w.grad.data.detach()
						if len(w.size())>1:
							grad = grad.transpose(1, 0)
						grad = grad.contiguous().view((1,-1))
						if y_ind ==0:
							H_n=grad
						else:
							H_n = torch.cat([H_n,grad],dim=0)
						self.state['mekf_groups'][group_ind][ii]['H'] = H_n
		else:
			for group_ind in range(len(self.param_groups)):
				H_mats = H_groups[group_ind]
				for ii, H_n in enumerate(H_mats):
					self.state['mekf_groups'][group_ind][ii]['H'] = H_n

		err_T = err.transpose(0,1)

		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			mekf_mat = self.state['mekf_groups'][group_ind]
			lbd = group['lbd']

			for ii,w in enumerate(group['params']):
				if w.grad is None:
					continue

				P_n_1 = mekf_mat[ii]['P']
				R_n = mekf_mat[ii]['R']
				Q_n = mekf_mat[ii]['Q']
				H_n = mekf_mat[ii]['H']
				H_n_T = H_n.transpose(0, 1)

				if group['R_decay']:
					miu = 1.0 / min(self.state['step'],group['R_decay_step'])
					R_n = R_n + miu * (err.mm(err_T) + H_n.mm(P_n_1).mm(H_n_T) - R_n)
					self.state['mekf_groups'][group_ind][ii]['R']= R_n

				g_n = H_n.mm(P_n_1).mm(H_n_T) + R_n
				g_n = g_n.inverse()
				K_n = P_n_1.mm(H_n_T).mm(g_n)
				V_n = K_n.mm(err)
				if len(w.size()) > 1:
					V_n = V_n.view((w.size(1),w.size(0))).transpose(1,0)
				else:
					V_n = V_n.view(w.size())
				w.data.add_(V_n)

				P_n = (1/lbd) * (P_n_1 - K_n.mm(H_n).mm(P_n_1) + Q_n)
				self.state['mekf_groups'][group_ind][ii]['P'] =P_n

		return err