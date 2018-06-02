import numpy as np


class RNN:
	def __init__(self, input_size, output_size, hidden_size):

		# input
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size

		# weights
		self.U = self.he_init(hidden_size, input_size)
		self.W = self.he_init(hidden_size, hidden_size)
		self.b = self.he_init(hidden_size)
		self.V = self.he_init(output_size, hidden_size)
		self.c = self.he_init(output_size)

		# states
		self.prev_state = None
		self.h = np.zeros((200,hidden_size))
		self.a = None
		self.o = None
		self.p = None

		# grads
		self.grad_U = np.empty_like(self.U)
		self.grad_W = np.empty_like(self.W)
		self.grad_b = np.empty_like(self.b)
		self.grad_V = np.empty_like(self.V)
		self.grad_c = np.empty_like(self.c)

	def he_init(self, row, columns=None):
		try:
			return np.random.normal(loc=0.0, scale=2 / np.sqrt(columns), size=(row, columns))
		except:
			return np.zeros(row)

	def evaluate(self, x, prev_state):
		a = self.U @ x + self.W @ prev_state + self.b
		h = np.tanh(a)
		o = self.V @ h + self.c
		softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
		p = softmax(o)
		return {
			'a': a,
			'h': h,
			'o': o,
			'p': p,
		}

	def forward(self, input_sequence):

		h_temp = []
		a_temp = []
		o_temp = []
		p_temp = []
		last_state = self.h[-1]
		self.prev_state = last_state

		for char in input_sequence:
			evars = self.evaluate(char, last_state)
			last_state = evars['h']
			h_temp.append(evars['h'])
			a_temp.append(evars['a'])
			o_temp.append(evars['o'])
			p_temp.append(evars['p'])

		self.h = np.asarray(h_temp)
		self.a = np.asarray(a_temp)
		self.o = np.asarray(o_temp)
		self.p = np.asarray(p_temp)


	def cross_entropy(self, Y):
		#y_p = self.p[Y, np.arange(Y.size)]
		y_p = np.multiply(self.p, Y)
		# or alternatively: y_p = diag(dot(p,Y_t))
		# select ascissa given by Y and scroll all p
		return -np.log(np.sum(y_p))

	def backprop(self, input_sequence, output_sequence):

		# call forward
		self.forward(input_sequence)

		# per il plot
		Loss = self.cross_entropy(output_sequence)

		# dL/dt
		assert np.shape(self.p) == np.shape(output_sequence)

		# init temp grads
		g_h = np.zeros((output_sequence.shape[0], self.hidden_size))
		g_a = np.zeros(np.shape(self.h))

		# dL/do
		g_o = -(output_sequence - self.p)

		# dL/dh_tau
		g_h[-1] = g_o[-1] @ self.V

		# dL/da_tau
		diag2fill = np.zeros((g_h[-1].shape[0], g_h[-1].shape[0]))
		np.fill_diagonal(diag2fill, self.h[-1])
		g_a[-1] = g_h[-1] @ (1 - diag2fill ** 2)

		for i in range(output_sequence.shape[0]-2, -1, -1):
			# dL/dh_i
			g_h[i] = g_o[i] @ self.V + g_a[i + 1] @ self.W

			# dL/da_i
			np.fill_diagonal(diag2fill, self.h[i])
			g_a[i] = g_h[i] @ (1 - diag2fill ** 2)
		print(self.prev_state.shape)
		print(g_h[:-1].shape)
		h_shift = np.concatenate((self.prev_state.reshape(1,-1), g_h[:-1]), axis = 0)
		print(h_shift.shape)
		print(g_a.shape)
		self.grad_W = np.dot(g_a.transpose(), h_shift)          # dL/dW
		print(self.grad_W.shape)
		self.grad_U = np.dot(g_a.transpose(), input_sequence)   # dL/dU
		print(self.grad_U.shape)
		self.grad_b = np.sum(g_a, axis=1)  # mean?              # dL/db
		self.grad_c = np.sum(g_o, axis=1)  # mean?              # dL/dc
		self.grad_V = np.dot(g_o.transpose(), self.h)           # dL/dV

		return Loss

	def recall(self, sample_len=200, *args):
		if len(args) < 2:
			init_state = np.zeros(self.hidden_size)
			if len(args) < 1:
				init_char = np.zeros(self.input_size)
				init_char[np.random.randint(0, self.input_size)] = 1

		generated_sample = [init_char]
		for i in range(sample_len):
			out = self.evaluate(x=init_char, prev_state=init_state)
			sample = np.random.choice(self.input_size, out['p'])
			init_char = np.zeros(self.input_size)
			init_char[sample] = 1
			init_state = out['h']
			generated_sample.append(init_char)

		return generated_sample
