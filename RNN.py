import numpy as np

class RNN:
	def __init__(self, input_size, output_size, hidden_size, seq_len):

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
		self.prev_state = np.empty(hidden_size)
		self.h = np.zeros(seq_len, hidden_size)
		self.a = np.empty_like(self.h)
		self.o = np.empty(seq_len, output_size)
		self.p = np.empty_like(self.o)

		# grads
		self.grad_U = np.empty_like(self.U)
		self.grad_W = np.empty_like(self.W)
		self.grad_b = np.empty_like(self.b)
		self.grad_V = np.empty_like(self.V)
		self.grad_c = np.empty_like(self.c)

	def he_init(self, row, columns = None):
		try:
			return np.random.normal(loc = 0.0, scale = 2/np.sqrt(columns), size = (row, columns))
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
		self.prev_state = self.h[-1]

		last_state = self.prev_state
		for char in input_sequence:
			vars = self.evaluate(char, last_state)
			last_state = vars['h']
			h_temp.append(vars['h'])
			a_temp.append(vars['a'])
			o_temp.append(vars['o'])
			p_temp.append(vars['p'])

		self.h = h_temp
		self.a = a_temp
		self.o = o_temp
		self.p = p_temp

	def cross_entropy(self, Y):
		y_p = self.p[Y, np.arange(Y.size)]
		# select ascissa given by Y and scroll all p
		return -np.log(y_p)

	def backprop(self, input_sequence, output_sequence):

		# per il plot
		Loss = sum(self.cross_entropy(output_sequence))

		# dL/dt
		assert np.shape(self.p) == np.shape(output_sequence)

		# init temp grads
		g_h = np.zeros(np.shape(output_sequence))
		g_a = np.zeros(np.shape(self.h))

		# dL/do
		g_o = -(output_sequence - self.p)

		# dL/dh_tau
		g_h[-1] = g_o[-1] @ self.V

		# dL/da_tau
		diag2fill = np.zeros(g_h[-1].shape[0], g_h[-1].shape[0])
		np.fill_diagonal(diag2fill, self.h[-1])
		g_a[-1] = g_h[-1] @ (1-diag2fill**2)

		for i in range(output_sequence-2, -1, -1):
			g_h[i] = g_o[i] @ self.V + g_a[i+1] @ self.W
			np.fill_diagonal(diag2fill, self.h[i])
			g_a[i] = g_h[i] @ (1-diag2fill**2)

		h_shift = np.asarray([self.prev_state, g_h[:-1]])
		self.grad_W = np.dot(g_a, h_shift.transpose())
		self.grad_U = np.dot(g_a, input_sequence.transpose())
		self.grad_b = np.sum(g_a, axis=1) #mean?
		self.grad_c = np.sum(g_o, axis=1)  # mean?
		self.grad_V = np.dot(g_o, self.o.transpose())
