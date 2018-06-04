import numpy as np
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt



#########################################################################################
import unittest

import numpy as np

def compute_grads_for_matrix(inputs, targets, initial_state, matrix,
                             network, name):
    # Initialize an empty matrix to contain the gradients
    matrix = np.atleast_2d(matrix)
    grad = np.empty_like(matrix)
    h = 1e-4
    # Iterate over the matrix changing one entry at the time
    print('Gradient computations for {} {}, sequence length {}'
          .format(name, matrix.shape, inputs.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] += h
            network.forward(inputs, debug=True, debug_state=initial_state)
            plus_cost = network.cross_entropy(targets)
            matrix[i, j] -= 2 * h
            network.forward(inputs, debug=True, debug_state=initial_state)
            minus_cost = network.cross_entropy(targets)
            grad[i, j] = (plus_cost - minus_cost) / (2 * h)
            matrix[i, j] += h
    return np.squeeze(grad)

def print_grad_diff(grad, grad_num, name=''):
    err = np.abs(grad - grad_num)
    rel_err = err / np.maximum(np.finfo('float').eps,
                               np.abs(grad) + np.abs(grad_num))
    print('Gradient difference {}: {:.2e}'.format(name, np.max(rel_err)))


def test_gradients(input_sequence, output_sequence, network):
    network.forward(input_sequence, debug=True, debug_state=np.zeros(network.hidden_size))
    network.backprop(input_sequence, output_sequence)
    for (param, grad, name) in network.get_weights_gradients():
        grad_num = compute_grads_for_matrix(
            input_sequence, output_sequence, np.zeros(network.hidden_size),
            param, network, name)
        print_grad_diff(grad=grad, grad_num=grad_num, name=name)


if __name__ == '__main__':
    unittest.main()
######################################################################################




class TextSource:
	def __init__(self, filename):
		self.char_sequence = list(self.load_text(filename))
		self.label_encoder = preprocessing.LabelBinarizer()
		self.encoded_text = self.label_encoder.fit_transform(self.char_sequence)
		self.total_chars, self.num_classes = self.encoded_text.shape

	def encode(self, *values):
		if len(values) == 1:
			return np.squeeze(self.label_encoder.transform(list(values[0])))
		else:
			return [self.encode(s) for s in values]

	def decode_to_strings(self, *sequences):
		if len(sequences) == 1:
			return ''.join(self.label_encoder.inverse_transform(np.asarray(sequences[0])))
		else:
			return [self.decode_to_strings(s) for s in sequences]

	@staticmethod
	def load_text(filename):
		with open(filename, 'r') as f:
			return f.read()



class RNN:
	def __init__(self, text_source, hidden_size = 150):

		# input
		self.input_size = text_source.label_encoder.classes_.shape[0]
		self.output_size = self.input_size
		self.hidden_size = hidden_size
		self.text_source = text_source

		# weights
		self.U = self.he_init(hidden_size, self.input_size)
		self.V = self.he_init(self.output_size, hidden_size)
		self.b = self.he_init(hidden_size)
		self.W = self.he_init(hidden_size, hidden_size)
		self.c = self.he_init(self.output_size)

		# momenta
		self.m_U = np.zeros_like(self.U)
		self.m_V = np.zeros_like(self.V)
		self.m_b = np.zeros_like(self.b)
		self.m_W = np.zeros_like(self.W)
		self.m_c = np.zeros_like(self.c)

		# states
		self.prev_state = None
		self.h = np.zeros((1, hidden_size))
		self.a = None
		self.o = None
		self.p = None

		# grads
		self.grad_U = np.empty_like(self.U)
		self.grad_V = np.empty_like(self.V)
		self.grad_b = np.empty_like(self.b)
		self.grad_W = np.empty_like(self.W)
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

		if np.max(np.abs(self.V @ h)) > self.log_Vh[-1]:
			self.log_Vh.append(np.max(np.abs(self.V @ h)))
		else:
			self.log_Vh.append(self.log_Vh[-1])

		if np.max(np.abs(self.V)) > self.log_V[-1]:
			self.log_V.append(np.max(np.abs(self.V)))
		else:
			self.log_V.append(self.log_V[-1])

		if np.max(np.abs(self.c)) > self.log_c[-1]:
			self.log_c.append(np.max(np.abs(self.c)))
		else:
			self.log_c.append(self.log_c[-1])

		if np.max(np.abs(h)) > self.log_h[-1]:
			self.log_h.append(np.max(np.abs(h)))
		else:
			self.log_h.append(self.log_h[-1])

		if np.max(np.abs(o)) > self.log_o[-1]:
			self.log_o.append(np.max(np.abs(o)))
		else:
			self.log_o.append(self.log_o[-1])

		softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
		p = self._softmax(o)
		return {
			'a': a,
			'h': h,
			'o': o,
			'p': p,
		}

	def forward(self, input_sequence, debug=False, debug_state = None):

		h_temp = []
		a_temp = []
		o_temp = []
		p_temp = []
		last_state = self.h[-1]
		self.prev_state = last_state

		if debug:
			last_state = debug_state

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
		# y_p = self.p[Y, np.arange(Y.size)]
		y_p = np.multiply(self.p, Y)
		# or alternatively: y_p = diag(dot(p,Y_t))
		# select ascissa given by Y and scroll all p
		return -np.sum(np.log(np.sum(y_p, axis=1)))
		#return self.cost(Y)

	def cost(self, targets):
		assert self.p.shape == targets.shape
		log_arg = (self.p * targets).sum(axis=1)
		log_arg[log_arg == 0] = np.finfo(float).eps
		return - np.log(log_arg).sum()

	def _softmax(self, o):
		try:
			e = np.exp(o)
			res = e / e.sum()
		except:
			res = np.full_like(o, fill_value=np.finfo(float).eps)
			res[np.argmax(o)] = 1 - (self.output_size - 1) * np.finfo(float).eps
		return res

	def backprop(self, input_sequence, output_sequence):

		# per il plot
		loss = self.cross_entropy(output_sequence)

		# dL/dt
		# assert np.shape(self.p) == np.shape(output_sequence)

		# init temp grads
		g_h = np.zeros((output_sequence.shape[0], self.hidden_size))
		g_a = np.zeros(np.shape(self.h))

		# dL/do
		g_o = -(output_sequence - self.p)

		# dL/dh_tau
		g_h[-1] = g_o[-1] @ self.V

		# dL/da_tau
		diag2fill = np.zeros((g_h[-1].shape[0], g_h[-1].shape[0]))
		#np.fill_diagonal(diag2fill, self.h[-1])
		diag2fill = self.h[-1]
		g_a[-1] = g_h[-1] @ (1 - diag2fill ** 2)

		for i in range(output_sequence.shape[0] - 2, -1, -1):
			# dL/dh_i
			g_h[i] = g_o[i] @ self.V + g_a[i + 1] @ self.W

			# dL/da_i
			#np.fill_diagonal(diag2fill, self.h[i])
			diag2fill = self.h[i]
			g_a[i] = g_h[i] @ (1 - diag2fill ** 2)

		h_shift = np.concatenate((self.prev_state.reshape(1, -1), g_h[:-1]), axis=0)
		self.grad_W = np.dot(g_a.transpose(), h_shift)  # dL/dW
		self.grad_U = np.dot(g_a.transpose(), input_sequence)  # dL/dU
		self.grad_b = np.sum(g_a, axis=0)  # mean?             # dL/db
		self.grad_c = np.sum(g_o, axis=0)  # mean?             # dL/dc
		self.grad_V = np.dot(g_o.transpose(), self.h)  # dL/dV

		######################### BALDAPART  #########################

		balda_grad_W = np.zeros_like(self.W)
		balda_grad_U = np.zeros_like(self.U)
		balda_grad_b = np.zeros_like(self.b)
		grad_V_balda = np.zeros_like(self.V)
		for t in range(output_sequence.shape[0]):
			grad_V_balda += np.outer(g_o[t], self.h[t])
		# print("DIFFERENZA VBALDA-VNOSTRO")
		# print(np.sum(np.abs(self.grad_V-grad_V_balda)))

		dL_da = np.zeros(self.hidden_size)
		for t in range(output_sequence.shape[0] - 1, 0 - 1, -1):
			dL_dh = g_o[t] @ self.V + dL_da @ self.W
			dL_da = dL_dh * (1 - self.h[t]**2)
			balda_grad_W += np.outer(dL_da, h_shift[t])
			balda_grad_U += np.outer(dL_da, input_sequence[t])
			balda_grad_b += dL_da

		"""
		print("DIFFERENZA W_BALDA-W_NOSTRO")
		print(np.sum(np.abs(self.grad_W-balda_grad_W)))
		print("DIFFERENZA U_BALDA-U_NOSTRO")
		print(np.sum(np.abs(self.grad_U-balda_grad_U)))
		print("DIFFERENZA b_BALDA-b_NOSTRO")
		print(np.sum(np.abs(self.grad_b-balda_grad_b)))
		"""
		"""
		self.grad_V = grad_V_balda
		self.grad_W = balda_grad_W
		self.grad_U = balda_grad_U
		"""
		return loss

	def recall(self, sample_len=200, *args):
		if len(args) < 2:
			init_state = np.zeros(self.hidden_size)
			init_state = self.h[-1]
		else:
			init_state = args[1]

		if len(args) < 1:
			init_char = np.zeros(self.input_size)
			init_char[np.random.randint(0, self.input_size)] = 1
		else:
			init_char = args[0]

		generated_sample = [init_char]
		for i in range(sample_len-1):
			out = self.evaluate(x=init_char, prev_state=init_state)
			sample = np.random.choice(self.input_size, p=out['p'])
			init_char = np.zeros(self.input_size)
			init_char[sample] = 1
			init_state = out['h']
			generated_sample.append(init_char)
		return generated_sample

	def RMSProp(self, eta=0.001, gamma=0.9):
		epsilon = 1e-10
		self.clip = 1
		#gamma = 0.5
		self.grad_U = np.clip(self.grad_U, -self.clip, +self.clip)
		self.grad_V = np.clip(self.grad_V, -self.clip, +self.clip)
		self.grad_b = np.clip(self.grad_b, -self.clip, +self.clip)
		self.grad_W = np.clip(self.grad_W, -self.clip, +self.clip)
		self.grad_c = np.clip(self.grad_c, -self.clip, +self.clip)

		self.m_U = gamma * self.m_U + np.square(self.grad_U) * (1 - gamma)
		self.m_V = gamma * self.m_V + np.square(self.grad_V) * (1 - gamma)
		self.m_b = gamma * self.m_b + np.square(self.grad_b) * (1 - gamma)
		self.m_W = gamma * self.m_W + np.square(self.grad_W) * (1 - gamma)
		self.m_c = gamma * self.m_c + np.square(self.grad_c) * (1 - gamma)

		self.delta_U = eta * np.divide(self.grad_U, np.sqrt(self.m_U + epsilon))
		self.delta_V = eta * np.divide(self.grad_V, np.sqrt(self.m_V + epsilon))
		self.delta_b = eta * np.divide(self.grad_b, np.sqrt(self.m_b + epsilon))
		self.delta_W = eta * np.divide(self.grad_W, np.sqrt(self.m_W + epsilon))
		self.delta_c = eta * np.divide(self.grad_c, np.sqrt(self.m_c + epsilon))

		self.U -= self.delta_U
		self.V -= self.delta_V
		self.b -= self.delta_b
		self.W -= self.delta_W
		self.c -= self.delta_c

	def fit(self, n_epochs=100, seq_len=25, eta=0.001, gamma=0.9, sample_len=1000):
		self.log_o = [0]
		self.log_V = [0]
		self.log_c = [0]
		self.log_h = [0]
		self.log_Vh = [0]
		input_data = self.text_source.encoded_text
		generated_sample = []
		init_time = time.time()
		for epoch in range(n_epochs):
			for e in range(np.shape(input_data)[0] - seq_len):
				input_sequence = input_data[e:e + seq_len]
				output_sequence = input_data[e + 1:e + seq_len + 1]
				self.forward(input_sequence)
				new_loss = self.backprop(input_sequence, output_sequence)
				if e == 0:
					loss_history = [new_loss]
				else:
					loss_history.append(loss_history[-1]*0.999 + new_loss*0.001)
				self.RMSProp(eta, gamma)
				if e % 1000 == 0:
					partial_time = time.time() - init_time
					print('###########################################################################################')
					print('epoch: ', epoch, ' - e_index: ', e, '- elapsed time: ', partial_time)
					generated_sample.append(self.recall(sample_len))
					"""
					print(self.text_source.decode_to_strings(generated_sample[-1]))
					plt.plot(loss_history)
					plt.show()
					"""
					plt.plot(self.log_o, label='o')
					plt.plot(self.log_V, label='V')
					plt.plot(self.log_c, label='c')
					plt.plot(self.log_h, label='h')
					plt.plot(self.log_Vh, label='Vh')
					#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
					plt.legend()
					plt.show()
			self.h = np.zeros((1, self.hidden_size))

		generated_sample.append(self.recall(sample_len*100))
		print(self.text_source.decode_to_strings(generated_sample[-1]))
		plt.plot(loss_history)
		plt.show()
		return {
			'history': loss_history,
			'samples': generated_sample,
		}

	def get_weights_gradients(self):
		yield (self.W, self.grad_W, 'W')
		yield (self.U, self.grad_U, 'U')
		yield (self.b, self.grad_b, 'b')
		yield (self.V, self.grad_V, 'V')
		yield (self.c, self.grad_c, 'c')
