
import numpy as np
import string
from typing import List
import unittest

def create_datasets(sequences_x, sequences_y, p_train=0.8, p_val=0.1, p_test=0.1):
    """
    Take inputs(sequences_x) and targets(sequences_y)
    and divide them to train, val, test datasets
    according to p_train, p_val, p_test coeffs
    """
    rng_state = np.random.get_state()
    np.random.shuffle(sequences_x)
    np.random.set_state(rng_state)
    np.random.shuffle(sequences_y)
    # Define partition sizes
    num_train = int(len(sequences_x)*p_train)
    num_val = int(len(sequences_x)*p_val)
    num_test = int(len(sequences_x)*p_test)

    # Split sequences into partitions
    inputs_train = sequences_x[:num_train]
    inputs_val = sequences_x[num_train:num_train+num_val]
    inputs_test = sequences_x[-num_test:]

    # Split sequences into partitions
    targets_train = sequences_y[:num_train]
    targets_val = sequences_y[num_train:num_train+num_val]
    targets_test = sequences_y[-num_test:]

    # Create datasets
    training_set = list(zip(inputs_train, targets_train))
    validation_set = list(zip(inputs_val, targets_val))
    test_set = list(zip(inputs_test, targets_test))

    return training_set, validation_set, test_set


def clip_gradient_norm(grads, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """
    max_norm = float(max_norm)
    total_norm = 0
    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If the total norm is larger than the maximum allowable norm, then clip the gradients
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef
    return grads


def backward_pass(inputs, outputs, hidden_states, targets, params):
    """
    Computes the backward pass of a RNN.

    Args:
     `inputs`: sequence of inputs to be processed
     `outputs`: sequence of outputs from the forward pass
     `hidden_states`: sequence of hidden_states from the forward pass
     `targets`: sequence of targets
     `params`: the parameters of the RNN
    """
    # unpack the parameters
    U, V, W, b_hidden, b_out, embedding_weight = params

    # Initialize gradients as zero
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)

    # Keep track of hidden state derivative and loss
    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0

    # embed input word indexes
    inputs = [embedding_weight[x] for x in inputs]

    # For each element in output sequence
    # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
    for t, output in reversed(list(enumerate(outputs))):
        # Compute cross-entropy loss (as a scalar)
        loss += -np.mean(np.log(output+1e-12) * targets[t])

        # Backpropagate into output (derivative of cross-entropy with softmax)
        d_o = output.copy()
        d_o[np.argmax(targets[t])] -= 1

        # Backpropagate into W
        d_W += np.dot(d_o, hidden_states[t].T)
        d_b_out += d_o

        # Backpropagate into h
        d_h = np.dot(W.T, d_o) + d_h_next

        # Backpropagate through non-linearity
        d_f = tanh(hidden_states[t], derivative=True) * d_h
        d_b_hidden += d_f

        # Backpropagate into U
        d_U += np.dot(d_f, inputs[t].T)

        # Backpropagate into V
        d_V += np.dot(d_f, hidden_states[t-1].T)
        d_h_next = np.dot(V.T, d_f)

    # Pack gradients
    grads = d_U, d_V, d_W, d_b_hidden, d_b_out
    return loss, grads


def init_param_orthogonal(param: np.ndarray) -> np.ndarray:
    """
    Initialize weights orthogonally
    Paper: https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")
    rows, cols = param.shape
    if rows < cols:
        new_param = np.random.randn(cols, rows)
    else:
        new_param = np.random.randn(rows, cols)
    # Compute QR factorization
    orthogonal_arr, _ = np.linalg.qr(new_param)
    # invert signs according to diagonal elements
    orthogonal_arr *= np.sign(np.diag(orthogonal_arr))
    if rows < cols:
        orthogonal_arr = orthogonal_arr.T
    return orthogonal_arr


def init_rnn(hidden_size, vocab_size, embedding_dim):
    """
    Initialize  recurrent neural network

    Args:
     `hidden_size`: dimension of the hidden state
     `vocab_size`: size of vocabulary
     `embedding_dim`: dimension of embedding
    """
    # Matrix for embedding of word indexes
    embedding_weight = np.random.normal(0, 1, (vocab_size, embedding_dim, 1))

    # Weight matrix (input to hidden state)
    U = np.zeros((hidden_size, embedding_dim))

    # Weight matrix (hidden state to hidden state)
    V = np.zeros((hidden_size, hidden_size))

    # Weight matrix (hidden state to output)
    W = np.zeros((vocab_size, hidden_size))

    # Bias of hidden state
    b_hidden = np.zeros((hidden_size, 1))

    # Bias of output
    b_out = np.zeros((vocab_size, 1))

    # Initialize weights
    U = init_param_orthogonal(U)
    V = init_param_orthogonal(V)
    W = init_param_orthogonal(W)

    # Return parameters as a tuple
    return U, V, W, b_hidden, b_out, embedding_weight


def sigmoid(input_array: np.ndarray, derivative=False) -> np.ndarray:
    """
    Computes the element-wise sigmoid activation function for an array input_array.

    Args:
     `input_array`: input numpy array
     `derivative`: if set to True will return the derivative of sigmoid
    """
    input_array = input_array + 1e-12
    res = 1 / (1 + np.exp(-input_array))

    if derivative:
        return res * (1 - res)
    else:
        return res


def tanh(input_array: np.ndarray, derivative=False) -> np.ndarray:
    """
    Computes the element-wise tanh activation function for input_array.

    Args:
     `input_array`: input numpy array
     `derivative`: if set to True will return the derivative of tanh
    """
    input_array = input_array + 1e-12
    exp1 = np.exp(input_array)
    exp2 = np.exp(-input_array)
    res = (exp1-exp2)/(exp1+exp2)

    if derivative:
        return 1-res**2
    else:
        return res


def softmax(input_array: np.ndarray) -> np.ndarray:
    """
    Computes the softmax for an array x.

    Args:
     `input_tensor`: input numpy array
     `derivative`: if set to True will return the derivative of softmax
    """
    input_array = input_array + 1e-12
    res = np.exp(input_array)
    return res / np.sum(res)


def forward_pass(inputs: List[np.ndarray], hidden_state: np.ndarray, params: tuple) -> tuple:
    """
    Computes the forward pass of a RNN.

    Args:
     `inputs`: sequence of inputs to be processed
     `hidden_state`: initialized hidden state
     `params`: parameters of the RNN
    """
    # unpack parameters
    U, V, W, b_hidden, b_out, embedding_weight = params

    outputs, hidden_states = [], []
    # embed input index vectors
    inputs = [embedding_weight[x] for x in inputs]
    # For each element in input sequence
    for input_emb in inputs:
        # Compute new hidden state
        hidden_state = tanh(np.dot(U, input_emb) + np.dot(V, hidden_state) + b_hidden)

        # Compute output
        out = softmax(np.dot(W, hidden_state) + b_out)

        # Save results and continue
        outputs.append(out)
        hidden_states.append(hidden_state.copy())

    return outputs, hidden_states


def optimize_sgd(params: tuple, grads: tuple,
                learning_rate: float=1e-3, clip_norm: float=0.5) -> tuple:
    """
    Clip gradients with clip_norm and update model params according to grads with learning_rate, 
    """
    grads = clip_gradient_norm(grads, clip_norm)
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
    return params


def one_hot_encode(idx: int, vocab_size: int) -> np.ndarray:
    """
    Encode index of word to one hot vector [0, ..., 1.0, ..., 0]
    Args:
     `idx`: index of word
     `vocab_size`: size of the vocabulary    

    Returns a 1-D numpy array of length `vocab_size`.
    """
    one_hot = np.zeros((vocab_size, 1))
    one_hot[idx][0] = 1.
    return one_hot


class TestTrainInference(unittest.TestCase):
    """
    Test train and inference
    """
    def test_train(self):
        with open('train.txt', 'r') as _file:
            train_text = _file.read()
        train_words = []
        strip_str = string.punctuation + string.whitespace
        for line in train_text.split('\n'):
            for sentence in line.split('.'):
                words = [x.lower().strip(strip_str) for x in sentence.split() if x != '—']
                train_words.extend(words)
        # size of input/target sequences
        window_size = 3
        # size of word embedding
        embedding_dim = 128
        p_train = 0.8
        # num of training epochs
        num_epochs = 2
        # size of hidden state
        hidden_size = 256
        # learning rate of SGD optimizer
        lr = .2
        # gradient norm clipping
        clip_norm = .8
        word_to_idx = {}
        idx_to_word = {}
        word_to_idx = {word: i for i, word in enumerate(sorted(set(train_words)))}
        idx_to_word = {i: word for word, i in word_to_idx.items()}
        data_x = []
        data_y = []
        n = len(train_words)
        vocab_size = len(word_to_idx)
        for i in range(0, n - window_size * 2, 1):
            rb = i + window_size
            words_x = train_words[i:rb]
            # input array of indexes of words
            sample_x = np.array([word_to_idx[word] for word in words_x])
            words_y = train_words[rb:rb + window_size]
            # one-hot encoded target array
            sample_y = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in words_y])
            data_x.append(sample_x)
            data_y.append(sample_y)
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        p_val = .5 - p_train / 2
        training_set, validation_set, test_set = create_datasets(data_x, data_y, p_train, p_val, p_val)
        train_num = len(training_set)
        val_num = len(validation_set)

        # Initialize a network
        params = init_rnn(hidden_size, vocab_size, embedding_dim)

        # Initialize hidden state as zeros
        hidden_state = np.zeros((hidden_size, 1))

        # Track loss
        training_loss, validation_loss = [], []
        for i in range(num_epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0

            for inputs, targets in validation_set:
                # Re-initialize hidden state
                hidden_state = np.zeros_like(hidden_state)
                outputs, hidden_states = forward_pass(inputs, hidden_state, params)
                loss, _ = backward_pass(inputs, outputs, hidden_states, targets, params)
                self.assertFalse(np.all(np.isnan(loss)))
                self.assertFalse(np.all(np.isnan(outputs)))
                # Update loss
                epoch_validation_loss += loss

            for inputs, targets in training_set:
                # Re-initialize hidden state
                hidden_state = np.zeros_like(hidden_state)
                outputs, hidden_states = forward_pass(inputs, hidden_state, params)
                loss, grads = backward_pass(inputs, outputs, hidden_states, targets, params)
                if np.isnan(loss):
                    raise ValueError('Gradients have vanished!')
                # Update parameters
                params = optimize_sgd(params, grads, lr, clip_norm)
                self.assertFalse(np.all(np.isnan(loss)))
                self.assertFalse(np.all(np.isnan(outputs)))
                self.assertFalse(all(np.all(np.isnan(param)) for param in params))
                self.assertFalse(all(np.all(np.isnan(param)) for param in grads))
                # Update loss
                epoch_training_loss += loss
                
            # Save loss for plot
            training_loss.append(epoch_training_loss / train_num)
            validation_loss.append(epoch_validation_loss / val_num)
        for i in range(2):
            inputs, targets = test_set[i]
            # Initialize hidden state as zeros
            hidden_state = np.zeros((hidden_size, 1))

            # Forward pass
            outputs, hidden_states = forward_pass(inputs, hidden_state, params)
            input_sentence = [idx_to_word[x] for x in inputs]
            output_sentence = [idx_to_word[np.argmax(x)] for x in outputs]
            target_sentence = [idx_to_word[np.argmax(x)] for x in targets]
            self.assertTrue(all(isinstance(x, str) for x in input_sentence))
            self.assertTrue(all(isinstance(x, str) for x in output_sentence))
            self.assertTrue(all(isinstance(x, str) for x in target_sentence))

if __name__ == '__main__':
    unittest.main()
