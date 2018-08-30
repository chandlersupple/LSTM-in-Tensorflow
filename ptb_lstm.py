# Chandler Supple, 8-30-18

import tensorflow as tf
sess = tf.InteractiveSession()

with open('ptb.train.txt') as ptbtextfile: # We'll train our LSTM network on the PTB dataset
    ptbtext = ptbtextfile.readlines()
    
def numerize(text): # Represents words with values, where each unique word is assigned a unique value
    chars = []
    char_dict = {}
    for line in range (len(text)):
        for char in range (len(text[line])):
            if text[line][char] not in chars:
                chars.append(text[line][char])
                char_dict[text[line][char]] = len(chars) - 1
    new_text = []
    for line in range (len(text)):
        for char in range (len(text[line])):
            new_text.append(char_dict.get(text[line][char]))
    return new_text, len(chars), chars, char_dict
        
ntext, vocab_size, chars, char_dict = numerize(ptbtext) # Applies the 'numerize' function to our data

class ptbLSTM():
    def __init__(self, ntext, vocab_size, batch_size, time_steps, hidden_units, num_layers):
        # Parameters
        hidden_units = hidden_units 
        num_layers = num_layers
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.ntext = ntext
        self.vocab_size = vocab_size
        
        # Containers for the input and target data
        self.x = tf.placeholder(tf.int32, [self.batch_size, self.time_steps])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.time_steps])
        
        with tf.variable_scope('main_scope', reuse= tf.AUTO_REUSE):
            # Building the LSTM layers
            lstm_unit = tf.contrib.rnn.BasicLSTMCell(hidden_units)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_unit] * num_layers)
            
            init_state = stacked_lstm.zero_state(self.batch_size, tf.float32) # Memory initial state
            init = tf.random_uniform_initializer(-0.1, 0.1) # Variable initializer
            
            embedding = tf.get_variable('embedding', [self.vocab_size, hidden_units], initializer= init)
            ebx = tf.nn.embedding_lookup(embedding, self.x) # Embeds our input data
            
            output, new_state = tf.nn.dynamic_rnn(stacked_lstm, ebx, initial_state= init_state)
            output = tf.reshape(output, [-1, hidden_units])
            
            # Fully connected layer
            W1 = tf.get_variable('W1', [hidden_units, self.vocab_size], initializer= init)
            b1 = tf.get_variable('b1', [self.vocab_size], initializer= init)
            
            self.logits = tf.matmul(output, W1) + b1
            _3dlogits = tf.reshape(self.logits, [self.batch_size, self.time_steps, self.vocab_size])
            
            # Evaluates performance
            loss = tf.contrib.seq2seq.sequence_loss(_3dlogits, self.y, tf.ones([batch_size, time_steps]), average_across_timesteps= False)
            self.cost = tf.reduce_sum(loss)
            
            # Applies gradients descent on all trainable variables
            optimizer = tf.train.AdamOptimizer(0.01)
            tvars = tf.trainable_variables()
            tvars_grads = tf.gradients(self.cost, tvars)
            grads, _ = tf.clip_by_global_norm(tvars_grads, 15)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def accuracy(self, lx, ly): # Calculates the accuracy of the model
        soft_out = tf.nn.softmax(tf.reshape(sess.run(ptblstm.logits, {ptblstm.x: lx, ptblstm.y: ly}), [-1, ptblstm.vocab_size]))
        so_pred = tf.cast(tf.argmax(soft_out, 1), tf.int32)
        correct_preds = tf.equal(so_pred, tf.reshape(ly, [-1]))
        return sess.run(tf.reduce_mean(tf.cast(correct_preds, tf.float32)))
    
    def train(self, sess, feed_dict):
        return sess.run(self.train_op, feed_dict= feed_dict)
    
    def build_batch(self, batch_iter, batches_in_epoch): # Returns the input and target for a given batch
        batchx = []
        batchy = []
        for bb in range (self.batch_size):
            batch_marker = (batch_iter * self.time_steps * self.batch_size) + (bb * self.time_steps)
            batchx.append(self.ntext[batch_marker:batch_marker+self.time_steps])
            batchy.append(self.ntext[batch_marker+1:batch_marker+self.time_steps+1])
        return batchx, batchy
    
def sample(seed, sess, num_chars, time_steps, chars, char_dict): # The seed must have the same number of characters as there are time steps
    stext = []
    num_seed = []
    seed = list(seed)
    for char in range (len(seed)):
        num_seed.append(char_dict.get(seed[char]))
    predictions = sess.run(sample_ptb.logits, {sample_ptb.x: [num_seed] * sample_ptb.batch_size})
    pred_am = sess.run(tf.argmax(predictions, axis= 1))
    for ts in range (time_steps):
        stext.append(chars[pred_am[ts]])
    for char in range (num_chars):
        predictions = sess.run(sample_ptb.logits, {sample_ptb.x: [pred_am] * sample_ptb.batch_size})
        pred_am = sess.run(tf.argmax(predictions, axis= 1))
        stext.append(chars[pred_am[-1]])
    return stext
        
ptblstm = ptbLSTM(ntext, vocab_size, 16, 32, 256, 2) # Creates an instance of our 'spLSTM' class
sample_ptb = ptbLSTM(ntext, vocab_size, 1, 32, 256, 2) # Creates an instance with a single batch for sampling

batches_in_epoch = len(ntext) // (ptblstm.batch_size * ptblstm.time_steps + 1) # Determines the number of batches in each epoch
sess.run(tf.global_variables_initializer()) # Initializes variables

# Training
for epoch_iter in range (32): # Tge number of epochs the network will perform
    for batch_iter in range (batches_in_epoch):
        lx, ly = ptblstm.build_batch(batch_iter, batches_in_epoch)
        ptblstm.train(sess, {ptblstm.x: lx, ptblstm.y: ly})
        if batch_iter % 200 == 0:
            cost = sess.run(ptblstm.cost, {ptblstm.x: lx, ptblstm.y: ly})
            acc = ptblstm.accuracy(lx, ly)
            print('Cost: %s, Accuracy: %s, %s / %s Batches, Epoch %s' %(cost, acc, batch_iter, batches_in_epoch, epoch_iter))
            p_bar = []
            for p_batch in range (1, batches_in_epoch, (batches_in_epoch // 100)):
                if batch_iter < p_batch:
                    p_bar.append('-')
                if batch_iter >= p_batch:
                    p_bar.append('>')
            print(''.join(p_bar))                
            print('\n')
    print(''.join(sample('This statement will seed the net', sess, 100, 32, chars, char_dict))) # Samples text from our network at the end of each epoch
