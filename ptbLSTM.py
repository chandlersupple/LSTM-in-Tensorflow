# Chandler Supple, 8-30-18

import tensorflow as tf
sess = tf.InteractiveSession()

with open('ptb.train.txt') as sptextfile: # We'll train our LSTM network on all Shakespeare's published plays
    sptext = sptextfile.readlines()
    
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
        
text, vocab_size, chars, char_dict = numerize(sptext) # Applies the 'numerize' function to our data

class spLSTM():
    
    def __init__(self, text, vocab_size, batch_size, time_steps, hidden_units, num_layers, mode):
        # Parameters
        hidden_units = hidden_units 
        num_layers = num_layers
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.text = text
        self.vocab_size = vocab_size
        self.mode = mode
        
        # Containers for the input and target data
        self.x = tf.placeholder(tf.int32, [self.batch_size, self.time_steps])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.time_steps])
        
        with tf.variable_scope('main_scope', reuse= tf.AUTO_REUSE):
            # Building the LSTM layers
            lstm_layers = [tf.contrib.rnn.BasicLSTMCell(hidden_units)] * num_layers
            dropout_lstm = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob= 0.5) for lstm in lstm_layers]
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(dropout_lstm)
            
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
        soft_out = tf.nn.softmax(tf.reshape(sess.run(splstm.logits, {splstm.x: lx, splstm.y: ly}), [-1, splstm.vocab_size]))
        so_pred = tf.cast(tf.argmax(soft_out, 1), tf.int32)
        correct = tf.equal(so_pred, tf.reshape(ly, [-1]))
        return sess.run(tf.reduce_mean(tf.cast(correct, tf.float32)))
    
    def train(self, sess, feed_dict):
        return sess.run(self.train_op, feed_dict= feed_dict)
    
    def build_batch(self, batch_iter, batches_in_epoch): # Returns the input and target for a given batch
        batchx = []
        batchy = []
        for bb in range (self.batch_size):
            batch_marker = (batch_iter * self.time_steps * self.batch_size) + (bb * self.time_steps)
            batchx.append(self.text[batch_marker:batch_marker+self.time_steps])
            batchy.append(self.text[batch_marker+1:batch_marker+self.time_steps+1])
        return batchx, batchy
    
def sample(sess, words, time_steps, seed, chars, char_dict):
    _text = []
    seed = list(seed)
    num_seed = []
    for char in range (len(seed)):
        num_seed.append(char_dict.get(seed[char]))
    predictions = sess.run(sample_rnn.logits, {sample_rnn.x: [num_seed] * sample_rnn.batch_size})
    pred_am = sess.run(tf.argmax(predictions, axis= 1))
    for ts in range (time_steps):
        _text.append(chars[pred_am[ts]])
    for word in range (words):
        predictions = sess.run(sample_rnn.logits, {sample_rnn.x: [pred_am] * sample_rnn.batch_size})
        pred_am = sess.run(tf.argmax(predictions, axis= 1))
        _text.append(chars[pred_am[-1]])
    return _text
        
splstm = spLSTM(text, vocab_size, 16, 32, 256, 2, 1) # Creates an instance of our 'spLSTM' class
sample_rnn = spLSTM(text, vocab_size, 1, 32, 256, 2, 1)

batches_in_epoch = len(text) // (splstm.batch_size * splstm.time_steps + 1)
sess.run(tf.global_variables_initializer()) # Initializes variables

# Training
for epoch_iter in range (32):
    for batch_iter in range (batches_in_epoch):
        lx, ly = splstm.build_batch(batch_iter, batches_in_epoch)
        splstm.train(sess, {splstm.x: lx, splstm.y: ly})
        if batch_iter % 200 == 0:
            cost = sess.run(splstm.cost, {splstm.x: lx, splstm.y: ly})
            acc = splstm.accuracy(lx, ly)
            print('Cost: %s, Accuracy: %s, %s / %s Batches, Epoch %s' %(cost, acc, batch_iter, batches_in_epoch, epoch_iter))
            p_bar = []
            for p_batch in range (1, batches_in_epoch, (batches_in_epoch // 100)):
                if batch_iter < p_batch:
                    p_bar.append('-')
                if batch_iter >= p_batch:
                    p_bar.append('>')
            print(''.join(p_bar))                
            print('\n')
    print(''.join(sample(sess, 100, 32, 'i believe that he had done a rat', chars, char_dict)))
