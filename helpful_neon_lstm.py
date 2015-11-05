from neon.backends import gen_backend
from neon.data import DataIterator, Text, load_text
from neon.initializers import Uniform, GlorotUniform
from neon.layers import GeneralizedCost, LSTM, Affine, Dropout, LookupTable, RecurrentSum
from neon.models import Model
from neon.optimizers import Adagrad
from neon.transforms import Rectlin, Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
import numpy as np
import os
import cPickle
data_root = os.path.expanduser("~") + '/data/CSE255/'

# dummy class for arguments


class Args():
    pass
args = Args()

# the command line arguments
args.backend = 'gpu'
args.batch_size = 128
args.epochs = 2

args.config = None
args.data_dir = '/home/linuxthink/nervana/data'
args.datatype = np.float32
args.device_id = 0
args.evaluation_freq = 1
args.history = 1
args.log_thresh = 40
args.logfile = None
args.model_file = None
args.no_progress_bar = False
args.output_file = '/home/linuxthink/nervana/data/neonlog.hd5'
args.progress_bar = True
args.rng_seed = 0
args.rounding = False
args.save_path = None
args.serialize = 0
args.verbose = 0


num_epochs = args.epochs

# hyperparameters from the reference
batch_size = 128
clip_gradients = True
gradient_limit = 15
vocab_size = 20000
sentence_length = 100
embedding_dim = 128
hidden_size = 128
reset_cells = True

print('batch_size: %s \nvocab_size: %s \nsentence_length: %s \nembedding_dim: %s \nhidden_size: %s' % \
      (batch_size,      vocab_size,      sentence_length,      embedding_dim,      hidden_size))

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

# make dataset
(X_train, y_train), (X_test, y_test), nclass = Text.pad_data(
    os.path.join(data_root, 'train_valid_text_index_in_binary_label_shuffled_10000.pickle'),
    vocab_size=vocab_size, sentence_length=sentence_length)

print "Vocab size - ", vocab_size
print "Sentence Length - ", sentence_length
print "# of train sentences", X_train.shape[0]
print "# of test sentence", X_test.shape[0]

train_set = DataIterator(X_train, y_train, nclass=2)
valid_set = DataIterator(X_test, y_test, nclass=2)

# weight initialization
init_emb = Uniform(low=-0.1 / embedding_dim, high=0.1 / embedding_dim)
init_glorot = GlorotUniform()

layers = [
    LookupTable(
        vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb),
    LSTM(hidden_size, init_glorot, activation=Tanh(),
         gate_activation=Logistic(), reset_cells=True),
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(2, init_glorot, bias=init_glorot, activation=Softmax())
]

print(layers)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
metric = Accuracy()

model = Model(layers=layers)

optimizer = Adagrad(learning_rate=0.01, clip_gradients=clip_gradients)


# configure callbacks
callbacks = Callbacks(model, train_set, args, eval_set=valid_set)

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=num_epochs,
          cost=cost,
          callbacks=callbacks)

# eval model
print "Test  Accuracy - ", 100 * model.eval(valid_set, metric=metric)
print "Train Accuracy - ", 100 * model.eval(train_set, metric=metric)


import ipdb; ipdb.set_trace()

# output result directly
for x, y in valid_set:
  x = model.fprop(x, inference=True)
  print(x.get())
  print(y.get())
  break

# chance to do analysis
import ipdb; ipdb.set_trace()