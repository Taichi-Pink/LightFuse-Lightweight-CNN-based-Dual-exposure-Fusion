import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, weight_path='gen.h5',  patience=0):
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.weight_path = weight_path
        super(EarlyStoppingAtMinLoss, self).__init__()

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch=0, logs=None, name='loss'):
        if name == "d_loss":
          current = logs.get("d_loss")
        else:
          current = logs.get("g_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weight_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                #self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            
class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule=None):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch=0, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
        # (epoch to start, learning rate) tuples
        (3, 0.05),
        (6, 0.01),
        (9, 0.005),
        (12, 0.001),
    ]
    
def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
    
def named_logs(model, logs):
    result = {}
    logs_list = []
    logs_list.append(logs)
    model_list = []
    model_list.append(model)
    zipped = zip(model_list, logs_list)
    zip_list = list(zipped)
    for l in zip_list:
        result[l[0]] = l[1]
   
    return result

def tensorboard_summary(predictions, hdrs, writer, model, step_count):
  with writer.as_default(), tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.image("G_ref", hdrs, step=step_count)          
    tf.contrib.summary.image("gen", predictions, step=step_count)
    #tf.contrib.summary.image("l1", l1, step=step_count)
    #tf.contrib.summary.image("l2", l2, step=step_count)
                
    for layer in model.layers:
      if not layer.weights:
        continue
      for weight, weights_numpy_array in zip(layer.weights, layer.get_weights()):
        weights_name = weight.name.replace(":", "_")
        tf.contrib.summary.histogram(weights_name, weights_numpy_array, step=step_count)

def record(model, bs, result_path):
  tensorboard_3 = keras.callbacks.TensorBoard(log_dir=result_path+'tensorboard/',batch_size=bs)
  #lr_auto_gen = keras.callbacks.ReduceLROnPlateau(monitor='g_loss', factor=0.2, cooldown=0,patience=0,mode="min", min_lr=.00001)
  early_gen = EarlyStoppingAtMinLoss(weight_path='model1.h5', patience=2)
  
  #lr_auto_gen.set_model(model)
  tensorboard_3.set_model(model)
  early_gen.set_model(model)
  
  writer = tf.contrib.summary.create_file_writer(result_path+'tfsummary/')
  return writer, tensorboard_3, early_gen
  
              