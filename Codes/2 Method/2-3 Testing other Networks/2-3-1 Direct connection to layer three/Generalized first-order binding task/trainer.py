import numpy as np
import tensorflow as tf

def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            y_train: np.ndarray = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.RMSprop(learning_rate=1e-4),
            loss_fn_kwargs: dict = None, # Optimizers: RMSprop > Adam > Nadam > Adagrad >Adadelta > Ftrl
            epochs: int = 1000000,
            batch_size: int = 1,
            buffer_size: int = 2048,
            shuffle: bool = False,
            verbose: bool = True,
            show_model_interface_vector: bool = False
            ) -> None:
    
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    X_train
        Training batch.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    loss_fn_kwargs
        Kwargs for loss function.
    epochs
        Number of training epochs.
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    shuffle
        Whether to shuffle training data.
    verbose
        Whether to print training progress.
    """
    
    model.show_interface_vector=show_model_interface_vector

    # Create dataset
    if y_train is None:  # Unsupervised model
        train_data = X_train
    else:
        train_data = (X_train, y_train)
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    if shuffle:
        train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    # Iterate over epochs
    history=[]
    for epoch in range(epochs):
        if verbose:
            pbar = tf.keras.utils.Progbar(target=epochs, width=40, verbose=1, interval=0.05)

        # Iterate over the batches of the dataset
        for step, train_batch in enumerate(train_data):

            if y_train is None:
                X_train_batch = train_batch
            else:
                X_train_batch, y_train_batch = train_batch

            with tf.GradientTape() as tape:
                preds = model(X_train_batch)

                if y_train is None:
                    ground_truth = X_train_batch
                else:
                    ground_truth = y_train_batch

                # Compute loss
                if tf.is_tensor(preds):
                    args = [ground_truth, preds]
                else:
                    args = [ground_truth] + list(preds)

                if loss_fn_kwargs:
                    loss = loss_fn(*args, **loss_fn_kwargs)
                else:
                    loss = loss_fn(*args)

                if model.losses:  # Additional model losses
                    loss += sum(model.losses)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #history.append(loss.numpy().mean()) #in each train    
        
        if verbose:
                loss_val = loss.numpy().mean()
                pbar_values = [('loss', loss_val)]
                pbar.update(epoch+1, values=pbar_values)

        history.append(loss.numpy().mean()) #in each epochs
    
    model.show_interface_vector= not show_model_interface_vector
    return history
            