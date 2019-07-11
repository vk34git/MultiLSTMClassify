import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, LSTM
from keras.callbacks import EarlyStopping

class Model:

    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100

    def __init__(self, df):
        """ Create a new point at the origin """
        self.df = df


    def LSTMModel(self):


        df = self.df

        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        self.tokenizer = tokenizer

        """Truncate and pad the input sequences so that they are all in the same length for modeling."""
        X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X.shape)

        """    Converting categorical labels to numbers. """

        print('Number of categories:', df['Product'].unique())

        Y = pd.get_dummies(df['Product']).values
        print('Shape of label tensor after:', Y.shape, ' ', Y[0])

        """     Train test split. """

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)

        """ The first layer is the embedded layer that uses 100 length vectors to represent each word.
        SpatialDropout1D performs variational dropout in NLP models.
        The next layer is the LSTM layer with 100 memory units.
        The output layer must create 13 output values, one for each class.
        Activation function is softmax for multi-class classification.
        Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function. """

        model = Sequential()
        model.add(Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(13, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 5
        batch_size = 64

        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


        accr = model.evaluate(X_test,Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show();

        plt.title('Accuracy')
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.show()
        self.model = model



    def test(self):
        """The plots suggest that the model has a little over fitting problem,
                more data may help, but more epochs will not help using the current data."""
        new_complaint = [
            'I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.']
        seq = self.tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        pred = self.model.predict(padded)
        labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection',
                  'Mortgage', 'Credit card or prepaid card', 'Student loan', 'Bank account or service',
                  'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan',
                  'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers',
                  'Prepaid card']
        print(pred, labels[np.argmax(pred)])
