import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import time



class MultiLSTMClassifier:
    df = None

    def timeit(f):
        def timed(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()

            print ('func:%r args:[%r, %r] took: %2.4f sec' % \
                  (f.__name__, args, kw, te-ts))
            return result


        return timed

    def exploratory_analysis(self):


        # https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

        df = pd.read_csv("consumer_complaints.csv") #.sample(frac=0.2)
        # df.info()
        # https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas

        df = df.astype(str)
        df = df[['Product', 'Consumer complaint narrative']]
        print(df.count())
        df = df.dropna()
        df = df[~df.isin(['nan', 'NaT']).any(axis=1)]
        print(df.count())


        print("product value counts")
        # print(df['Product'].value_counts())

        df.loc[df[
                   'Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
        df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit card or prepaid card'
        df.loc[df['Product'] == 'Payday loan', 'Product'] = 'Payday loan, title loan, or personal loan'
        df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
        df = df[df.Product != 'Other financial service']


        df['Product'].value_counts().sort_values(ascending=False).plot(kind='bar',
                                                                       title='Number complaints in each product')
        # plt.show()

        print(df.columns)

        df.describe()
        self.df = df


    # let's look at how dirty the text is' \
    def print_complaint(self, index):
        # example = df[df.index == index][['Consumer complaint narrative', 'Product']]
        df = self.df
        example = df.iloc[[index]].values[0]
        # p_200 = df.Product[200]
        # df=df.dropna()
        # print(df.count())
        # c = df['Consumer complaint narrative']
        # print(c.iloc[41650])

        # print(c_256)
        print(example)
        print(example[1], ' and Product:', example[0])

    def clean_text(self, text):

        text = text.lower()
        #print(text)

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        stop_words = set(stopwords.words('english'))

        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('',text)
        text = text.replace('x', '')
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text

    def clean_df(self):
        df = self.df
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(self.clean_text)



    def remove_stop_words(self, text):

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        nltk.download('stopwords')

        # example_sent = "This is a sample sentence, showing off the stop words filtration."

        stop_words = set(stopwords.words('english'))

        word_tokens = word_tokenize(text)

        filtered_sentence1 = [w for w in word_tokens if not w in stop_words]

        # filtered_sentence = []

        # for w in word_tokens:
        #     if w not in stop_words:
        #         filtered_sentence.append(w)

        print(word_tokens)
        # print(filtered_sentence)
        print(filtered_sentence1)

    @timeit
    def pipeline(self):
        self.exploratory_analysis()
        self.print_complaint(4165)
        self.clean_df()
        self.print_complaint(4165)
        print ("pipeline finished")



