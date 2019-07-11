import time
from model import Model
from preprocessing import PreProcessing

class MultiLSTMClassifier:
    p = PreProcessing()
    m = Model(p.df)

    def timeit(f):
        def timed(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()

            print ('func:%r args:[%r, %r] took: %2.4f sec' % \
                   (f.__name__, args, kw, te-ts))
            return result
        return timed



    @timeit
    def pipeline(self):
        p = PreProcessing()
        p.exploratory_analysis()
        p.print_complaint(4165)
        df=p.clean_df()
        p.print_complaint(4165)
        model = Model(df)
        model.LSTMModel()
        model.test()
        print ("pipeline finished")



