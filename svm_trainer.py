# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:15:18 2018

@author: mosbah emad
"""
from sklearn import svm
from sklearn.model_selection import cross_validate,train_test_split
import numpy

class Svm_model:
    '''
        svm trainer based on sklearn module ( libsvm )
    '''
    
    def __init__(self , k = 10, test_size = 0.3 , random_state = 0 , c=40):
        self.__model = svm.SVC(C=c)
        self.__data = []
        self.__cls = []
        self.__k_folds = k
        self.score = []
        self.__test_size = test_size
        self.__random_state = random_state
    
    def load_data(self,file_data,file_cls):
        '''
         LOAD DATA FROM FILE DATA TO ATTRIBUT DATA
         LOAD CLASS OF DATA FROM FILE CLS TO ATTRIBUT CLS
        '''
        self.__data = numpy.loadtxt(file_data,delimiter = ',')
        self.__cls = numpy.loadtxt(file_cls, delimiter = ',')
        
    def cross_validation_test(self):
        '''
            PROVIDE CROSS VALIDATION NUMBER OF PARTITION EQUAL TO k (ATTRIBUTE)
        '''
        if ( len(self.__data[0]) == 0 ):
            self.score  = [0]
            return 
        self.score = cross_validate(estimator=self.__model,
                                       X=self.__data, y=self.__cls,
                                       cv=self.__k_folds,
                                       n_jobs=2)
    
    def train(self):
        '''
             TRAIN SVM WITH DATA SPLITED WITH >train_test_split<
        '''
        X_train, X_test, y_train, y_test = train_test_split(
                self.__data,self.__cls,
                test_size=self.__test_size
                )
        self.__model.fit(X_train,y_train)
        self.score = self.__model.score(X_train,y_train)
        
       
    def set_k(self,k):
        self.__k_folds = k
        
    def set_data_cls(self,data,cls):
        '''
            set the data and the class to be trained 
        '''
        self.__data = data
        self.__cls = cls
        
    def set_data(self , data):
        '''
            set the data
        '''
        self.__data = data
        
    def set_cls(self,cls):
        '''
            set classes
        '''
        self.__cls = cls
        
    def get_mean_scor_validation_train(self):
        '''
            after training the svm model  get the mean accuracy in test reselt
        '''
        return numpy.mean(self.score['test_score'])
    
    def get_max_scor_validation_train(self):
        '''
             after tarining the svm model get the max accuracy get it when test the model
        '''
        return numpy.max(self.score['test_score'])
    
    def get_min_scor_validation_train(self):
        '''
            get the min value of accurcy of the model after training wiht test data
        '''
        return numpy.min(self.score['test_score'])
        
if __name__ == '__main__':
    m = Svm_model()
    print('start learning')
    m.load_data('data-normalize\wavlete_,coif1.csv','cls.csv')
    m.cross_validation_test()
    print('ok',m.get_max_scor_validation_train(),m.get_mean_scor_validation_train(),m.get_min_scor_validation_train())  
    print('fine',m.score)      
        