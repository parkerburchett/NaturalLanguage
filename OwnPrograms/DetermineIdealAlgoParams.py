
from NaturalLanguage.custom_NLTK_Utils import dataLabeling as dl
from NaturalLanguage.custom_NLTK_Utils import AlgoEvalutationUtils as AE
import datetime


def Main():
    start = datetime.datetime.now()
    print('You have started')
    paramList = AE.createParamList()
    counter =1
    for p in paramList:
        try:
            start2 =  datetime.datetime.now()
            print("At {} of {} where N={}".format(counter,len(paramList),p.NmostFrequent))
            counter = counter +1
            FS = dl.create_feature_sets(p)
            print('Created Feature_sets {}:'.format((datetime.datetime.now()-start2)))
            classifiers = AE.createAndTrain_Classifiers(FS)
            print('Trained Classifiers: {}:'.format((datetime.datetime.now()-start2)))
            AE.writeAlgoEvaluation(p, classifiers, FS, fileName="RandomLabelingTest")
            print('Tested Classifiers: {}:'.format((datetime.datetime.now()-start2)))

        except (ValueError):
            print('Error part of speech that breaks: {}'.format(p.PartsOfSpeech))
            print(ValueError)

    print('finished')
    print(datetime.datetime.now() -start)

Main()