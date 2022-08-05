from flask import Flask, request, render_template


import numpy as np
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import itertools

from tensorflow import keras




# Create Flask object to run
app = Flask(__name__,template_folder= 'templates' )



import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
df = pd.read_csv('new_ing_csv.csv')
X = df[['Age', 'SkinType', 'SkinTone', 'SkinConcerns', 'Gender',
       'Race', 'Climate']]
Xencoded = pd.get_dummies(X)
X = Xencoded
Yencoded_Extracts = pd.get_dummies(df['Extracts'])
Yencoded_All_Age_Suitable_Ing = pd.get_dummies(df['All_Age_Suitable_Ing'])
Yencoded_Anti_Acne_Moisturizer = pd.get_dummies(df['Anti-Acne Moisturizer'])
Yencoded_Antioxidant_Anti_Aging_Moisturizer = pd.get_dummies(df['Antioxidant+Anti-Aging Moisturizer'])
Yencoded_F_Skin_ID_Soothing = pd.get_dummies(df['F_Skin_ID+Soothing'])
Yencoded_M_Antioxidant_Occlusive = pd.get_dummies(df['M_Antioxidant+Occlusive'])
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute = pd.get_dummies(df['EA_Anti_Age+Skin_ID+Cell_Commute'])
Yencoded_IA_Antioxidant = pd.get_dummies(df['IA_Antioxidant'])
Yencoded_YA_Occusive = pd.get_dummies(df['YA_Occusive'])
Yencoded_BR_Anti_Acne = pd.get_dummies(df['BR_Anti_Acne'])
Yencoded_WR_Skin_ID_Occusive = pd.get_dummies(df['WR_Skin_ID+Occusive'])
Yencoded_DRY_Dry = pd.get_dummies(df['DRY_Dry'])
Yencoded_TROPICAL_Antioxidant_Humectant = pd.get_dummies(df['TROPICAL_Antioxidant+Humectant'])
Yencoded_TEMPERATE_Humectant = pd.get_dummies(df['TEMPERATE_Humectant'])
Yencoded_CONTINENAL_Emolient = pd.get_dummies(df['CONTINENAL_Emolient'])
Yencoded_POLAR_Humectants_Occlusive_Emollients = pd.get_dummies(df['POLAR_Humectants+Occlusive+Emollients'])

import keras

Yencoded_Extracts_model = keras.models.load_model('Yencoded_Extracts_model')
Yencoded_All_Age_Suitable_Ing_model = keras.models.load_model('Yencoded_All_Age_Suitable_Ing_model')
Yencoded_Anti_Acne_Moisturizer_model = keras.models.load_model('Yencoded_Anti_Acne_Moisturizer_model')
Yencoded_Antioxidant_Anti_Aging_Moisturizer_model =keras.models.load_model('Yencoded_Antioxidant_Anti_Aging_Moisturizer_model')
Yencoded_F_Skin_ID_Soothing_model = keras.models.load_model('Yencoded_F_Skin_ID_Soothing_model')
Yencoded_M_Antioxidant_Occlusive_model  = keras.models.load_model('Yencoded_M_Antioxidant_Occlusive_model')
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model  = keras.models.load_model('Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model')
Yencoded_IA_Antioxidant_model  = keras.models.load_model('Yencoded_IA_Antioxidant_model')
Yencoded_YA_Occusive_model  = keras.models.load_model('Yencoded_YA_Occusive_model')
Yencoded_BR_Anti_Acne_model  = keras.models.load_model('Yencoded_BR_Anti_Acne_model')
Yencoded_WR_Skin_ID_Occusive_model  = keras.models.load_model('Yencoded_WR_Skin_ID_Occusive_model')
Yencoded_DRY_Dry_model  = keras.models.load_model('Yencoded_DRY_Dry_model')
Yencoded_TROPICAL_Antioxidant_Humectant_model  = keras.models.load_model('Yencoded_TROPICAL_Antioxidant_Humectant_model')
Yencoded_TEMPERATE_Humectant_model  = keras.models.load_model('Yencoded_TEMPERATE_Humectant_model')
Yencoded_CONTINENAL_Emolient_model  = keras.models.load_model('Yencoded_CONTINENAL_Emolient_model')
Yencoded_POLAR_Humectants_Occlusive_Emollients_model  = keras.models.load_model('Yencoded_POLAR_Humectants_Occlusive_Emollients_model')

st = ['Yencoded_Extracts_model','Yencoded_All_Age_Suitable_Ing_model','Yencoded_Anti_Acne_Moisturizer_model','Yencoded_Antioxidant_Anti_Aging_Moisturizer_model','Yencoded_F_Skin_ID_Soothing_model','Yencoded_M_Antioxidant_Occlusive_model','Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model','Yencoded_IA_Antioxidant_model','Yencoded_YA_Occusive_model','Yencoded_BR_Anti_Acne_model','Yencoded_WR_Skin_ID_Occusive_model','Yencoded_DRY_Dry_model','Yencoded_TROPICAL_Antioxidant_Humectant_model','Yencoded_TEMPERATE_Humectant_model','Yencoded_CONTINENAL_Emolient_model','Yencoded_POLAR_Humectants_Occlusive_Emollients_model'
]



modellist = [Yencoded_Extracts_model,Yencoded_All_Age_Suitable_Ing_model,Yencoded_Anti_Acne_Moisturizer_model,Yencoded_Antioxidant_Anti_Aging_Moisturizer_model,Yencoded_F_Skin_ID_Soothing_model,Yencoded_M_Antioxidant_Occlusive_model,Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model,Yencoded_IA_Antioxidant_model,Yencoded_YA_Occusive_model,Yencoded_BR_Anti_Acne_model,Yencoded_WR_Skin_ID_Occusive_model,Yencoded_DRY_Dry_model,Yencoded_TROPICAL_Antioxidant_Humectant_model,Yencoded_TEMPERATE_Humectant_model,Yencoded_CONTINENAL_Emolient_model,Yencoded_POLAR_Humectants_Occlusive_Emollients_model]


Ylist = [Yencoded_Extracts, 
Yencoded_All_Age_Suitable_Ing, 
Yencoded_Anti_Acne_Moisturizer, 
Yencoded_Antioxidant_Anti_Aging_Moisturizer, 
Yencoded_F_Skin_ID_Soothing ,
Yencoded_M_Antioxidant_Occlusive, 
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute,
Yencoded_IA_Antioxidant, 
Yencoded_YA_Occusive, 
Yencoded_BR_Anti_Acne, 
Yencoded_WR_Skin_ID_Occusive, 
Yencoded_DRY_Dry ,
Yencoded_TROPICAL_Antioxidant_Humectant, 
Yencoded_TEMPERATE_Humectant ,
Yencoded_CONTINENAL_Emolient,
Yencoded_POLAR_Humectants_Occlusive_Emollients ]

def out(el):
    
    if el >0.5:
        
        return 1
    else:
        return 0

def arrr(predictionss):
    
    prod = list([0] * len(list(predictionss)))
    maxm = max(predictionss)

    indd = list(predictionss).index(maxm)
    return indd


def givlis(dictionin):
    X_fu = list(X.columns)

    listofzeros = [0] * len(X_fu)
    listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
    listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
    listofzeros[X_fu.index('SkinTone'+'_'+dictionin['SkinTone'])] = 1
    listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
    listofzeros[X_fu.index('Gender'+'_'+dictionin['Gender'])] = 1
    listofzeros[X_fu.index('Race'+'_'+dictionin['Race'])] = 1
    listofzeros[X_fu.index('Climate'+'_'+dictionin['Climate'])] = 1
    return listofzeros

def modreccomenderz(custdetails,modelz,Ymod):
    Y_col = list(Ymod.columns)
    #print(Y_col)
    predictionn = (modelz.predict([givlis(custdetails)]))[0]

    maxm = max(predictionn)

    indd = list(predictionn).index(maxm)



    
   

    return  Y_col[indd]

def modreccomender(custtdetails):
    dc = {}
    final = {}
    for modind in range(len(modellist)):

        predictionn = (modellist[modind].predict([givlis(custtdetails)]))[0]
        maxm = max(predictionn)
        

        indd = list(predictionn).index(maxm)

        dc[st[modind]] = (Ylist[modind].columns)[indd]
    final['Extracts'] = dc['Yencoded_Extracts_model']
        
    if   (custtdetails['Age'] == '13-17') or (custtdetails['Age'] == '18-24')  or (custtdetails['Age'] == '25-34'):
        
          final['Occusive'] = dc['Yencoded_YA_Occusive_model']
            
    if   (custtdetails['Age'] == '35-44') or (custtdetails['Age'] == '45-54')  :
        
          final['Antioxidant'] = dc['Yencoded_IA_Antioxidant_model'] 
            
            
    if   (custtdetails['Age'] == '55-120') :
        
          final['Skin_Identical and Cell_Commute'] = dc['Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model'] 
            
            
            
            
            
    if   (custtdetails['SkinConcerns'] == 'Acne'):
        
          final['Anti-Acne'] = dc['Yencoded_Anti_Acne_Moisturizer_model']
            
            
    if   (custtdetails['SkinConcerns'] == 'Aging') or (custtdetails['Age'] == '45-54')  :
        
          final['Anti-Aging and Antioxidant'] = dc['Yencoded_Antioxidant_Anti_Aging_Moisturizer_model'] 
            
            
            
    if   (custtdetails['Climate'] == 'Continental')  :
        
          final['Emolient'] = dc['Yencoded_CONTINENAL_Emolient_model'] 
            
            
    if   (custtdetails['Climate'] == 'Polar') :
        
          final['Humectants Occlusive and Emollients'] = dc['Yencoded_POLAR_Humectants_Occlusive_Emollients_model']             
            
    if   (custtdetails['Climate'] == 'Tropical') :
        
          final['Antioxidant and Humectant'] = dc['Yencoded_TROPICAL_Antioxidant_Humectant_model']                
        
    if   (custtdetails['Climate'] == 'Dry') :
        
          final['Skin_Identical and Occusive'] = dc['Yencoded_DRY_Dry_model']             
            
    if   (custtdetails['Climate'] == 'Temperate') :
        
          final['Humectant'] = dc['Yencoded_TEMPERATE_Humectant_model']              
            
            
            
            
            
            
            
    if   (custtdetails['Gender'] == 'Male') :
        
          final['Antioxidant and Occlusive'] = dc['Yencoded_M_Antioxidant_Occlusive_model']             
            
    if   (custtdetails['Gender'] == 'Female') :
        
          final['Skin_Identical and Soothing'] = dc['Yencoded_F_Skin_ID_Soothing_model']               
            
    if   (custtdetails['Race'] == 'Black'):
        
          final['Anti-Acne'] = dc['Yencoded_BR_Anti_Acne_model'] 
    if   (custtdetails['Race'] == 'White')  :
        
          final['Skin_Identical and Occusive '] = dc['Yencoded_WR_Skin_ID_Occusive_model']     
            

 
    
   

    return  final












  
  
  
  
  
  
  
  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    
    SkinConcerns = str(request.form.get('SkinConcerns'))

    Age = str(request.form.get('Age'))

    SkinType = str(request.form.get('SkinType'))

    Gender = str(request.form.get('Gender'))

    SkinTone = str(request.form.get('SkinTyone'))

    Race = str(request.form.get('Race'))
    
    Climate = str(request.form.get('Climate'))   

    custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}

    print(custdetails)
    out = {}


    out = modreccomender(custdetails)
    print(out)

    return render_template('index.html', prediction_text= out)



    
    
if __name__ == "__main__":
	app.run()