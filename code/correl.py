import pandas as pd
import numpy as np
import pickle
def correlation(df_label):

    r = df_label.corr(method='pearson')
    print(r)
     
    r.to_csv("corrnew.csv",index=None)

def drop_correlated_columns(df_train,df_test):
                
    final_list=[]
    pkl_file = open('dictionary9.pkl', 'rb')
    dict_corr_col1= pickle.load(pkl_file)
    pkl_file.close()
    for k,v in dict_corr_col1.items():
        print( k,v[0],v[1])
        #final_list.append(v[0])
        final_list.append(k)
             
    print("size of list",len(final_list))
    print(final_list)

    new_list = [str(x) for x in final_list]
    #print(new_list)
    df_train = df_train.drop(columns=new_list)
    df_test = df_test.drop(columns=new_list)
    #df_train = pd.DataFrame(df_train,columns=new_list)
    #df_test = pd.DataFrame(df_test,columns=new_list)
    #df = df.iloc[:,1:]
    print(df_train.shape)
    print(df_test.shape)
    


    df_train.to_csv("train_bp-2.csv",index=None)
    df_test.to_csv("test_bp-2.csv",index=None)
        
    
def make_dictionary_correlation(threshold):
    corr1 = pd.read_csv("corrnew.csv")
        
    condition = (corr1 >threshold)
    coorelationvalue= np.extract(condition,corr1)
    #print(len(coorelationvalue))
    dictionary = np.where(corr1 >threshold) # gives an 2d array where row1 is x coordinates and row2 is y coordinate respectively.
    #print(len(dictionary[0]))
    listOfCoordinates= list(zip(dictionary[0], dictionary[1],coorelationvalue))  # we are making dictionarywith coordinates and coreelation value
    dic={}
    for cord in listOfCoordinates:
        #print(cord[0],cord[1],cord[2])
        if cord[0]!=cord[1] and cord[0] < cord[1]:
            if cord[0] not in dic:
                dic[cord[0]]=(cord[1],cord[2])   # if dic have (15 -> 20,0.95) i.e 15 col is corel to 20 and coorealtion is 0.95
            else:                                # now if (15-> 21,0.90) , then we check (0.95>0.90) , hence finally dic contain
                 if cord[2] >  dic[cord[0]][1]:   # (15->20,0.95)
                     dic[cord[0]]=(cord[1],cord[2])
    output = open('dictionary9.pkl', 'wb')
    pickle.dump(dic, output)
    output.close()
     
    pkl_file = open('dictionary9.pkl', 'rb')
    dict_corr_col1= pickle.load(pkl_file)
    pkl_file.close()
    for k,v in dict_corr_col1.items():
        print( k,v[0],v[1])
    
    






 


    


df_train = pd.read_csv("train-bp.csv")
df_test = pd.read_csv("test-bp.csv")

df_label = df_train.iloc[:,1:] # all rows col start from 1st to end
correlation(df_label)

make_dictionary_correlation(0.9)
drop_correlated_columns(df_train,df_test)







