import numpy as np
from flask import Flask, request, jsonify, render_template,send_file
import pickle
import pandas as pd
import test
#import requests

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics
from sklearn import preprocessing

app = Flask(__name__)
#model = open('test.py', 'rb')
ay=pd.read_csv("values.csv")

#az=az.style.hide_index()




@app.route('/')





def home():
#For windows you need to use drive name [ex: F:/Example.pdf]
    # path = "C:/Users/bonam/Downloads/q1/New folder/values.csv"
    return render_template('index.html')
# @app.route('/download')
# def downloadFile ():
    
#     return send_file(path, as_attachment=True)
@app.route('/download')
def download():
    path = "/home/ubuntu/values.csv"
    return send_file(path, as_attachment=True)

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    '''

    For rendering results on HTML GUI

    '''

    
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        dd = request.form.get("name")
    
    ch=True
    # String_name = request.form.values()
    # dd = String_name
    
    x=[]
    act_val=0
    predict_val=0
    
    if (len(ay.loc[ay['short_name']==dd])>0):
           x.append((ay.loc[ay['short_name']==dd]))
    else:
        ch=False
        conn = None
        try:
                    # read connection parameters
                    #params = config()

                    # connect to the PostgreSQL server
                    print('Connecting to the PostgreSQL database...')
                    #conn = psycopg2.connect(**params)
                    conn = psycopg2.connect(
                    host="localhost",
                    database="postgres",
                    #Port="5432",
                    user="postgres",
                    password="V@msi123")
                        
                    # create a cursor
                    #cur = conn.cursor()
                    
                # execute a statement
                    print('PostgreSQL database version:')
                    squery = f"SELECT short_name FROM player_data WHERE ts @@ to_tsquery('english', 'ronaldos')"
                    squery = "SELECT short_name,player_face_url FROM player_data WHERE ts @@ to_tsquery('english', '%s')"%(dd, )
                    #"SELECT short_name,player_face_url FROM player_data WHERE ts @@ to_tsquery('english', '{1}')".format(dd2,dd)
                    conn.autocommit = True
                    cursor = conn.cursor()
                    #print(cursor.execute("select * from player_data WHERE short_name='L. Messi'"))
                    cursor.execute(squery)
                    dbver=cursor.fetchall();
                    #cur.execute('SELECT short_name FROM player_data WHERE ts @@ to_tsquery('english', 'ronaldo')')

                    # display the PostgreSQL database server version
                    #db_version = cur.fetchone()
                    names_list=[]
                    url_list=[]
                    for ieg in dbver:
                        names_list.append(ieg[0])
                        url_list.append(ieg[1])
                    xin=str(dd+" is not match with in our database DID YOU mean")
                   
                # close the communication with the PostgreSQL
                    cursor.close()
                    posts=[{'ch':False,'name':xin,"names":names_list,"url":url_list}]
                    #posts=[{'name':dd+" is not in our database"},{"url":"https://wellesleysocietyofartists.org/wp-content/uploads/2015/11/image-not-found.jpg"},{'actual':act_val},{'predict':predict_val}]
                    return render_template('index.html',posts=posts )
        except (Exception, psycopg2.DatabaseError) as error:
                    print(error)
        

    if ch==True:
        dd=str(dd)
        actual_val,predict_val=test.value_pr(dd)
        label_encoder = preprocessing.LabelEncoder()
        #dd="L. Messi"
        
        df=pd.read_csv("players_22.csv")
        useless_columns = ['short_name','sofifa_id','player_url','long_name','dob','club_joined','club_loaned_from',
                           'nation_position','nation_jersey_number','body_type','real_face','player_face_url',
                           'club_logo_url','club_flag_url','nation_logo_url','nation_flag_url','goalkeeping_speed',
                           'player_tags','nation_team_id','club_jersey_number','ls', 'st', 'rs',
               'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
               'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',
               'rcb', 'rb', 'gk']
        fifa_22 = df.drop(useless_columns, axis=1)
        fifa_22.dropna(inplace=True)
        fifa_22.reset_index(inplace = True)
        obj_typ=fifa_22.select_dtypes('object').columns.to_list()
        OnehotPlayer_positions=[]
        for x in fifa_22['player_positions']:
            stli=x.split(',')
            sl=[st.replace(" ", "") for st in stli]
            OnehotPlayer_positions.append(sl)
        uniPlayer_positions=[]
        for x in OnehotPlayer_positions:
            for y in x:
                if y not in uniPlayer_positions:
                    uniPlayer_positions.append(y)
        Player_positionsArr=[]
        for x in OnehotPlayer_positions:
            row=[]
            for y in uniPlayer_positions:
                if y in x:
                    row.append(1)
                else:
                    row.append(0)
            Player_positionsArr.append(row)

        df_Player_positions=pd.DataFrame(Player_positionsArr,columns=uniPlayer_positions)
        fifa_22['club_name']= label_encoder.fit_transform(fifa_22['club_name'])
        fifa_22['league_name']= label_encoder.fit_transform(fifa_22['league_name'])

        unqclub_position={}
        for x in fifa_22['club_position']:
            if x in unqclub_position:
                unqclub_position[x]+=1
            else:
                unqclub_position[x]=1
        fifa_22['club_position']= label_encoder.fit_transform(fifa_22['club_position'])
        fifa_22['nationality_name']= label_encoder.fit_transform(fifa_22['nationality_name'])

        unqpreferred_foot={}
        for x in fifa_22['preferred_foot']:
            if x in unqpreferred_foot:
                unqpreferred_foot[x]+=1
            else:
                unqpreferred_foot[x]=1
        fifa_22['preferred_foot']= label_encoder.fit_transform(fifa_22['preferred_foot'])
        unqwork_rate={}
        for x in fifa_22['work_rate']:
            if x in unqwork_rate:
                unqwork_rate[x]+=1
            else:
                unqwork_rate[x]=1
        fifa_22['work_rate']= label_encoder.fit_transform(fifa_22['work_rate'])

        Onehotplayer_traits=[]
        for x in fifa_22['player_traits']:
            stli=x.split(',')
            #sl=[st.replace(" ", "") for st in stli]
            Onehotplayer_traits.append(stli)

        uniplayer_traits=[]
        for x in Onehotplayer_traits:
            for y in x:
                if y not in uniplayer_traits:
                    uniplayer_traits.append(y)
        len(uniplayer_traits)

        player_traitsArr=[]
        for x in Onehotplayer_traits:
            row=[]
            for y in uniplayer_traits:
                if y in x:
                    row.append(1)
                else:
                    row.append(0)
            player_traitsArr.append(row)

        df_player_traits=pd.DataFrame(player_traitsArr,columns=uniplayer_traits)

        Unqplayer_traits={}
        for x in df_player_traits.columns:
            Unqplayer_traits[x]=df_player_traits[x].sum()

        fifa_22.drop(['index'], axis=1,inplace=True)
        #player_face=(fifa_22['player_face_url'])
        #short_name=fifa_22['short_name']

        fifa_22=pd.concat([fifa_22,df_Player_positions,df_player_traits], axis=1)
        fifa_22.drop(['player_positions', 'player_traits'], axis=1,inplace=True)
        # fifa_22['value_eurr']=fifa_22['value_eur']

        # fifa_22 = fifa_22.drop(['value_eur'],axis=1)
        fifa_22 = fifa_22.drop(columns=['value_eur'])
        print(fifa_22)
        fifa_22arr=fifa_22.values
        #dd="H. Son"
        y=(ay.loc[ay['short_name']==dd])







        model = None

        # model = open('model.py', 'rb')
        with open('model.pkl', 'rb') as f:
          model = pickle.load(f)



        jc = True
        if (len(ay.loc[ay['short_name']==dd])>0):
          
          print(y.to_dict())
          x = ay.loc[ay['short_name'] == dd].reset_index()
          idx = x.iloc[0, 0]
          url = (x.iloc[0, 4])

          x = x.iloc[0, 1]
          
          print(idx, x)
          # x = ay.loc[ay['short_name']==dd].reset_index().drop(columns=['index']).iloc[0, 0]

        else:
            jc=False
        print(x)
        print(type(model))
        xt_test=np.reshape(fifa_22arr[idx], (1,)+fifa_22arr[idx].shape)
        print(xt_test.shape)

        predi_val=model.predict(xt_test)
        predict_val=float(predi_val[0])
        act_val=x
        #fifa_22=pd.concat([fifa_22,player_face,short_name], axis=1)

        print(act_val)
        print(predict_val)
        
        
        posts=[{'name':dd},{'url':url},{'actual':act_val},{'predict':predict_val}]

    

    # need to get index of the player searched; ali and charan figure it
    # pass the index to int_features
    #final_features = [np.array(int_features)]
    #prediction = model.our_predict(final_features)

    #output = round(prediction[0], 2)

    #result=[actual_text:'Player Cost should be: $ {}'.format(act_val), predict_text :'But Player Cost predicted is: $ {}'.format(predict_val)]
    return render_template('index.html',posts=posts )


if __name__ == "__main__":
        app.run(host='0.0.0.0',port=8080)
    #app.run(debug=True)
