#!/usr/bin/env python
# coding: utf-8

# In[667]:


#!/usr/bin/env python
# coding: utf-8


# In[1]:

# In[668]:


import pandas as pd
import numpy as np
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


# In[669]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

# In[670]:


df=pd.read_csv("players_22.csv")


# In[3]:

# In[671]:


useless_columns = ['short_name','sofifa_id','player_url','long_name','dob','club_joined','club_loaned_from',
                   'nation_position','nation_jersey_number','body_type','real_face','player_face_url',
                   'club_logo_url','club_flag_url','nation_logo_url','nation_flag_url','goalkeeping_speed',
                   'player_tags','nation_team_id','club_jersey_number','ls', 'st', 'rs',
       'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
       'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',
       'rcb', 'rb', 'gk']
fifa_22 = df.drop(useless_columns, axis=1)
print(fifa_22.columns)


# In[ ]:

# In[4]:

# In[672]:


fifa_22


# In[5]:

# In[673]:


fifa_22.dropna(inplace=True)
fifa_22.reset_index(inplace = True)


# In[6]:

# In[674]:


fifa_22.describe()


# In[7]:

# In[675]:


fifa_22.info()


# In[8]:

# In[676]:


obj_typ=fifa_22.select_dtypes('object').columns.to_list()
obj_typ


# In[9]:

# In[677]:


for x in obj_typ:
    print(fifa_22[x].head(5))
    print("-------------")


# In[10]:

# layer_positions

# In[678]:


OnehotPlayer_positions=[]
for x in fifa_22['player_positions']:
    stli=x.split(',')
    sl=[st.replace(" ", "") for st in stli]
    OnehotPlayer_positions.append(sl)


# In[11]:

# nehotPlayer_positions

# In[679]:


uniPlayer_positions=[]
for x in OnehotPlayer_positions:
    for y in x:
        if y not in uniPlayer_positions:
            uniPlayer_positions.append(y)
uniPlayer_positions
len(uniPlayer_positions)


# In[12]:

# In[680]:


Player_positionsArr=[]
for x in OnehotPlayer_positions:
    row=[]
    for y in uniPlayer_positions:
        if y in x:
            row.append(1)
        else:
            row.append(0)
    Player_positionsArr.append(row)
len(Player_positionsArr)


# In[13]:

# In[681]:


df_Player_positions=pd.DataFrame(Player_positionsArr,columns=uniPlayer_positions)
df_Player_positions.head()


# In[14]:

# In[682]:


UnqPlayer_positions={}
for x in df_Player_positions.columns:
    UnqPlayer_positions[x]=df_Player_positions[x].sum()
#UnqPlayer_positions
plt.bar(*zip(*UnqPlayer_positions.items()))
plt.show()


# In[15]:

# lub_name

# In[683]:


unqclub_name={}
for x in fifa_22['club_name']:
    if x in unqclub_name:
        unqclub_name[x]+=1
    else:
        unqclub_name[x]=1
#unqclub_name
f = plt.figure()
f.set_figwidth(70)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("strength of each club")
plt.xlabel('club_name')
plt.ylabel('Count')
plt.bar(*zip(*unqclub_name.items()))
plt.show()


# In[16]:

# In[684]:


len(fifa_22['club_name'].unique())


# In[17]:

# Import label encoder

# In[685]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[686]:


fifa_22['club_name']= label_encoder.fit_transform(fifa_22['club_name'])


# In[687]:


len(fifa_22['club_name'].unique())


# In[18]:

# eague_name

# In[688]:


len(fifa_22['league_name'].unique())
unqleague_name={}
for x in fifa_22['league_name']:
    if x in unqleague_name:
        unqleague_name[x]+=1
    else:
        unqleague_name[x]=1
#unqclub_name
f = plt.figure()
f.set_figwidth(30)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("strength of each leauge")
plt.xlabel('league_name')
plt.ylabel('Count')
plt.bar(*zip(*unqleague_name.items()))
plt.show()


# In[19]:

# In[689]:


fifa_22['league_name']= label_encoder.fit_transform(fifa_22['league_name'])


# In[690]:


len(fifa_22['league_name'].unique())


# In[20]:

# lub_position

# In[691]:


len(fifa_22['club_position'].unique())
unqclub_position={}
for x in fifa_22['club_position']:
    if x in unqclub_position:
        unqclub_position[x]+=1
    else:
        unqclub_position[x]=1
#unqclub_name
f = plt.figure()
f.set_figwidth(30)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("Each club_position")
plt.xlabel('Club_name')
plt.ylabel('Position')
plt.bar(*zip(*unqclub_position.items()))
plt.show()


# In[21]:

# In[692]:


fifa_22['club_position']= label_encoder.fit_transform(fifa_22['club_position'])


# In[693]:


len(fifa_22['club_position'].unique())


# In[22]:

# ationality_name

# In[694]:


len(fifa_22['nationality_name'].unique())
unqnationality_name={}
for x in fifa_22['nationality_name']:
    if x in unqnationality_name:
        unqnationality_name[x]+=1
    else:
        unqnationality_name[x]=1
#unqclub_name
f = plt.figure()
f.set_figwidth(40)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("Players from each nation")
plt.xlabel('Countries')
plt.ylabel('Count')
plt.bar(*zip(*unqnationality_name.items()))
plt.show()


# In[23]:

# In[695]:


fifa_22['nationality_name']= label_encoder.fit_transform(fifa_22['nationality_name'])


# In[696]:


len(fifa_22['nationality_name'].unique())


# In[24]:

# referred_foot

# In[697]:


len(fifa_22['preferred_foot'].unique())
unqpreferred_foot={}
for x in fifa_22['preferred_foot']:
    if x in unqpreferred_foot:
        unqpreferred_foot[x]+=1
    else:
        unqpreferred_foot[x]=1
#unqclub_name
plt.title("Prefered Foot")
plt.xlabel('Foot')
plt.ylabel('Count')
plt.bar(*zip(*unqpreferred_foot.items()))
plt.show()


# In[25]:

# In[698]:


fifa_22['preferred_foot']= label_encoder.fit_transform(fifa_22['preferred_foot'])


# In[699]:


len(fifa_22['preferred_foot'].unique())


# In[26]:

# ork_rate

# In[700]:


len(fifa_22['work_rate'].unique())
unqwork_rate={}
for x in fifa_22['work_rate']:
    if x in unqwork_rate:
        unqwork_rate[x]+=1
    else:
        unqwork_rate[x]=1
#unqclub_name
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("Count Of Workrate")
plt.xlabel('Type')
plt.ylabel('Count')
plt.bar(*zip(*unqwork_rate.items()))
plt.show()


# In[27]:

# In[701]:


fifa_22['work_rate']= label_encoder.fit_transform(fifa_22['work_rate'])


# In[702]:


len(fifa_22['work_rate'].unique())


# In[28]:

# layer_traits

# In[703]:


Onehotplayer_traits=[]
for x in fifa_22['player_traits']:
    stli=x.split(',')
    #sl=[st.replace(" ", "") for st in stli]
    Onehotplayer_traits.append(stli)


# In[29]:

# In[704]:


uniplayer_traits=[]
for x in Onehotplayer_traits:
    for y in x:
        if y not in uniplayer_traits:
            uniplayer_traits.append(y)
len(uniplayer_traits)


# In[30]:

# In[705]:


player_traitsArr=[]
for x in Onehotplayer_traits:
    row=[]
    for y in uniplayer_traits:
        if y in x:
            row.append(1)
        else:
            row.append(0)
    player_traitsArr.append(row)
len(player_traitsArr)


# In[31]:

# In[706]:


df_player_traits=pd.DataFrame(player_traitsArr,columns=uniplayer_traits)
df_player_traits


# In[32]:

# In[707]:


Unqplayer_traits={}
for x in df_player_traits.columns:
    Unqplayer_traits[x]=df_player_traits[x].sum()
Unqplayer_traits
f = plt.figure()
f.set_figwidth(40)
f.set_figheight(5)
plt.xticks(rotation = 90)
plt.title("Player_traits")
plt.xlabel('Type')
plt.ylabel('Count')
plt.bar(*zip(*Unqplayer_traits.items()))
plt.show()


# In[33]:

# In[708]:


fifa_22.drop(['index'], axis=1,inplace=True)
fifa_22


# In[709]:


#player_face=(fifa_22['player_face_url'])
#short_name=fifa_22['short_name']


# In[34]:

# In[710]:





# In[711]:


fifa_22=pd.concat([fifa_22,df_Player_positions,df_player_traits], axis=1)


# In[35]:

# In[712]:


fifa_22.drop(['player_positions', 'player_traits'], axis=1,inplace=True)
fifa_22


# In[36]:

# In[713]:


fifa_22['value_eurr']=fifa_22['value_eur']
print(len(fifa_22.columns))
fifa_22.drop(['value_eur'],axis=1,inplace=True)
print(len(fifa_22.columns))


# In[37]:

# In[714]:


fifa_22


# In[38]:

# In[715]:


fifa_22.info()


# In[39]:

# In[716]:


fifa_22arr=fifa_22.values


# In[40]:

# In[717]:


fifa_22arr[:,-1]


# In[41]:

# In[718]:


from sklearn import preprocessing


# In[42]:

# In[719]:


XX=fifa_22arr[:,:-1]


# In[43]:

# In[720]:


yy=fifa_22arr[:,-1]


# In[44]:

# In[721]:


X_train, X_test, Y_train, Y_test = train_test_split(XX, yy)


# In[45]:

# In[722]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)
y_pred=reg.predict(X_test)


# In[723]:


X_test[1]


# In[46]:

# In[724]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("MAE",mean_absolute_error(y_pred,Y_test))
print("R2",r2_score(y_pred,Y_test))
print("MAE",mean_absolute_error(y_pred,Y_test))
print("MSE",mean_squared_error(Y_test,y_pred))
from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(Y_test,y_pred))


# In[47]:

# In[725]:


diff_df=pd.DataFrame({"Act":Y_test,"Pred":y_pred,"diff":Y_test-y_pred})


# In[48]:

# In[726]:


diff_df


# In[49]:

# In[727]:


plt.figure(figsize=(15,10))
plt.scatter(y_pred,Y_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Actual vs Predicted")


# In[ ]:

# In[728]:


diff_df


# iff_df.to_csv('C:\\Users\\alsag\\OneDrive\\Desktop\\s1.csv')

# In[729]:


#diff_df=pd.concat([diff_df,short_name], axis=1)


# In[730]:


diff_df=diff_df.dropna()


# In[731]:


diff_df


# In[754]:


#diff_df.to_csv('values.csv',index=False)


# In[734]:
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


def our_predict(y_xtest):
    
    #print(X_test[indx].rshape(-1, no_of_features))
    xt_test=np.reshape(y_xtest[inx], (1,)+y_xtest[inx].shape)
    xcd=reg.predict(xt_test)
    print(xcd)
    return  xcd


# In[735]:



