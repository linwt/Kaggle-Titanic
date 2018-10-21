# 模块导入
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
warnings.filterwarnings('ignore')

# 数据导入
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 了解数据
train.head()
test.head()
train.info()
test.info()
train.describe()
test.describe()

# 缺失值填充
train.loc[pd.isnull(train.Embarked), ['Embarked']] = train['Embarked'].mode()
train.loc[pd.isnull(train.Age), ['Age']] = train['Age'].mean()
test.loc[pd.isnull(test.Fare),['Fare']] = train['Fare'].mean()
test.loc[pd.isnull(train.Age), ['Age']] = train['Age'].mean()

# 数据分割，拆分成训练集和验证集
label = train['Survived']
train.drop('Survived', axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(train, label, test_size=0.3, random_state=1)
X_train['Survived'] = Y_train
X_test['Survived'] = Y_test




#**************** 特征工程start ********************

##################################################
# Sex
##################################################

# 画图分析训练集和验证集的Sex特征。
# 训练集和验证集图形相似且特征区别明显则为好特征。
# 若特征有价值，则在训练集和测试集生成或保留该特征，再进行其他操作。
fig, axis = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot('Sex', 'Survived', data=X_train, ax=axis[0])
sns.barplot('Sex', 'Survived', data=X_test, ax=axis[1])

# Sex归类
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# 对训练集和测试集的Sex特征进行one_hot编码
train = pd.get_dummies(data= train,columns=['Sex'])
test = pd.get_dummies(data= test,columns=['Sex'])


##################################################
# Name
##################################################

# 提取尊称并统计存活率
X_train['Name_Title'] = X_train['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split()[0])
X_test['Name_Title'] = X_test['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split()[0])
X_train.groupby('Name_Title')['Survived'].count()

# 画图分析Name_Title特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.barplot('Name_Title', 'Survived', data=X_train.sort_values('Name_Title'), ax=axis[0])
sns.barplot('Name_Title', 'Survived', data=X_test.sort_values('Name_Title'), ax=axis[1])

# Name_Title归类
def Name_Title_Code(x):
    if x == 'Mr.':
        return 1
    if (x == 'Mrs.') or (x=='Ms.') or (x=='Lady.') or (x == 'Mlle.') or (x =='Mme'):
        return 2
    if x == 'Miss':
        return 3
    if x == 'Rev.':
        return 4
    return 5

train['Name_Title'] = train['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split()[0]).apply(lambda x : Name_Title_Code(x))
test['Name_Title'] = test['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split()[0]).apply(lambda x : Name_Title_Code(x))

# 训练集和测试集生成Name_Title特征，并进行one_hot编码
train = pd.get_dummies(columns = ['Name_Title'], data = train)
test = pd.get_dummies(columns = ['Name_Title'], data = test)

# 提取名字长度
X_train['Name_len'] = X_train['Name'].apply(lambda x : len(x))
X_test['Name_len'] = X_test['Name'].apply(lambda x : len(x))

# 画图分析训练集和验证集的Name_len特征
fig, axis = plt.subplots(1, 2, figsize=(20, 10))
sns.barplot('Name_len', 'Survived', data = X_train, ax = axis[0])
sns.barplot('Name_len', 'Survived', data = X_test, ax = axis[1])

# 训练集和测试集生成Name_len特征
train['Name_len'] = train['Name'].apply(lambda x : len(x))
test['Name_len'] = test['Name'].apply(lambda x : len(x))


##################################################
# Ticket
##################################################

# 提取票的首字符，并统计存活率
X_train['Ticket_First_Letter'] = X_train['Ticket'].apply(lambda x : x[0])
X_test['Ticket_First_Letter'] = X_test['Ticket'].apply(lambda x : x[0])
X_train.groupby('Ticket_First_Letter')['Survived'].count()

# 画图分析Ticket_First_Letter特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.barplot('Ticket_First_Letter', 'Survived', data = X_train.sort_values('Ticket_First_Letter'), ax = axis[0])
sns.barplot('Ticket_First_Letter', 'Survived', data = X_test.sort_values('Ticket_First_Letter'), ax = axis[1])

# Ticket_First_Letter归类
def Ticket_First_Letter_Code(x):
    if (x == '1'):
        return 1
    if x == '3':
        return 2
    if x == '4':
        return 3
    if x == 'C':
        return 4
    if x == 'S':
        return 5
    if x == 'P':
        return 6
    if x == '6':
        return 7
    if x == '7':
        return 8
    if x == 'A':
        return 9
    if x == 'W':
        return 10
    return 11

train['Ticket_First_Letter'] = train['Ticket'].apply(lambda x : x[0]).apply(lambda x : Ticket_First_Letter_Code(x))
test['Ticket_First_Letter'] = test['Ticket'].apply(lambda x : x[0]).apply(lambda x : Ticket_First_Letter_Code(x))

# 训练集和测试集生成Ticket_First_Letter特征,并进行one_hot编码
train = pd.get_dummies(columns = ['Ticket_First_Letter'], data = train)
test = pd.get_dummies(columns = ['Ticket_First_Letter'], data = test)


##################################################
# Cabin
##################################################

# 缺失值填充，提取船舱首字符，统计存活率
X_train['Cabin_First_Letter'] = X_train['Cabin'].fillna('X').apply(lambda x : x[0])
X_test['Cabin_First_Letter'] = X_test['Cabin'].fillna('X').apply(lambda x : x[0])
X_train.groupby('Cabin_First_Letter')['Survived'].count()

# 画图分析Cabin_First_Letter特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.barplot('Cabin_First_Letter', 'Survived', data = X_train.sort_values('Cabin_First_Letter'), ax = axis[0])
sns.barplot('Cabin_First_Letter', 'Survived', data = X_test.sort_values('Cabin_First_Letter'), ax = axis[1])

# Cabin_First_Letter归类
def Cabin_First_Letter_Code(x):
    if x == 'X':
        return 1
    if x == 'B':
        return 2
    if x == 'C':
        return 3
    if x == 'D':
        return 4
    return 5

train['Cabin_First_Letter'] = train['Cabin'].fillna('X').apply(lambda x : x[0]).apply(Cabin_First_Letter_Code)
test['Cabin_First_Letter'] = test['Cabin'].fillna('X').apply(lambda x : x[0]).apply(Cabin_First_Letter_Code)

# 训练集和测试集生成Cabin_First_Letter特征,并进行one_hot编码
train = pd.get_dummies(columns = ['Cabin_First_Letter'], data = train)
test = pd.get_dummies(columns = ['Cabin_First_Letter'], data = test)


##################################################
# Embarked
##################################################

# 画图分析Embarked特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.barplot('Embarked', 'Survived', data = X_train.sort_values('Embarked'), ax = axis[0])
sns.barplot('Embarked', 'Survived', data = X_test.sort_values('Embarked'), ax = axis[1])

# 训练集和测试集对Embarked特征进行one_hot编码
train = pd.get_dummies(columns = ['Embarked'], data = train)
test = pd.get_dummies(columns = ['Embarked'], data = test)


##################################################
# SibSp + Parch
##################################################

# 相加得到家庭人数
X_train['Fam_Size'] = X_train['SibSp'] + X_train['Parch']
X_test['Fam_Size'] = X_test['SibSp'] + X_test['Parch']

# 画图分析Fam_Size特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.barplot('Fam_Size', 'Survived', data = X_train.sort_values('Parch'), ax = axis[0])
sns.barplot('Fam_Size', 'Survived', data = X_test.sort_values('Parch'), ax = axis[1])

# Fam_Size归类
def Family_feature(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0, 'Single', np.where((i['SibSp']+i['Parch']) <= 3,'Middle', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test

train, test  = Family_feature(train, test)

# 训练集和测试集对Fam_Size特征进行one_hot编码
train = pd.get_dummies(columns = ['Fam_Size'], data = train)
test =  pd.get_dummies(columns = ['Fam_Size'], data = test)


##################################################
# Pclass
##################################################

# 画图分析Pclass特征
fig, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.barplot('Pclass', 'Survived', data = X_train, ax = axis[0])
sns.barplot('Pclass', 'Survived', data = X_test, ax = axis[1])

# 训练集和测试集对Pclass特征进行one_hot编码
train = pd.get_dummies(columns = ['Pclass'], data = train)
test = pd.get_dummies(columns = ['Pclass'], data = test)


##################################################
# Age
##################################################

# 画图分析Age特征
fig, axis = plt.subplots(1,2,figsize=(15,5))
sns.distplot(X_train[X_train.Survived==1]['Age'].dropna().values, bins=range(0, 81, 6),color='red', ax=axis[0])
sns.distplot(X_train[X_train.Survived==0]['Age'].dropna().values, bins=range(0, 81, 6),color = 'blue', ax=axis[0])
sns.distplot(X_test[X_test.Survived==1]['Age'].dropna().values, bins=range(0, 81, 6),color='red', ax=axis[1])
sns.distplot(X_test[X_test.Survived==0]['Age'].dropna().values, bins=range(0, 81, 6),color = 'blue', ax=axis[1])

# Fam_Size归类
def Age_feature(x):
    if x <= 5:
        return 'Small'
    if x >= 15 and x <= 25:
        return 'Middle'
    if x >= 65:
        return 'Old'

train['Age_level'] = train['Age'].apply(Age_feature)
test['Age_level'] = test['Age'].apply(Age_feature)

# 训练集和测试集对Age特征进行one_hot编码
train = pd.get_dummies(columns = ['Age_level'], data = train)
test = pd.get_dummies(columns = ['Age_level'], data = test)


##################################################
# Fare
##################################################

# 分布范围大，做log处理
X_train['Fare'] = (X_train['Fare'] + 1).apply(np.log)
X_test['Fare'] = (X_test['Fare'] + 1).apply(np.log)

# 画图分析Fare特征
fig, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.distplot(X_train[X_train.Survived == 1]['Fare'].dropna().values, bins=range(0, 10, 1), color='red', ax=axis[0])
sns.distplot(X_train[X_train.Survived == 0]['Fare'].dropna().values, bins=range(0, 10, 1), color='blue', ax=axis[0])
sns.distplot(X_test[X_test.Survived == 1]['Fare'].dropna().values, bins=range(0, 10, 1), color='red', ax=axis[1])
sns.distplot(X_test[X_test.Survived == 0]['Fare'].dropna().values, bins=range(0, 10, 1), color='blue', ax=axis[1])

# 训练集和测试集做log处理
train['Fare'] = (train['Fare'] + 1).apply(np.log)
test['Fare'] = (test['Fare'] + 1).apply(np.log)

# Fare归类
def Fare_Code(x):
    if x <= 2:
        return '0_2'
    if x > 2 and x <= 3:
        return '2_3'
    if x > 3 and x <= 4:
        return '3_4'
    if x > 4 and x <= 5:
        return '4_5'
    if x > 5:
        return '5_n'

train['Fare'] = train['Fare'].apply(Fare_Code)
test['Fare'] = test['Fare'].apply(Fare_Code)

# 训练集和测试集对Fare特征进行one_hot编码
train = pd.get_dummies(columns = ['Fare'], data = train)
test = pd.get_dummies(columns = ['Fare'], data = test)

##################################################

# 特征选择。模型一般只能处理数值型数据
train.info()
useless = ['Ticket','PassengerId','Name','Cabin','Age']
train.drop(useless, axis = 1, inplace = True)
test.drop(useless, axis = 1, inplace = True)

# 查看处理后的数据
pd.set_option('display.max_columns',50)
train.head()

#**************** 特征工程end ********************





# 新建训练集和验证集
X_train_ = train.loc[X_train.index]
X_test_ = train.loc[X_test.index]
Y_train_ = label.loc[X_train.index]
Y_test_ = label.loc[X_test.index]

# 共同列提取与对齐
X_test_ = X_test_[X_train_.columns]
test = test[train.columns]

# 建立模型
rf = RandomForestClassifier(criterion='gini',
                             n_estimators=700,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             random_state=10,
                             n_jobs=-1)
# 训练模型，测试分数
rf.fit(X_train_,Y_train_)
rf.score(X_test_,Y_test_)

# 训练模型
rf.fit(train,label)

# 模型特征重要性检测
pd.concat(
    (
        pd.DataFrame(train.columns, columns=['variable']),
        pd.DataFrame(rf.feature_importances_, columns=['importance'])
    ), axis=1
).sort_values(by='importance', ascending=False)[:20]

# 预测并生成预测结果文件
submit = pd.read_csv('../input/gender_submission.csv')
submit.set_index('PassengerId', inplace=True)
res_rf = rf.predict(test)
submit['Survived'] = res_rf
submit.to_csv('submit.csv')
