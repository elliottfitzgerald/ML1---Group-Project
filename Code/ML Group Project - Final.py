import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import weightstats as stests
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# SUBROUTINE 1
# use categorical non-likert features to determine correlation with overall satisfaction in order to
# gauge how helpful they will be in a model
def category(dataset, feature, target):
    labels = dataset[feature].unique().tolist()
    data_labels = np.array(dataset[feature])
    data_targets = np.array(dataset[target])
    list_names = []
    lists = {}
    means = dataset[[feature, target]].groupby([feature], as_index=False).mean()
    for item in labels:
        list_names.append(item)
        item_targets = [data_targets[x] for x in range(0, len(data_targets)-1) if data_labels[x] == item]
        lists[item] = item_targets
        item_index = means.loc[means[feature] == item].index[0]
        print("Sample is", round((len(item_targets)/len(data_targets)*100)), "% ", item, "with a mean satisfaction of", str(np.round(means.iloc[item_index, 1], 2)), ".")
    if len(labels) == 2:
        ztest, pval = stests.ztest(x1=lists[list_names[0]], x2=lists[list_names[1]], value=0, alternative='two-sided')
        # print(float(pval))
        if pval < 0.05:
            print("{}: {} and {} groups have significantly different satisfaction levels. (p-value is {})".format(feature, list_names[0], list_names[1], str(pval)))
        else:
            print("{}: %s and %s do not have significantly different satisfaction levels. (p-value is {})".format(feature, list_names[0], list_names[1], str(pval)))
    else:
        pass

# SUBROUTINE 2
# use ordinal/continuous non-likert features to determine correlation with overall satisfaction in order to
# gauge how helpful they will be in a model
def ordinal(dataset, feature, target):
    corr_data = dataset[[feature, target]]
    corr_matrix = corr_data.corr()
    print('The correlation coefficient between {} and satisfaction is {}.'.format(feature, str(round(corr_matrix.iloc[0,1], 2))))

# SUBROUTINE 3
# use likert questions in a function to calculate correlation with overall satisfaction
def likert(dataset, columns):
    options = [1, 2, 3, 4, 5]
    cols = columns.keys()
    question = []
    response = []
    mean = []
    for c in cols:
        corr = dataset[[c, 'satisfaction_v2']].groupby([c], as_index=False).mean()
        for n in options:
            n_index = corr.loc[corr[c] == n].index[0]
            question.append(columns[c])
            response.append(n)
            mean.append(corr.iloc[n_index, 1])
    # print(question)
    # print(response)
    # print(mean)
    final_df = pd.DataFrame(
        {'question': question,
         'response': response,
         'mean': mean
         })
    return final_df

# SUBROUTINE 4
# create function to run through different n's to see which would be best for PCA
def pca_opt(n_list):
    pca_results_dict = {}
    for i in n_list:
        pca = PCA(n_components=i)
        X_train_pca = pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        pca_results_dict[i] = np.round(np.sum(pca.explained_variance_ratio_), 2)
    # dataframe = pd.DataFrame.from_dict(pca_results_dict)
    return pca_results_dict






####################################################
# PREPROCESSING
####################################################

# read in dataset
sat_orig = pd.read_csv('/Users/megelliottfitzgerald/Documents/satisfaction.csv', header=0)

# count rows/columns
print("sat_orig shape: ", sat_orig.shape)
# sat_orig shape:  (129880, 24)

# look at data
print("sat_orig head", sat_orig.head(3))

# identify any nulls
print(sat_orig.isna().sum(axis=0))

# create working copy of df without nulls
sat = sat_orig.dropna()
print("sat shape:", sat.shape)
# sat shape: (129487, 24)

# get list of unique values in each column
satcols = list(sat.columns)
for column in satcols:
  print(column, sat[column].unique())

# change satisfaction to binary
sat['satisfaction_v2'].replace({"satisfied": 1, "neutral or dissatisfied": 0}, inplace=True)
sat['satisfaction_v2'] = sat['satisfaction_v2'].astype(int)
sat['satisfaction_v2'].unique()

# format labels for charts
sat['Customer Type'].replace({"Loyal Customer": "Loyal", "disloyal Customer": "Disloyal"}, inplace=True)
sat['Type of Travel'].replace({"Business travel": "Business", "Personal Travel": "Personal"}, inplace=True)
sat['Class_v2'] = sat['Class']
sat['Class_v2'].replace({"Eco Plus": "Eco"}, inplace=True)






####################################################
# EDA
####################################################

# examine each feature individually

category(sat, "Gender", "satisfaction_v2")
# women seem to be more satisfied than men, by a relatively large margin
# it looks like gender may be a strong predictor of satisfaction.

category(sat, "Customer Type", "satisfaction_v2")
# loyal customers are by far more satisfied than disloyal customers
# it looks like customer type may be a strong predictor of satisfaction.

category(sat, "Type of Travel", "satisfaction_v2")
# people flying for business reasons seem to be a bit more satisfied than those flying for personal reasons
# it looks like customer type might be a predictor of satisfaction.

category(sat, "Class_v2", "satisfaction_v2")
# people flying in business class seem to be a bit more satisfied than those flying economy
# it looks like class might be a predictor of satisfaction.

# Set up a grid to plot satisfaction probability against these variables
g = sns.PairGrid(sat, y_vars="satisfaction_v2", x_vars=["Gender", "Customer Type", "Type of Travel", "Class"], height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
plt.show()

# examine the ordinal/continuous features
ordinal(sat, 'Departure Delay in Minutes', 'satisfaction_v2')
ordinal(sat, 'Arrival Delay in Minutes', 'satisfaction_v2')
# neither seem to be strongly correlated with satisfaction outcomes
# both are strongly skewed datasets so perhaps some thought could be applied to
# transformations (log, etc.) that would compensate. preliminary applications
# of logscaling do not seem to help (unless you logscale the counts instead of
# the delay in minutes. even so, the histograms don't look all that different)

ordinal(sat, 'Flight Distance', 'satisfaction_v2')
# Flight Distance does not seem to correlate with satisfaction at all

splot1 = sns.displot(sat, x="Departure Delay in Minutes", row="satisfaction_v2", binwidth=60, height=3, facet_kws=dict(margin_titles=True))
splot1.set(yscale="log")
plt.show()

splot2 = sns.displot(sat, x="Arrival Delay in Minutes", row="satisfaction_v2", binwidth=60, height=3)
splot2.set(yscale="log")
plt.show()

splot3 = sns.displot(sat, x="Flight Distance", row="satisfaction_v2", binwidth=60, height=3)
plt.show()

ordinal(sat, 'Age', 'satisfaction_v2')
# Age does not seem to correlate with satisfaction at all

splot4 = sns.displot(sat, x="Age", row="satisfaction_v2", binwidth=5, height=3)
plt.show()

# boxplot
splot5 = sns.boxplot(x="satisfaction_v2", y="Age", hue="satisfaction_v2", palette=["m", "g"], data=sat)
plt.show()

# look at age and gender together
splot6 = sns.displot(sat, x="Age", row="satisfaction_v2", col = 'Gender', binwidth=5, height=3)
plt.show()

# look at age and class together
splot7 = sns.displot(sat, x="Age", row="satisfaction_v2", col = 'Class', binwidth=5, height=3)
plt.show()

# look at age and type of travel together
splot8 = sns.displot(sat, x="Age", row="satisfaction_v2", col = 'Type of Travel', binwidth=5, height=3, facet_kws=dict(margin_titles=True))
plt.show()

# look at age and customer type together
splot9 = sns.displot(sat, x="Age", row="satisfaction_v2", col = 'Customer Type', binwidth=5, height=3)
plt.show()
# age really doesn't seem to play an important role in satisfaction
# even when combined with other features

# look at likert-type features
columns = {'Online boarding' : 'online board'
      , 'Cleanliness' : 'clean'
      , 'Checkin service' : 'checkin'
      , 'Baggage handling' : 'baggage'
      , 'Departure/Arrival time convenient' : 'time convenient'
      , 'Leg room service' : 'leg room'
      , 'On-board service' : 'on-board service'
      , 'Ease of Online booking' : 'ease of booking'
      , 'Online support' : 'online support'
      , 'Inflight wifi service' : 'wifi'
      , 'Food and drink' : 'food/drink'
      , 'Gate location' : 'gate loc'
      , 'Inflight entertainment' : 'flight fun'
      , 'Seat comfort' : 'comfort'}

likert_df = likert(sat, columns)

# Initialize a grid of plots with an Axes for each question
grid2 = sns.FacetGrid(likert_df, col="question", hue="question", palette="tab20c",
                     col_wrap=4, height=3)
grid2.map(plt.axhline, y=0, ls=":", c=".5")                                         # Draw a horizontal line to show the starting point
grid2.map(plt.plot, "response", "mean", marker="o")                                 # Draw a line plot to show the trajectory of each random walk
grid2.set(xticks=np.arange(5), yticks=[0, 1], xlim=(0.5, 5.5), ylim=(0, 1.5))       # Adjust the tick positions and labels
grid2.fig.tight_layout(w_pad=1)                                                     # Adjust the arrangement of the plots
plt.show()

# from these results it seems like if you can make your customer as comfortable as
# possible during the flight you may be able to increase their overall satisfaction
# greatly - even more so than if you increased their satisfaction in airport or
# administrative features that aid in pre- and post- flight chores

# I wonder if total satisfaction score (all these columns added together, either
# as a sum or average) could be used as a single input to make a model perform
# more quickly. let's see if we get any kind of correlation

#convert columns to numeric
satcols = ['Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']
sat_num = sat[satcols]
sat_num = sat_num.apply(pd.to_numeric)
sat_num['total_sat'] = sat_num.sum(axis=1)
sat_num['satisfaction_v2'] = sat['satisfaction_v2']

# look at the relationship between flight distance and satisfaction
satdf = sat_num[['total_sat', 'satisfaction_v2']]
print(satdf.corr())
# not a great correlation. not too strong. we'll hold off on using this for now

# since flight enjoyment satisfaction seemed to be the most correlated with
# satisfaction, maybe in the future we group the sat questions by topic
# (sat based on convenience vs. sat based on comfort/enjoyment, etc.)

# I'm wondering if there are people who are just "straight-5's", "straight-3's"
# or "straight-0's" who are skewing the results. some people just rush through
# surveys and mark the max/min/middle values without really considering the
# specific question seriously. something to look into in the future
# possibly generate heatmap or some other way to figure out the straight-5/3/0's
# sat[(sat['A']>0) & (sat['B']>0) & (sat['C']>0)].count()






####################################################
# PREP DATA FOR MODEL
####################################################

y = sat['satisfaction_v2']
X = sat.drop(columns=['satisfaction_v2', 'Class_v2'])
X.columns
# encode x labels
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

# separate test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)






####################################################
# FEATURE REDUCTION
####################################################

# CHOOSE PCA COMPONENTS N
pca_results = pca_opt(range(0,20))
pca_df = pd.DataFrame.from_dict(pca_results, orient = 'index', columns = ['explained_variance'])
e = sns.lineplot(x=pca_df.index.to_list(), y=pca_df["explained_variance"])
e.set(xlabel='PCA n Components', ylabel='Explained Variance')
plt.show()

# going to go with n = 12 to keep the explained variance around 80-85% to avoid overfitting
pca = PCA(n_components=12)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
final_explained_variance = np.round(np.sum(pca.explained_variance_ratio_), 2)






####################################################
# CHOOSE BACKPROP METHOD
####################################################

# adam
adam_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver = 'adam')
adam_model.fit(X_train, y_train)
adam_predictions = adam_model.predict(X_test)
adam_confusion = confusion_matrix(y_test, adam_predictions)
adam_class = classification_report(y_test, adam_predictions)
conf = plot_confusion_matrix(adam_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (10/10/10) w/ ADAM')
plt.show()
print(adam_class)

# lbfgs
lbfgs_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver = 'lbfgs')
lbfgs_model.fit(X_train, y_train)
lbfgs_predictions = lbfgs_model.predict(X_test)
lbfgs_confusion = (confusion_matrix(y_test, lbfgs_predictions))
lbfgs_class = classification_report(y_test, lbfgs_predictions)
conf = plot_confusion_matrix(lbfgs_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (10/10/10) w/ L-BFGS')
plt.show()
print(lbfgs_class)

# sgd
sgd_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver = 'sgd')
sgd_model.fit(X_train, y_train)
sgd_predictions = sgd_model.predict(X_test)
sgd_confusion = (confusion_matrix(y_test, sgd_predictions))
sgd_class = (classification_report(y_test, sgd_predictions))
conf = plot_confusion_matrix(sgd_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (10/10/10) w/ SGD')
plt.show()
print(sgd_class)

print(adam_confusion)
print(lbfgs_confusion)
print(sgd_confusion)

# adam and lbfgs perform the best
# we'll go with lbfgs because it's marginally better






####################################################
# CHOOSE NETWORK ARCHITECTURE
####################################################

# SCORES TO BEAT
#            precision    recall  f1-score   support
#            0       0.91      0.91      0.91     11725
#            1       0.92      0.93      0.92     14173
#     accuracy                           0.92     25898
#    macro avg       0.92      0.92      0.92     25898
# weighted avg       0.92      0.92      0.92     25898

# TRY A FOURTH LAYER
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=1000, solver = 'lbfgs')
mlp_model.fit(X_train, y_train)
layer_4_predictions = mlp_model.predict(X_test)
layer_4_cm = confusion_matrix(y_test, layer_4_predictions)
layer_4_cr = classification_report(y_test, layer_4_predictions)
print(layer_4_cr)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('6 Layer MLP (10/10/10/10) w/ L-BFGS')
plt.show()
# roughly the same performance but takes much longer. we'll stick with 3 layers

# INCREASE NEURONS IN SECOND LAYER
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 15, 10), max_iter=1000, solver = 'lbfgs')
mlp_model.fit(X_train, y_train)
predictions_10_15_10 = mlp_model.predict(X_test)
cm_10_15_10 = confusion_matrix(y_test, predictions_10_15_10)
cr_10_15_10 = classification_report(y_test, predictions_10_15_10)
print(cr_10_15_10)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (10/15/10) w/ L-BFGS')
plt.show()
# roughly the same performance

# INCREASE NEURONS IN ALL LAYERS
mlp_model = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=1000, solver = 'lbfgs')
mlp_model.fit(X_train, y_train)
predictions_15_15_15 = mlp_model.predict(X_test)
cm_15_15_15 = confusion_matrix(y_test, predictions_15_15_15)
cr_15_15_15 = classification_report(y_test, predictions_15_15_15)
print(cr_15_15_15)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (15/15/15) w/ L-BFGS')
plt.show()

# gained a couple points across all metrics
# final results
#            precision    recall  f1-score   support
#            0       0.91      0.92      0.92     11739
#            1       0.94      0.93      0.93     14159
#     accuracy                           0.93     25898
#    macro avg       0.92      0.93      0.93     25898
# weighted avg       0.93      0.93      0.93     25898

mlp_model = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=1000, solver = 'lbfgs')
mlp_model.fit(X_train, y_train)
predictions_20 = mlp_model.predict(X_test)
cm_20 = confusion_matrix(y_test, predictions_20)
cr_20 = classification_report(y_test, predictions_20)
print(cr_20)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (20/20/20) w/ L-BFGS')
plt.show()

mlp_model = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=1000, solver = 'lbfgs', activation = 'logistic')
mlp_model.fit(X_train, y_train)
predictions_15_15_15_log = mlp_model.predict(X_test)
cm_15_15_15_log = confusion_matrix(y_test, predictions_15_15_15_log)
cr_15_15_15_log = classification_report(y_test, predictions_15_15_15_log)
print(cr_15_15_15_log)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (15/15/15) w/ L-BFGS w/ Logistic Activation Function')
plt.show()

mlp_model = MLPClassifier(hidden_layer_sizes=(15, 15), max_iter=1000, solver = 'sgd', activation = 'logistic', alpha=0.01, momentum=0.95)
mlp_model.fit(X_train, y_train)
predictions_15_15 = mlp_model.predict(X_test)
cm_15_15 = confusion_matrix(y_test, predictions_15_15)
cr_15_15 = classification_report(y_test, predictions_15_15)
print(cr_15_15)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('4 Layer MLP (15/15) w/ SGD w/ Logistic AF, 0.01 ALPHA/0.95 MOM')
plt.show()

mlp_model = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=1000, solver = 'sgd', alpha=0.01, momentum=0.95)
mlp_model.fit(X_train, y_train)
predictions_15_15_15_new = mlp_model.predict(X_test)
cm_15_15_15_new = confusion_matrix(y_test, predictions_15_15_15_new)
cr_15_15_15_new = classification_report(y_test, predictions_15_15_15_new)
print(cr_15_15_15_new)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (15/15/15) w/ SGD w/ Logistic AF, 0.01 ALPHA/0.95 MOM')
plt.show()






####################################################
# CROSS VALIDATE FINAL MODEL
####################################################

mlp_model = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=1000, solver = 'lbfgs', activation = 'logistic')
mlp_model.fit(X_train, y_train)
predictions_15_15_15 = mlp_model.predict(X_test)
cm_15_15_15 = confusion_matrix(y_test, predictions_15_15_15)
cr_15_15_15 = classification_report(y_test, predictions_15_15_15)
print(cr_15_15_15)
conf = plot_confusion_matrix(mlp_model, X_test, y_test, cmap = plt.cm.Blues)
conf.ax_.set_title('5 Layer MLP (15/15/15) w/ L-BFGS')
plt.show()

cross_val_score(mlp_model,X_train,y_train,cv=6)






####################################################
# USE GRIDSEARCH TO SYSTEMATIZE MODEL PERFORMANCE OPTIMIZATION
####################################################

from sklearn.model_selection import GridSearchCV

mlp_gs = MLPClassifier(max_iter=1000)
parameter_space = {
    'hidden_layer_sizes': [(15,),(20,)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'momentum': [0.9, 0.95],
    'learning_rate': ['constant','adaptive'],}
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)

# using a maximum of 100 iterations fails to produce convergence
# in every version of the model, and it's likely too performance
# intensive to run with an increased max_iter unless we dramatically
# limit the number of parameter options






####################################################
# COMPARE PERFORMANCE TO CLASSICAL MODEL (RANDOM FOREST)
####################################################

# now try a classical model - random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(36)
rfcl = RandomForestClassifier()
rfcl.fit(X_train,y_train)
rfcl.score(X_test,y_test)

rf_y_predictions = rfcl.predict(X_test)
accuracy_score(y_test,rf_y_predictions)
# 0.9159008417638428

# # now i'll recheck it without PCA
# # separate test and train
# X.columns
# X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size = 0.20)
# np.random.seed(36)
# rfcl_dt = RandomForestClassifier()
# rfcl_dt.fit(X_train_dt,y_train_dt)
# rfcl_dt.score(X_test_dt,y_test_dt)
#
# rf_y_predictions_dt = rfcl.predict(X_test_dt)
# accuracy_score(y_test,rf_y_predictions_dt)
# # 0.9159008417638428

