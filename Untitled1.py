
# coding: utf-8

# # Project 7:
# # Machine Learning

# In[1]:




import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier






# #### In the cell below I place most of the feature in the feature_list and  call the data file and place it in  data_dict. I locate the total
# #### number of people in the list and and the number feature in the dictionary.  Then i want to see how many NaN are in the salary,
# #### bonus , exercised_stock_options  and total_stock_value. 

# In[2]:


###Task1
###Select your features

features_list = ['poi','salary','bonus','expenses','exercised_stock_options',
                 'total_stock_value',"from_poi_to_this_person",
                "from_this_person_to_poi","from_messages",
                 "to_messages",'loan_advances','long_term_incentive',
                 'deferral_payments','deferred_income'
    ]
#data = featureFormat(data_dict, features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
#number of people 
print "Number of People :{}".format(  len(data_dict) )
print "Number of Features :{}".format(len(data_dict[data_dict.keys()[0]]))
count  =0
count1 =0
count2 =0 
count3 =0
count4 =0
count5 =0
count6 =0
#missing value in for poi
for x in data_dict:
    na = data_dict[x]["poi"]==1
    if (na != True):
        count += 1
for x in data_dict:
    na = data_dict[x]["poi"]==1
    if (na == True):
        count6 += 1
for x in data_dict:
    na = data_dict[x]["salary"]
    if na == "NaN":
        count1 += 1
for x in data_dict:
    na = data_dict[x]["bonus"]
    if na == "NaN":
        count2 += 1
for x in data_dict:
    na = data_dict[x]["expenses"]
    if na == "NaN":
        count3 += 1
for x in data_dict:
    na = data_dict[x]['exercised_stock_options']
    if na == "NaN":
        count4 += 1
for x in data_dict:
    na = data_dict[x]['total_stock_value']
    if na == "NaN":
        count5 += 1
print "False on poi : {} ".format(count)
print "POI  : {}".format(count6)
print "NaN on salary : {}".format(count1)
print "NaN on bonus: {}".format(count2)
print "NaN on expense: {}".format(count3)
print "NaN on exercise_stock_options: {}".format(count4)
print 'NaN on total_stock_value :{}'.format(count5)


# ## Outliers 
# 
# #### In the code below I want to find the outliers for exercised_stock_options , total_stock_value , salary, expenses and bonus by graphing them to see the outliers.

# In[3]:


data = featureFormat(data_dict, features_list)
for point in data:
    exercised_stock_options = point[4]
    total_stock_value = point[5]
    matplotlib.pyplot.scatter(exercised_stock_options , total_stock_value )

matplotlib.pyplot.xlabel("exercised_stock_options")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()
#graph from and to poi
for point in data:
    salary = point[1]
    expenses = point[3]
    matplotlib.pyplot.scatter( salary, expenses )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("expenses")
matplotlib.pyplot.show()

#salary and bonus

for point in data:
    
    salary = point[1]
    bonus  = point[2] 
    matplotlib.pyplot.scatter(salary,bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# #### In the graph above we can see there is a outlier when salary reach above 2.5  and another outlier when exercised_stock_options  above 3.0 . The first thing we look at salary we find out that the outlier is 26704229 in the cell below we have to locate the person that is equal to that salary.

# In[4]:


#task 2 remove outlier
print "Salary Outlier:"
#locate outlier for salary
for x in data_dict:
    da = data_dict[x]['salary']  
    if da == 26704229 :      
        print x
        print data_dict[x]['salary']
print


# ####  We find out that the person with 26704229 is not person but the total salary of all the people in our data_dict. So we a have to remove that outlier by popping like you see in the graph below.

# In[5]:


data_dict.pop("TOTAL", 0 ) 


# #### Now that we pop the Total we going to relook at the graph again to see if we can notice another outlier, 

# In[6]:


data = featureFormat(data_dict, features_list)

for point in data:
    exercised_stock_options = point[4]
    total_stock_value = point[5]
    matplotlib.pyplot.scatter(exercised_stock_options , total_stock_value )

matplotlib.pyplot.xlabel("exercised_stock_options")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()

for point in data:
    salary = point[1]
    expenses = point[3]
    matplotlib.pyplot.scatter( salary, expenses )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("expenses")
matplotlib.pyplot.show()

for point in data:
    
    salary = point[1]
    bonus  = point[2] 
    matplotlib.pyplot.scatter(salary,bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# #### In the graph above we see more outlier that we need to get rid off. In the code below we going to locate the outlier for salary and bonus. 

# In[7]:


print "Salary Outlier(after removing Total):"
#locate outlier for salary
for x in data_dict:
    da = data_dict[x]['salary']  
    if da >= 800000 and da != "NaN":      
        print x
        print data_dict[x]['salary']


# In[8]:


print "bonus Outlier:"
#locate outlier for salary
for x in data_dict:
    da = data_dict[x]['bonus']  
    if da >= 5000000 and da != 'NaN' :      
        print x
        print data_dict[x]['bonus']


# #### In the code above we found some outlier for salary and bonus usually we would remove them from our dictonary but since they are important people that affect this case we cannot remove them. 

# In[ ]:





# ## Feature Selection 
# #### In the code below i create a new feature creating the fraction  from_poi_to _this_person and a fraction from_this_person_to_poi and the adding them to the data_dict

# In[9]:


### Task 3: Create new feature(s)
import pickle


def computeFraction( poi_messages, all_messages ):

    fraction = 0.
    if poi_messages == "NaN":
        poi_message = 0 
    if all_messages == "NaN":
        return 0
    fraction = float(poi_messages)/float(all_messages)


    return fraction


 

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]
    #print
   
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
   
    

def submitDict():
    return submit_dict



# #### Now that you create the new feature it time to use the KBest to locate the  best feature to use in you but first you must add the feature you create to the list then run the KBest. KBest will tell you the best feature to use in you classifiers

# In[10]:


features_list = ['poi','salary','bonus','expenses','exercised_stock_options',
                 'total_stock_value',"from_poi_to_this_person",
                "from_this_person_to_poi","from_messages",
                 "to_messages",'loan_advances','long_term_incentive',
                 'deferral_payments','deferred_income','fraction_from_poi',"fraction_to_poi"
    ]



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[11]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(k=2).fit(features, labels)
##X_new.shape
print X_new.scores_


# In[12]:


features_list = ['poi','salary','bonus','expenses','exercised_stock_options',
                 'total_stock_value',"from_poi_to_this_person",'loan_advances',
                 'long_term_incentive','deferred_income',"fraction_to_poi" ]



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(k=2).fit(features, labels)
#X_new.shape
print X_new.scores_


# #### With the code above we see now choose the best features to use which are poi, salary,bonus, exercise_stock_options, total_stock_value, fraction_to_poi , and deferred_income. In the code below i run the code again to make sure that the KBest didn't change when i remove the features that i was going to work with. The reason i choose the 6 features if i have less than 6 than my data will be underfit because too few feature and the reason i don't have more feature 6 feature my data will be overfitting because it haves to many features.

# In[13]:


### Store to my_dataset for easy export below.
#my_dataset = data_dict

#features_list = ['poi','salary','bonus','exercised_stock_options','total_stock_value',
#                 'fraction_to_poi','deferred_income']


### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)


# In[14]:


#accuracy for each feature in the list 
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2

#X_new = SelectKBest(k=2).fit(features, labels)
#X_new.shape
#print X_new.scores_


# ## Classifiers
# #### Now that we know which features we going to work with it time to pass them through some classifiers and compare their precision and recall with one another.

# In[15]:



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn import cross_validation



# ### Naive Bayes

# In[16]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset, features_list )



# ## AdaBoost

# In[17]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
test_classifier(clf , my_dataset, features_list)


# ## Decision Tree

# In[18]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
test_classifier(clf , my_dataset, features_list)


# #### In the graph above we pass our dataset and features_list to Naive Bayes, AdaBoost and Decision Tree and we can see that Naive Bayes haves the best Precision and Recall out of all the 3 classifier , AdaBoost have a better precision than recall , and Decision Tree have the worst Precision and Recall out of all the 3 classifies. The reason Decison Tree have the worst because we haven't the the classifier to get a better Precision and Recall.

# #### Tuning 
# 

# In[19]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.htm
#test_classifier(clf,my_dataset,features_list)


# #### In order to get a better Precision and Recall for Decision Tree  we have to tune it so that we can get them both above .3. The code below i tune the tree by give the min_samples_split =13 , class_weight = balanced and the min_samples_leaf = 5 . By tuning it Decision Tree have the best Recall out of all of the classifiers.

# In[26]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 13,class_weight = "balanced",
                                  min_samples_leaf = 5 , max_leaf_nodes = 10)
test_classifier(clf , my_dataset, features_list)


# In[28]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 5,class_weight = "balanced",
                                  min_samples_leaf = 5 , max_leaf_nodes = 5)
test_classifier(clf , my_dataset, features_list)


# In[40]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 13,class_weight = "balanced",
                                  min_samples_leaf = 8  , max_leaf_nodes = 5)
test_classifier(clf , my_dataset, features_list)


# #### In the below i get the dataset and features_list after it been split  and then run the code with decision tree from above and get new precision and recall.

# In[23]:



# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
#test_classifier(clf , my_dataset , features_list)


# In[41]:


from sklearn import cross_validation
cv = cross_validation.StratifiedShuffleSplit(labels, 100, random_state = 42)
test_classifier(clf , my_dataset , features_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




