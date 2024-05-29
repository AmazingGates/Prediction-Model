# There are other forms of loss as well that work especially well when we deal with more real valued cases, cases
#like the mapping between the advertising budget and the amount that we do in sales, for example.
# Because in that case, we care not just that we get the number exactly right, but we care how close we were to 
#the actual value.

# If the actual value that we did was 2800 dollars in sales, and we predicted that we would do 2900 dollars in sales,
#maybe that's pretty good, maybe that's much better than if we predicted that we would do 1000 dollars in sales, 
#for example.

# And so we would like our loss function to be able to take that into account as well.

# Take into account not just whether the actual value and the expected value are exactly the same, but also
#take into account, how far apart they were.

# And for that one approach, we call L One Loss.

#   L1 Loss Function -
# - L(actual, predicted) = |actual - predicted|

# L1 loss doesn't just look at if actual and predicted are equal to each other, but we take the absolute value
#of the actual value minus the predicted value.

# In other words, we just ask, how far apart were the actual and predicted values, and we sum that up across all
#of the data points to be able to get what our answer ultimately is.

# So, what might this actual look like for our data set?

# Well if we go back to this representation, where we had advertising along the x axis, and sales along the y axis,
#our line was our prediction, our estimate, for any given amount of advertising, what we predicted sales were going
#to be.


#  y axis     
#    |               /              
#    |              /  o
# s  |        o    /          
# a  |            /     o
# l  |      o    /       
# e  |          /  o
# s  |     o   /     o
#    |        /  o
#    |  o    /  
#    |    o /   
#    |   o /  
#    |  o /  o
#    |   /   o
#    |o /     
#    | /   o
#    |/_____________________________________________________
#                        advertising                  x axis


# And our L1 loss, is just how far apart vertically along the sales axis our prediction was from each of the data
#points.

# So we could figure out exactly how far apart our prediction was for each of the data points, and figure out as a 
#result of that, what our loss is over all for this particular hypothesis, just by adding up all of these various 
#individual loses for each of these data points.


#  y axis                 /
#    |                   /|
#    |                  / |
#    |                 /| |
#    |                /|| |
#    |               / || |            
#    |              /  |o |
# s  |        o    /|  |  |      
# a  |        |   / |  |  o
# l  |      o |  /| |  |    
# e  |      | | / | o  |
# s  |     o| |/| |    o
#    |     || / | o
#    |  o  ||/| | 
#    |  | o|/ | | 
#    |  |o|/  | |
#    | o||/|  | o
#    | ||/ |  o
#    |o|/  |   
#    ||/   o
#    |/_____________________________________________________
#                        advertising                  x axis


# And our goal then, is to try and minimize that loss.

# To try and come up with some line that minimizes what the utility loss is by judging how far away our estimate
#amount of sales are from the actual amount of sales.

# And it turns out that there are other loss functions as well. 

# One that's quite popular is the L2 loss function.

#   L2 Loss Function -
# - L(actual, predicted) = |actual - predicted|2

# The L2 loss, instead of just using the absolute value, like how far apart away the actual value is from the predicted
#value, it uses the square of the actual minus the predicted.

# So how far apart are the actual and predicted value, and it squares that value, effectively penalizing much more
#harshly anything that is a worst prediction.

# So image that we have two data points that we predict as being one value away from the actual value, as opposed
#to one data point that we predict as being two away from the actual value.

# The L2 loss function will more harshly penalize that one that is two away, because it is going to square how 
#ever much the difference is between the actual value and predicted value. 

# And depending on the situation, we might want to choose the loss function we use, depending on what we care 
#about minimizing.

# But what we do run the risk of with any of these loss functions, as with anything that w're trying to do, is a
#problem known as overfitting. 

#   Overfitting -
# - A model that fits too closely to a particular data set and therefore may fail to generalize to future data 

# And overfitting is a big problem we can encounter in machine learning.

# We would like our model to be able to accurately predict data, in the inputs and outputs pairs, for the data
#that we have access to.

# But the reason we want to do so, is because we want our model to generalize well to data that we haven't seen
#before.

# We would like to take data from the past year of whether it was raining or not raining, and use that data to
#generalize towards the future.

# Say in the future, whether it is going to be raining or not raining.

# Or if we have a whole bunch of data on what counterfeit and not counterfeit U.S dollar bills looked like in the
#past when people have encountered them.

# We'd like to train a computer to be able to in the future generalize to other dollar bills that we might see as well.

# And the problem with overfitting is that if we train and tie ourselves to closely to the data set that we're 
#training our model on, we could end up not generalizing very well.

# So what does this look like.

# Well we might image the rainy days and not rainy days example agian.

# Remember that our o data points indicate rainy days, and our x data points indicate not rainy days.

# And we decided that we felt pretty comfortable with drawing a line like this, as the decision boundary 
#bewteen rainy days and not rainy days.


# x axis (Pressure)
#|                          x             /    o
#|                       x           x   /    o
#|                  x          x        /     o
#|                    x             x  /     o
#|                     x              /       o
#|                        x          /     o        o
#|                   x              /     o
#|                                 /                   o    o
#|      x        x                /       o               x
#|           x                   /     o       o      o
#|                              /                       o
#| x          x                /
#|__________________________________________________________
#                                    y axis (Humidity)


# So we can pretty comfortably say that points on the right side of the decision line are more likely to be rainy days.

# And points on the left of the decision line are more likely to be not rainy days.

# But the imperical loss isn't zero in this particular case, because we couldn't categorize everything perfectly.

# There was one outlier, that one day where it wasn't raining but yet our model still predicts that it is a rainy
#day.

# But that doesn't neccessarily mean that our model is bad, it just means that our model isn't 100 percent accurate.

# If we really wanted to try and find the hypothesis that resulted in minimizing the loss, we could come up with
#a different decision boundary that wouldn't be a line, but would look something like this.


# x axis (Pressure)
#|                          x             /    o
#|                       x           x   /    o
#|                  x          x        /     o
#|                    x             x  /     o
#|                     x              /       o
#|                        x          /     o        o
#|                   x              /     o
#|                                 /________________________o    o
#|      x        x                ________________________x_|
#|           x                   /        o
#|                              /      o       o      o 
#| x          x                /                        o
#|__________________________________________________________
#                                    y axis (Humidity)


# This decision boundary does separate all of the x data points from the o data points, because all the x data points
#stil fall on the left side of the decision line, and all of the o data points still fall on the right side of the 
#decision line. 

# But we would probably argue this is not as good of a prediction.

# Even though it seems to be more accurate based on all of the available training data that we have for training
#this machine learning model.

# We might say that it's probably not going to generalize well.

# Because of the o data points that are so close to the x data point that is to the far right, we would want to
#still consider those to be rainy days, because we think that this x data point was probably just an outlier.

# So the only thing we care about is minimizing the loss on the data we have available to us, we run the risk of
#overfitting.

# And this can happen in a classification case, and it can also happen in a regression case.

# Like here where we predicted what we thought was a pretty good line, relating advertising to sales.

# Trying to predict what sales were going to be for a given amount of advertising.


#  y axis     
#    |               /              
#    |              /  o
# s  |        o    /          
# a  |            /     o
# l  |      o    /       
# e  |          /  o
# s  |     o   /     o
#    |        /  o
#    |  o    /  
#    |    o /   
#    |   o /  
#    |  o /  o
#    |   /   o
#    |o /     
#    | /   o
#    |/_____________________________________________________
#                        advertising                  x axis


# But we could come up with a line that does a better job of predicting the traing data.

# And it would look something like this.

# Something connecting all the various differnt data points.


#  y axis     
#    |                             
#    |          _________o
# s  |        o|_______             
# a  |        _________|o
# l  |      o__________|           
# e  |      |______o
# s  |     o____o_|
#    |     |____o
#    |  o_______|      
#    |  |__o    
#    |   o|__   
#    |  o___|o
#    |  |__o
#    |o   /   
#    || \o
#    ||_____________________________________________________
#                        advertising                  x axis


# And now, there is no loss at all.

# Now we've perfectly predicted given any advertisement, what sales are.

# And for all the data available to us, it's going to be accurate.

# But it's probably not going to generalize very well.

# We have overfit our model on the training data that is available to us.

# And so in general we want to avoid overfitting, we like strategies to make sure that we haven't overfit our model
#to a particular dataset.

# And there are a number of ways that we could try and do this.

# One way is by examining what it is that we're optimizing for.

# In an optimization problem, all we do is we say there is some cost, and we want to minimize that cost.

# And so far, we define that cost function.

# The cost of a hypothesis, just would be equal to the imperical loss of that hypothesis.

#   cost(h) = loss(h)

# Like how far away are the actual data points, the outputs, away from what we predicted them to be, based on 
#that particular hypothesis.

# And if all we're trying to do is minimize cost, which means minimizing the loss in this case, then the result
#is going to be that we might overfit.

# That to minimize cost, we're going to try and find a way to perfectly match all of the input data points.

# And that might happen as a result of overfitting on that particular input data.

# So in order to address this, we add something to the cost function.

# What counts as cost?

# Well not just loss, but some measure of the complexity of the hypothesis.

#   cost(h) = loss(h) + complexity(h)

# The complexity of the hypothesis is something we would need to define, for how complicated does our line look.

# This is sort of an arcum razor style approach, where we would want to give preference to a simpler decision
#boundary, like a straight line, for example, some simpler curve, as opposed to something far more complex, that
#might represent the training data better, but might not generalize as well.

# We'll generally say that a simpler solution is probably the better solution, and probably the one that is more
#likely to generalize well to other inputs.

# So we measure what the loss is, but we also measure the complexity.

# And now that all gets taken into account when we consider the overall cost.

# That yes something might have less loss if it better predicts the training data, but if it's much more complex
#it still might not be the best option that we have.

# And we need to come up with some balance between loss and complexity, and for that reason we'll often see the 
#equation represented as multiplying the complexity by some parameter, that we have to choose, parameter lambda
#in this case, where we're saying, if lambda is a greater value, then we really wanna penalize more complex 
#hypotheses.

# Where as if lambda is smaller, we're going to penalize more complex hypotheses a little bit.

#   cost(h) = loss(h) + (lambda)complexity(h)

# It's up to the machine learning programmer to decide, where we want to set that value of lambda for how much
#do we want to penalize a more complex hypothesis that might fit the data a little better.

# And again, there's no one right answer for a lot of these things, that depending on the data set, depending
#on the data we have available to us, and the problem we wre trying to solve, our choice of these parameters
#may vary, and we may need to experiment a little bit to figure out what the right choice of that is ultimately
#going to be.

# This process then, considering not only loss, but also some measure of the compexity, is known as regularization.

#   Regularization -
# - Penalizing hypotheses that are more complex, to favor a simpler, more general hypotheses.

# This is more likely to be able to apply to other situations that are dealing with other input points unlike
#the ones that we've neccessarily seen before.

# So often times we'll see us add some regularizing term to what we're trying to minimize, in order to avoid
#this problem of overfitting.

# Now another way of making sure we don't overfit is by running some experiments, and to see whether or not
#we are able to generalize our model that we created to other data sets as well.

# And it's for that reason that often times that when we are doing a machine learning experiment, when we've
#got some data and we want to try and come up with some function that predicts given some input, what the 
#output is going to be, we don't neccessarily want to do our training on all of the data we have available to
#us, but we could imploy a method known as holdout cross validation.

#   Holdout Cross-Validation -
# - Splitting data into a training set and a test set, such that learning happens on the training set, and is
#evaluated on the test set.

# So the learning happens on the training set, we figure out what the parameters should be, we figure out what
#the right model is, and then we see how well it does at predicting things inside of the testing set, some set 
#of data that we haven't seen before.

# And the hope then, is that we're going to be able to predict the testing set pretty well, if we're able to
#generalize based on the training data that is available to us.

# If we overfit the training data though, and we're not able to generalize, well then when we look at the
#testing set, it's likely going to be the case that we're not going to predict things in the testing set nearly
#as effectively.

# So this is one method of cross validation.

# Validating to make sure that the work we have done is actually going to generalize to other data sets as well.

# And there are other statistical techniques we can use as well.

# One of the downsides of just this holdout cross validation, is if we split something 50 -50.
# We train using 50 percent of the data, and test using the other 50 percent, is that there is a fair amount of 
#data that we are now not using to train, that we might be able to get a better model as a result, for example.

# One approach is known as k-fold cross-validation

#   K-Fold Cross-Validation -
# - Splitting data into k sets, and experimenting k times, using each set as a test set once, and using remaining
#data as training set.

# In k fold cross validation, rather than just divide things into two sets and run one experiment, we divide things
#into k different sets.

# So maybe we divide things up into 10 different sets, and then run 10 different experiments.

# So if we split up our data into 10 different sets of data, then what we'll do is each time for each one of 
#our experiments, we will hold out one of those sets of data, where we'll say, let us train our model on these
#nine sets, and then test to see how well it predicts on set number 10.

# And then pick another set of nine sets to train on, and then test it on the other one that we held out, where
#each time we train the model on everything minus the one set that we are holding out, and then test to see how
#well our model performs on the set that we did hold out.

# And what we end up getting is 10 different results, 10 different answers for how accurately our model worked.

# And oftentimes, we could just take the average of those 10 to get an approximation for how well we think our
#model performs overall.

# But the key idea is separating the training data fron the testing data, because we want to test our model
#on data that is different from what we trained the model on.

# Because the training, we want to avoid overfitting.

# We want to be able to generalize.

# And the way we test whether we're able to generalize is by looking at some data that we haven'tseen before and
#seeing how well we're actually able to perform.

# And so if we want to actually implement any of these techniques inside of a programming language like Python,
#there are a number of ways we can do that.

# We could write this from scratch on our own, but there are libraries out there that allow us to take advantage
#of existing implementations of these algorithms, that we can use the same types of algorithms in a lot of 
#different situations.

# And so, there's a library, a very popular one, known as scikit-learn, which allows us, in Python, to be able
#to very quickly get set up with a lot of different machine learnng models.

# This library has already written an algorithm for nearest-neighbor classification, for doing perceptron learning,
#for doing a bunch of other types of inference and supervised learning that we haven't yet talked about.

# But using it, we can begin to try actually testing how these models work and how accurately they perform.

# So let's go ahead and take a look at one approach to trying to solve this type of problem.

# First, we will add our imports, mainly scikit learning as sklearn.

import csv
import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Next we will start to write our code.

# Now we will make sure we have our banknotes csv, which is information about various different banknotes that 
#people took pictures of and measured Various different properties of those banknotes.

# And in particular, some human categorized each of those banknotes as either a counterfeit banknote, or as 
#an authentic banknote.

# So inside our csv, each row represnts one banknote.

# We have four different input values for each of the data points, just information, some measurement that was made
#on the banknote.

# And what those measurements are isn't exactly as important as the fact that we do have access to that data.

# But more importantly, we have access for each of these data points to a label, where zero indicates something
#like this is an autthentic bill, and a data point labeled one, which means the bill is counterfeit, as least
#according to the human researcher who labeled this particular data.

# So we have a whole bunch of data representing a whole bunch of different data points, each of which has these 
#various different measurements that were made on that particular bill, and each of which has an output value,
#zero or one.
# Zero meaning the bill is authentic, or One, meaning the is counterfeit.

# And what we would like to do is use supervised learning to begin to predict or model some sort of function 
#that can take the four values as input and predict what the output would be.

# We want our learning algorithm to find some sort of pattern that is able to predict based on these measurements,
#something that we could measure just by taking a photo of a bill, predict whether that bill is aurhentic or
#whether that bill is counterfeit.

# Now let's go over the process of implementing this in our code.

# We will set our model equal to the perceptron model. 
# This is one of those models where we are just trying to figure out some settings of weights that is able to divide
#ur data into two different groups.
model = KNeighborsClassifier(n_neighbors=1) # SVC() (Support Vector Classifier) Model can also be used to run this algorithm 
# The KNeighborsClassifier Model can also be must, but it takes a parameter of (n_neighbors=1). 
# Note we can specify how many neighbors we want to look at, but for example purposes we'll just look at 1 for now.

# Then we are going to read our data in for our file from banknotes.csv

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    # Basically for every row, we're going to separate that row into the first four values of that row, which is
    #the evidence for that row.
    # And then the label, where if the final column in that row is a zero, the label is authentic, else, it's
    #counterfeit.
    # So we're effectively reading data in from our csv file, dividing it into a whole bunch of rows where each row
    #has some evidence, those four input values that are going to be inputs to our hypothesis function.
    # And then the label, the output, whether it is authentic or counterfeit, that is the thing that we are 
    #trying to predict.
    data = []

    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# The next step is that we would like to split our dataset up into a traing set, and a testing set, some set of data
#that we would like to train our machine learning model on, and some set of data that we would like to use to test
#that model, see how well it performed.
# So what we'll do is we'll go ahead and figure out the length of the data, how many data points do we have.
# We'll take half of them, save the number as a number called holdout.
# That is how many items we are going to holdout for our dataset to save for the testing phase.
# We'll randomly shuffle the data so it's in some random order.
# And then we'll say, our testing set will be all of the data up to the holdout.
# Our training data will be everything else, the information that we are going to train our model on.

# Separate data into training and testing groups
holdout = int(0.50 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Next we split our training data into two different sets. 
# We need to divide it into our x values, where x represents the inputs.
# So the x valuesm the x values that we are going to train on, are basically for every row in our traing set,
#we are going to get the evidence for that row, those four values, which is basically a vector of four numbers,
#where that is going to be all of the input.
# And then we need the y values.
# What are the outputs that we want to learn from, the labels that belong to each of these various different
#input points?
# Well, that's going to be the same thing for each row in the training data.
# But this time, we take that row and get what it's label is, whether it is authentic or counterfeit.
# So we end up with one list of all these vectors of our input data, and one list, which follows the same order,
#but is all of the labels that correspond with each of those vectors.
# And then to train our model, which in this case is just this perceptron model, we just call model.fit, pass in
#the training data, and what the labels for those training data are.
# And scikit learn will take care of fitting the model, and will do the entire algorithm for us. 

# Train model on traing set
X_training = [row["evidence"] for row in training]
Y_training = [row["label"] for row in training]
model.fit(X_training, Y_training)

# And when everything is done, we can test to see how well that model performed.
# So we can say, let's get all of these input vectors for what we want to test on.
# So for each row in our testing data set, go ahead and get the evidence.
# And the y values, those are what the actual values were for each of the rows in the testing data set, what the
#actual label is.
# But then we are going to generate some predictions.
# We are going to use this model and try to predict, based on the testing vectors, we want to predict what the
#output is.
# And our goal then is to now compare y tseting with predictions.
# We want to see how well our predictions, based on the model, actually reflect what the y values were, what 
#the output is, that were actually labeled.
# Because we now have this label data, we can assess how well the algorithn worked.

# Make predictions on the testing set 
X_testing = [row["evidence"] for row in testing]
Y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing,)

# Now we can just compute how well we did.
# The zip function basically lets us look through two different lists, one by one at the same time.
# So for each actual value and for each predicted value, if the actual is the same thing as what we predicted,
#we'll go ahead and increment the correct counter by one.
# Otherwise, we'll increment the incorrect counter by one.

# Compute how well we performed 
correct = 0
incorrect = 0
total = 0

for actual, predicted in zip(Y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else: 
        incorrect +=1

# And in the end, we can print out the results.
# How many we got right.
# How many we got wrong.
# And our overall accuracy, for example.

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")

# So we can go ahead and run this.

# We can run these experiments multiple times, because we are randomly reorganizing the data every time.

# We're technically training these on slightly different training sets so we might want to run multiple experiments
#to see how well they are actually going to perform.

# But in short, they all perform very well.

# While some of them performed slightly better than others, that might not always be the case for every dataset.

# But we can begin to test now, by putting together this machine learning models using scikit learn to be able to
#train some training set, and then test on some testing set as well.