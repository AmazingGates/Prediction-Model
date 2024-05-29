import csv
import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# We can begin to test now, by putting together this machine learning models using scikit learn to be able to
#train some training set, and then test on some testing set as well.

# And this splitting up, into training groups and testing groups, and testing, happens so often that scikit learn has
#functions built in for trying to do it.

model = KNeighborsClassifier(n_neighbors=1)

with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []

    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

holdout = int(0.50 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

X_training = [row["evidence"] for row in training]
Y_training = [row["label"] for row in training]
model.fit(X_training, Y_training)

X_testing = [row["evidence"] for row in testing]
Y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing,)

correct = 0
incorrect = 0
total = 0

for actual, predicted in zip(Y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else: 
        incorrect +=1

print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")


# We did it all by hand just now, but if we take a look at our next algorithm, we will see that we can take advantage
#of some other features that exist in scikit learn.

# We can really simplify a lot of our logic.

model = KNeighborsClassifier(n_neighbors=1)

with open("banknotes1.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []

    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# There is a function built into scikit learn called train_test_split, which we automatically split data into a
#training group and testing group.

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

# We just have to specify what proportions should go to the testing group, something like .5, for half the data.
X_training, X_testing, Y_training, Y_testing = train_test_split(
    evidence, labels, test_size = 0.5
)

# Then we can fit the model on the training data

# Model fit
model.fit(X_training, Y_training) 

# Next we can make the predictions.on the testing data

# Make predictions on the testing set
predictions = model.predict(X_testing)

# And then we just count how many times our testing data matched the prediction, and how many times it didn't.

# Compute how well we peformed 
correct = (Y_testing == predictions).sum()
incorrect = (Y_testing != predictions).sum()
total = len(predictions)


print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")

# So very quickly we can write programs with far fewer lines of code.

# It took us less than half the lines of code of our previous algorithm to run the same program and acheive the same
#results.

# So these types of libraries can allow us without really knowing the implementation details of these algorithms
#to be able to use the algoriithm in a very practical way to be able to solve these types of problems.

# And that, was supervised learning.


# This task of giving a whole set of data, some input output pairs, we would like to learn some function that maps
#those inputs to those outputs.

# It turns out that there other forms of learning as well.

# And another popular type of machine learning is known as reinforcement learning.

#   Reinforcement Learning -
# - Given a set of rewards or punishments
# - Learn what actions to take

# And the idea of reinforcement learning, is rather than just being given a whole dataset at the beginning,
#input output pairs, reinforcement learning is all about learning from experience.

# In reinforcement learning, our agent, whether it's a physical robot trying to make actions in the world, or some
#virtual agent that is a program running somewhere.

# Our agent is going to be given a set of rewards or punishments in the form of numerical values, and based on that
#it learns what actions to take in the future. 

# Our agent, our AI, will be put in some sort of enviornment, it will make some action, and based on the actions
#that it makes, it learns something.

# It either gets a reward when it does something well, it gets a punishment when it does something poorly, and it
#learns what to do, or what not to do in the future, based on those individual experiences.

# And so what this will often look like, is it will often start with some agent, some AI, which again, might be
#a physical robot, or it could just be a program, and our agent is situated in their enviornment, where the
#enviornment is where they are going to make their actions, and it's what going to give them rewards or punishments
#for various actions that they take. 


#   Enviornment

#   Agent


# So for example, the enviorment is going to start off by putting our agent inside of a state.


#   Enviornment ---
#                 | State
#   Agent <-------|


# Our agent has some state, that in a game might be the state of the game that the agent is playing, in a world
#that the agent is exploring, might be some position inside of a grid, representing the world the agent is 
#exploring.

# But the agent is in some sort of state, and in that state, the agent needs to choose to take an action.

# The agent likely has multiple actions it can choose from, but they pick an action.


#    ---> Enviornment ---
#   |                   | State
#   |---Agent <---------|


# So they take an action in a particular state, and as aresult of that, the agent will generally get two things
#in response.

# The agent gets a new state that they find themselves in.

# After being in this state, taking one action and ending up in some other state.

# And their also given some sort of numerical reward.

# Positive meaning reward, meaning they did a good thing.

# Negative generally meaning they did something bad and they received some sort of punishment.

#    ---> Enviornment ----------------
#   |                   | State  | Reward
#   |---Agent <---------|---------    |
#         |<__________________________|

# And that is all the information the agent has.

# It's told what state it's in, it makes some sort of action, and based on that it ends up in another state, and it
#ends up getting some particular reward.

# And it needs to learn based on that information, what actions to begin to take in the future.

# We could imagine generalizing this to a lot of different situations.

# This is often times how we train through reinforcement learning.

# Give the agent some sort of numerical reward everytime it does something good, and punish it everytime it does
#something bad.

# And then just let the AI learn, based on that sequence of rewards, based on trying to take various different 
#actions, we can begin to have the agent learn what to do in the future, and what not to do.

# So in order to begin to formalize this, the first thing we need to do is formalize this notion of what we mean
#about states and actions and rewards, like what does this world look like.

# And often times we'll formulate this world as what is known as a markov decision process. 


#   Markov Decision Process -
# - Model for decision making, representing states, actions, and their rewards.


# Similar in spirit to a markov chain, but a markov decision process is a model we can use for decision making, for 
#an agent trying to make decisions in it's enviornment.

# And it is a model that allows us to represent the various different states that an agent can be in, the various
#different actions they can take, and also, what the reward is for taking one action as opposed to another action.

# So then, what does it actually look like?

# Well, if we recall a markov chain from before, it looked a little something like this.

# We have a whole bunch of these individual states, and each state immediately transisitons to another state
#based on some probability distribution.

# We saw this in the context of the weather, where we said if it was sunny, there was some probability that it 
#would be sunny the next day, with some other probability that it would be raining, for example.

# Sun -----> Sun -----> Rain -----> Rain -----> Rain -----> Rain ----->
# X0         X1          X2          X3          X4          X5


# But we could also image generalizing.

# O -----> O -----> O -----> O -----> O -----> O ----->

# So not just sun and rain anymore, but we just have these states, where one state leads to another state, according
#to some probability distribution.

# But in this original model there was no agent that had any control over this process.

# It was just entirely probability based, where it was some probability that we moved to this next state, or maybe
#it was going to be some other state with some other probability.

# What we'll now have is the ability for the agent in this state to choose from a set of actions, where maybe instead
#of just one path forward, they have 3 different choices of actions that each lead down differents paths.

# And even this is a bit of an over simplification, because each of these states we might image more branching points,
#where there are more decisions that can be taken as well.


#   O -----> O -----> O -----> O -----> O -----> O ----->
#  /
# O -- O -----> O -----> O -----> O -----> O -----> O ----->
#  \
#   O -----> O -----> O -----> O -----> O -----> O ----->


# So we've extended the markov chain to say that from a state, we now have available action choices.

# And each of those action choices might be associated with its own probability distribution of going to various
#different states.

# Then in addition, we'll add another extention.

# Where anytime we move from a state, taking an action, going into another state, we can associate a reward with
#that outcome.

# Saying either (r) is positive, meaning some positive reward, or (r) is negative, meaning some punishment.

# And this is what we'll consider to be a markov decision process.

# That a markov decision process has some initial set of states, states of the world that we can be in. 

# We have some set of actions, that given a state, we can say what are the available set of actions to me in 
#that state, an action that we can choose from.

# Then we have some transition model.

# The transition model before just said that given our current state, what is the probability that we end up in that 
#next state, or this other state.

# The transition model now has effectively two things we're conditioning.

# We're saying, given that we're in this state, and that we take this action, what's the probability that we end
#up in this next state.

# Now maybe we live in a very demanding deterministic world in this markov decision process, where given a state
#and given an action, we know for sure what next state we'll end up in.

# But maybe there's some randomness in the world, that when given a state and taking an action, we might not always
#end up in the exact same state, there might be some probabilities involved there as well.

# The markov decision process can handle most of those possible cases.

# And then finally, we have a reward function, generally called R, that in this case says, what is the reward for
#being in this state, taking this action, and then getting to s prime, this next state.

# So we start in our original state, we take this action, we get to the next state.

# What is the reward for doing that process.

# We can add up these rewards everytime we take an action, to get the total amout of rewards that an agent might get
#from interacting in a particular enviornment, while using this markov decision process.

#   Markov Decision Process -
# - Set of states S
# - Set of Actions(s)
# - Transition model P(s'|s,a)
# - Reward function R(s,a,s')

# So what might this acutually look like in practice?

# Let's create a simulated world here, where we have an agent that is just trying to navigate its way.

# The agent is our letter A in our grid, like a robot in the world, trying to navigate its way through this grid.

# And ultimately it's trying to find its way to the goal.

# And if it gets to the goal, then it's going to get some sort of reward.

# But then, we might also have some red squares that are places where we get some sort of punishment, some bad
#place where we don't want the agent to go. These squares are represented by our letter R.

# And if it ends up in the red square, then our agent is going to get some sort of punishment as a result of that.

# But the agent doesn't originally know all of these details.

# It doesn't know that these states are associated with punishments.

# But maybe it does know that our goal state is associated with a reward. Maybe it doesn't.

# But it just needs to sort of interact with the environment to try and figure out, what to do and what not to do. 


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   |   |
#   |   |   |   | R |   |
#   |_A_|_R_|___|___|___|


# So the first thing the agent might do is, given no additional information, if it doesn't know what the punishments
#are, it doeesn't know where the rewards are, it just might try and take an action.

# And it takes an action and ends up realizing that it got some sort of punishment.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   |   |
#   |   |   |   | R |   |
#   |___|_A_|___|___|___|


# And so what does it learn from that experience?

# Well, it might learn that when we're in our original state in the future, don't take the action move to the right,
#that that is a bad action to take.

# That in the future, if we ever find ourselves back in the original state, don't take this action of going to
#the right when we're in that original state, because that leads to punishment.

# That might be the intuition at least.

# And so we could try doing other actions.

# We can move up, but that didn't lead to any immediate rewards.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   |   |
#   | A |   |   | R |   |
#   |___|_R_|___|___|___|


# So we'll try something else.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   |   |
#   |   | A |   | R |   |
#   |___|_R_|___|___|___|


# Then maybe try something else.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   |   |
#   |   |   |   | R |   |
#   |___|_A_|___|___|___|


# Now we've found that we got another punishment, so we learn from this experience also.

# So the next time we do this whole process, we know that if we ever end up in that previous square, we shouldn't
#take the down action, because being in that state and taking the down action leads to some sort of punishment,
#a negative reward in other words.

# And the process repeats, so we might imagine just leting our agent explore the world, learning over time what
#states tend to correspond with poor actions, until eventually, if it tries enough things randomly, it might find
#that when we get to this state, and we take the up action, it might find that it actually gets some kind of reward
#for that.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   | A |   |
#   |   |   |   | R |   |
#   |___|_R_|___|___|___|


# And what it can learn from that is that if it's ever in this state, we should take the up action, because that
#leads to a reward.

# And over time, we can also learn that if we're in this state, we should take the left action, beacuse that 
#leads to the state that leads to the reward.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   |   | A |
#   |   |   |   | R |   |
#   |___|_R_|___|___|___|


# So we begin to learn over time not only which actions are good in particular states, but also which actions
#are bad, such that once we know some sequence of good actions that leads us to a reward, our agent can follow
#those intructions, follow the experience that it has learned.

# We didn't tell the agent what the goal was.

# We didn't tell the agent where the punishments were.

# But the agent can begin to learn from this experience, and learn to begin to perform these sorts of task better
#in the future. 


# So let's now try to formalize this idea. 

# Formalize the idea that we would like to be able to learn in this state, taking this action, is that a good thing,
#or a bad thing.

# There are lots of different models for reinforcement learning,and we're going look at just one of them today.

# The one that we're going to look at is a method known as Q learning.


#   Q-Learning -
# - Method for learning a function Q(s,a),
# - Estimate of the value of performing action a in state s


# And what Q learning is all about, is about learning a function, a function Q, that takes input s and a.
# Where s is a state and a is an action we take in that state.

# And what this Q function is going to do, is it is going to estimate the value, how much reward will we get
#from takng this action in this state.

# Originally we don't know what this Q functio should be. 

# But over time, based on experience, based on trying things out and seeing what the result is, we would like to
#try and learn what Q of s,a is for any particular state, and any particular action that we might take in that state.

# So what is the approach?

# Well the approach originally is we'll start with Q s,a equal to zero for all states s and for all actions a.

# That initially, before we've ever started anything, before we've had any experiences, we don't know the value 
#of taking any action in any given state, so we're going to assume that the value is just zero all across the 
#board.

# But then, as we interact with the world, as we experience rewards or punishments, or maybe we go to a cell where 
#we don't get either reward or punishment, we wanna somehow update our estimate of Q s,a.

# We wanna continually update our estimate of Q s,a based on the experiences and rewards and punishments that we 
#receive, such that in the future, our knowledge of what actions are good in what states will be better.

# So when we take an action, and receive some sort of reward, we want to estimate the new value of Q s,a.

# And we estimate that based on a couple of different things.

# We estimate it based on the reward that we're getting from taking this action and getting into the next state.

# But assuming the situation isn't over, assuming that there are future actions that we might take as well, we
#also need to take into account the expected future reward.

# That if we imagine an agent interacting with the enviorment, and sometimes we'll take an action and get a reward
#but then we could keep taking more actions and get more rewards, that these both are relevent, both the current 
#reward we're getting from this current step, and also our future reward.

# And it might be the case that we'll want to take a step that doesn't immediately lead to a reward, because later
#on down the line we know it will lead to more rewards as well.

# So there's a balancing act between current rewards that the agent experiences and future rewards that the agent
#experiences as well.

# And then, we need to update Q (s,a).

# Estimate the value of Q (s,a) based on the current reward, and the expected future rewards.

# And then we need to update this Q function to take into account this new estimate.

# Now we'll already have an estimate for what we think the value is, now we have a new estimate, and somehow we need 
#to combine these two estimates together, and we'll look at more formal ways that we can actually begin to do that.


#   Q Learning Overview

# - Start with Q(s,a) = 0 for all s,a
# - When we've taken an action and receivve a reward
#   - Estimate the value of Q(s,a) based on a current reward and expected future reward
#   - Update Q(s,a) to take into account the old estimate as well as our new estimate


# So to actually see what this formula looks like, here is the approach we'll take with Q learning.

# We're going to start with Q (s,a) being equal to zero for all states.

# And then eveytime we take an action a in state s and observe a reward r, we update:, our value, our estimate
#for Q (s,a).

# And the idea is that we're going to figure out what the new value estimate is minus what our existing value estimate is.

# So we have some preconceived notion for what the value is for taking this action in this state.

# Maybe our expectation is we currently think the value is 10, but then we're going to estimate what we now think 
#it's going to be.

# Maybe the new value estimate is something like 20.

# So there's a delta of 10.

# Meaning our new value estimate is 10 points higher then what our current value estimate is.

# And so we have a couple of options here.

# We need to decide, how much we want to adjust our current expectation of what the value is of taking this action,
#in this particular state, and what that difference is. 
# How much we add or subtract from our existing notionof how much we expect the value to be, is dependent on the 
#parameter alpha a(). Also called the learning rate.

# And alpha represents, in effect, how much we value new information compared to how much we value old information.

# An alpha value of 1 means we really value new information. That if we have a new estimate, that it doesn't
#matter what our old estimate is, we are only going to consider our new estimate because we always only want to
#take into consideration our new information.

# So the way that works is that if we image alpha being 1, well then we're taking the old balue of Q (s,a) and
#then adding 1 times the new value, minus the old value, which just leaves us with the new value.

# So when alpha is 1, all we take into consideration is what our new estimate happens to be.

# But over time, if we go through a lot of experiences, we already have some existing information.

# We might have tried taking this action nine times already, and now we just tried a tenth time.

# And we don't only wanna consider this tenth experience, we also want consider the fact that our prior nine
#experiences, those were meaningful too.

# And that's data we don't necessarily want to lose, so the alpha controls that decisison, controls how important is
#the new information.

# Zero would mean ignore all the new information, just keep the Q value the same.

# 1 means replace the old information entirely with the new information, and somewhere in between keep some sort
#of balance between these two values.

# We can put this equation a little more formally as well.

# The old value estimate is our old estimate for what the value is for taking this action, in a particular state.
# That's just Q (s,a).

# So we have it once here as (new value estimate), and we're going to add something to it.

# We're going to add alpha times the new value estimate minus the old value estimate. But the old value estimate 
#we just look up by calling this Q function Q(s,a).

# And what then is the new value estimate, based on this experience we have just taken, what is our estimate for the
#value of taking this action, in this particular state.

# Well it's going to be composed of two parts.

# It's going to be composed of what reward did we just get from taking this action in this state.
# And then it's going to be, what can we expect our future rewards to be from this point forward.

# so it's going to be (r), some reward we're getting right now, plus, whatever we estimate we're going to get
#in the future.

# And how do we estimate what we are going to get in the future?

# It's going to be another call to the Q function. It's going to be, take the maximum across all possible actions
#we could take next, and then say of all these possible actions, which one is going to have the highest reward.

# This is going to be our notion for how we going to perform this kind of update. 

# We have some estimate, some old estimate for what the value is for taking this action, in this state, and we're
#going to update it based on new information, that we experience some reward, we predict what our future
#reward is going to be, and using that we update what we estimate the reward is going to be for taking this action 
#in this particular state.

# And there are other additions that we might make to this algorithm as well.

# Sometimes it might not be the case that future rewards we want to weight equally to current rewards, maybe 
#we want an agent that values rewards now over rewards later, and so sometimes we can even add another term or
#another parameter where we discount future rewards, and say future rewards are not as valuable as rewards now,
#but that's up to the programmer to decide what that parameter should be.



#   Q Learning

# Start with Q(s,a) = 0 for all s,a
# Eveytime we take an action a in state s and observe a reward r, we update:


# Q(s,a) <---- Q(s,a) + a(new value estimate - old value estimate)
# Q(s,a) <---- Q(s,a) + a(new value estimate - Q(s,a))
# Q(s,a) <---- Q(s,a) + a((r + future reward estimmate) - Q(s,a))
# Q(s,a) <---- Q(s,a) + a((r + maxa' Q(s',a')) - Q(s,a))
# Q(s,a) <---- Q(s,a) + a((r + ymaxa' Q(s',a')) - Q(s,a)) (This is how we add another term)


# But the big picture idea, of this entire formula, is to say that everytime we experience some new reward,
#we take that into account, we update our estimate of how good is this action.

# And then in the future we can make decisions based on that algorithm.

# Once we have some good estimate for every state and every action, what the value is of taking that action, and
#then we can do something like implement a greedy decision making policy.

# That if we are in a state, and we want to know what action should we take in that state, we'll consider for all
#of our possible actions, what is the value of Q(s,a), what is our estimated value of taking that action in that
#state, and we would just pick the action that has the highest value after we evaluate that expression.

# So we pick the action that has the highest value, and based on that, that tells us what action we should take.

# At any given state that we are in we could just greedily say across all of our actions this action gives us the
#highest expected value, and so we'll go ahead and choose that action as the action that we take as well.


#   Greedy Decision-Making -
# - When in state s, choose action a with highest Q(s,a)


# But there is a downside to this kind of approach.

# And the downside comes up in a situation like this, where we know that there is some solution that gets us to
#the reward, and our agent has been able to figure that out.


#   _____________________ 
#   |   |   |   | G |   |
#   |   | R |   | A |   |
#   |   |   |   | R |   |
#   |___|_R_|___|___|___|


# But it might not necessarily be the best way, or the fastest way.

# If the agent is allowed to explore a little bit more, it might find that it can get the reward faster by taking
#some other route instead.

# And maybe we would like for the agent to figure that out as well.

# But if the agent always takes the action that it knows to be best, when it gets to a particularly good square that
#is a better option that it could take, it wouldn't know because it has never tried it. 

# It has learned to take the route that it knows will lead to the reward.

# It is never going to explore any other possible routes once it has learned a successful route.

# So in reinforcement learning there is this tension between exploration and exploitation.

# And exploitation generally refers to using knowledge that the AI already has.

# The AI already knows that this is a move that leads to a reward, so we'll go ahead and use that move.

# And exploration, is all about exploring other actions that we may not have explored as thoroughly before, 
#because maybe one of these actions might lead to better rewards faster, or to more rewards in the future.

# And so an agent that only ever exploits information, and never explores, might be able to get rewards, but
#it might not maximize its rewards, because it doesn't know what other possibilities are out there, possibilities
#that we would only know about by taking advantage of exploration.


#   Explore vs Exploit


# And so, how can we try and address this?

# One possible solution is known as the epislon greedy algorithm.

# Where we set episolon to just how often we want to make a random move.

# Where ocassionally we will just make a random move in order to say, let's try to explore and see what happens.

# And then the logic of the algorithm will be, with probability 1 - E, choose estimated best move.

# In a greedy case we would always choose the best move. 

# But in epislon greedy, where most of the time we're going to choose the best move, or sometimes going to
#choose the best move, but sometimes with probability epislon, we're going to choose a random move instead.


#   E - greedy -
# - Set E equal to how often we want to move randomly
# - With probability 1 - E, choose estimated best move.
# - With probability E, choose random move.


# So everytime we're faced with the ability to take an action, sometimes we're gonna choose the best move,
#sometimes we're just gonna choose a random move.

# This type of algorithm can be quite powerful in a reinforcement learning context.

# By not always just choosing the best possible move right now, but sometimes, especially eariler on, allowing
#ourselves to make random moves that allow us to explore various different possible states and actions more,
#and maybe over time we might decrease our value of episolon, more often choosing the best move after we're
#more confident we've explored what all of the possibilities actually are.

# So we can put this into practice.
