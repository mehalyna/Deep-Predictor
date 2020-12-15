import GA
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
import os
import random
import concurrent.futures
import shutil
import tkinter as tk
from tkinter import messagebox
# manages project libraries


sportNameRecord = "DEFAULT"
# sets a global variable that stores the selected sport name


def startNeuralNet(evalType, mode, testAmount, homeTeamID, awayTeamID, loadTxt):  # starts neural network based on given configurations

    if evalType:  # if specified, a new model will be trained
        clearDir("models\\training")

        if mode == 0:  # standard model train
            iterNum = 10
            schemaConst = 0.2
            learningRate = 0.1
            accumulatorVal = 0.1

            standardTrain(True, testAmount, homeTeamID, awayTeamID, loadTxt, iterNum, schemaConst, learningRate, accumulatorVal, "", "models\\training\\BASE")
            name = "BASE"

        elif mode == 1:  # selective train (random selection hyper-parameter optimization)
            modelNum, iterNum, schemaConst, learningRate, accumulatorVal = selectiveTrain(True, testAmount, homeTeamID, awayTeamID, loadTxt, 10, 10, True, None)[1:]
            name = "BASE" + str(modelNum)

        elif mode == 2:  # genetic algorithm (normal size)
            modelNum, iterNum, schemaConst, learningRate, accumulatorVal = GATrain(True, testAmount, homeTeamID, awayTeamID, loadTxt, 10, 10, 10)[1:]
            name = "BASE" + str(modelNum)

        else:  # genetic algorithm (extra)
            modelNum, iterNum, schemaConst, learningRate, accumulatorVal = GATrain(True, testAmount, homeTeamID, awayTeamID, loadTxt, 20, 15, 20)[1:]
            name = "BASE" + str(modelNum)

        saveModel(name, iterNum, schemaConst, learningRate, accumulatorVal)

    if homeTeamID is not None and awayTeamID is not None:  # if home and away team IDs provided, a prediction is made
        results, accuracy, probableAccuracy = standardTrain(False, testAmount, homeTeamID, awayTeamID, loadTxt, None, None, None, None, "", None)

        if results is None and accuracy is None and probableAccuracy is None:
            results = "\n"
            accuracy = ""
            probableAccuracy = ""
        else:
            results = results["probabilities"]
            accuracy = accuracy["accuracy"]

    else:  # if IDs are not provided, no prediction is made
        results = ""
        accuracy = ""
        probableAccuracy = ""

    f = open("MINFO.txt", "w")
    f.write(str(results) + "\n" + str(accuracy) + "\n" + str(probableAccuracy))
    f.close()
    # writes prediction data to file


def saveModel(name, iterNum, schemaConst, learningRate, accumulatorVal):  # saves trained model
    dirName = os.path.join("models\\saved", sportNameRecord)

    try:  # makes or clears a directory
        os.mkdir(dirName)
    except:
        clearDir(dirName)

    try:  # attempts to save mode, data
        origPath = os.path.join("models\\training", name)
        shutil.move(origPath, os.path.join(dirName, "MODEL"))  # experimental

        f = open(os.path.join(dirName, "schema.txt"), "w")
        f.write(str(iterNum) + "\n" + str(schemaConst) + "\n" + str(learningRate) + "\n" + str(accumulatorVal))
        f.close()

        clearDir("models\\training")
        clearDir("norm")

    except:  # displays an error if model data cannot be saved
        tk.messagebox.showerror("ERROR: Model Save Error", "Unknown Error Attempting to Save Model")


def standardTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, iterNum, schemaConst, learningRate, accumulatorVal, normFileAddition, modelName):  # trains standard neural network
    global sportNameRecord

    if not evalType:  # if specified, some data is gathered from a model save file and the database
        connection = sqlite3.connect("myData.db")
        crsr = connection.cursor()

        try:
            sportNameRecord = str(crsr.execute("SELECT Name FROM SPORTS WHERE Selector=1").fetchone()[0])
        except:
            sportNameRecord = "DEFAULT"

        connection.close()

        dirName = os.path.join("models\\saved", sportNameRecord)

        if not os.path.isdir(dirName):  # if no model exists, an error message is displays, and the function returns
            tk.messagebox.showerror("ERROR: No Model Found", "You must train a model before making predictions")
            return None, None, None

        modelName = os.path.join(dirName, "MODEL")

        f = open(os.path.join(dirName, "schema.txt"), "r")
        schema = f.readlines()
        f.close()

        iterNum = int(float(schema[0].rstrip("\n")))
        schemaConst = float(schema[1].rstrip("\n"))
        learningRate = float(schema[2].rstrip("\n"))
        accumulatorVal = float(schema[3])
        # gets schema from file

    mainSet, data, columns, sportName = getMainSet(iterNum)
    sportNameRecord = sportName
    # gets main set of data

    mainSet = normalize(mainSet, normFileAddition, True)
    # normalizes main set

    np.random.shuffle(mainSet)
    parseNum = int(len(mainSet) * testAmount)

    loadTxt.set("Compiling...")

    testSet = mainSet[0: parseNum]
    trainingSet = mainSet[parseNum:]

    if homeTeamID is not None and awayTeamID is not None:  # if valid IDs are given, a set of data is gathered to make a prediction
        evalSet = getEvalSet(data, homeTeamID, awayTeamID, iterNum, columns)
        evalSet = normalize(evalSet, normFileAddition, False)
    else:
        evalSet = None

    columns = getColumns(columns)

    model = createModel(evalType, trainingSet, columns, schemaConst, learningRate, accumulatorVal, modelName, loadTxt)  # turn numpy to datasets first
    # creates a model

    loadTxt.set("Calculating Predictions...")

    results, metrics, probableAccuracy = runModel(model, testSet, evalSet, columns)
    # runs model

    return results, metrics, probableAccuracy


def selectiveTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, numOfModels, trainSelectNum, mode, schemaData):  # trains many neural networks, and selects the bets schema and model
    threads = []
    schema = np.empty([numOfModels, 4])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads.append(executor.submit(standardTrain, evalType, testAmount, homeTeamID, awayTeamID, loadTxt, 10, 0.2, 0.1, 0.1, "0", None))
        schema[0] = np.array([10, 0.2, 0.1, 0.1])

        for x in range(1, numOfModels):  # randomizes/initializes schema
            if schemaData is None:
                iterNum = int(random.random() * 30) + 5
                schemaConst = random.random() * 0.5
                learningRate = random.random() * 0.2
                accumulatorVal = random.random() * 0.2
            else:
                iterNum = schemaData[x][0]
                schemaConst = schemaData[x][1]
                learningRate = schemaData[x][2]
                accumulatorVal = schemaData[x][3]

            schema[x] = np.array([iterNum, schemaConst, learningRate, accumulatorVal])
            threads.append(executor.submit(standardTrain, evalType, testAmount, homeTeamID, awayTeamID, loadTxt, iterNum, schemaConst, learningRate, accumulatorVal, str(x), None))
            # adds functions to a thread, and caches schema

        results = []

        for t in threads:  # runs threads
            try:
                results.append(t.result()[1]["accuracy"])  # inspection for this is probably dumb
            except:
                results.append(0)

    bestSchema = schema[results.index(max(results))]
    # finds the best schema from trained models

    iterNum = int(bestSchema[0])
    schemaConst = bestSchema[1]
    learningRate = bestSchema[2]
    accumulatorVal = bestSchema[3]
    # extracts schema values from this list

    if mode:  # if specified, the model will run another function to train a set of models as a final selection
        results = massTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, trainSelectNum, iterNum, schemaConst, learningRate, accumulatorVal)
        accuracy = max(results)
        modelNum = results.index(accuracy)

        return accuracy, modelNum, iterNum, schemaConst, learningRate, accumulatorVal

    return schema, results


def massTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, trainSelectNum, iterNum, schemaConst, learningRate, accumulatorVal):  # runs a set of models
    threads = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        for y in range(trainSelectNum):  # creates a bunch of threads with given schema
            threads.append(executor.submit(standardTrain, evalType, testAmount, homeTeamID, awayTeamID, loadTxt, iterNum, schemaConst, learningRate, accumulatorVal, str(y), os.path.join("models\\training", "BASE" + str(y))))

        results = []

        for t in threads:  # runs threads
            try:
                results.append(t.result()[1]["accuracy"])
            except:
                results.append(0)

    return results


def GATrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, generationCount, generationSize, finalSelectionCount):  # trains neural network using a genetic algorithm
    population, fitnessValues = selectiveTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, generationSize, 0, False, None)
    numOfParents = int(generationSize / 3) + 1
    # gets initial population information

    for gen in range(generationCount):
        loadTxt.set("Running Genetic Evaluation...")

        parents = GA.getParents(population, fitnessValues, numOfParents)
        offspring = GA.breed(parents, generationSize, 4)
        offspring = GA.crossover(offspring, 0.4, 4)
        population = GA.mutate(offspring, 0.4, 4)
        # runs through the processes of the genetic algorithm to generate a new generation

        fitnessValues = selectiveTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, generationSize, 0, False, population)[1]
        # generates fitness values based on this next generation

    bestSchema = population[fitnessValues.index(max(fitnessValues))]
    # finds the best schema from trained models

    iterNum = bestSchema[0]
    schemaConst = bestSchema[1]
    learningRate = bestSchema[2]
    accumulatorVal = bestSchema[3]
    # extracts schema values from this list

    results = massTrain(evalType, testAmount, homeTeamID, awayTeamID, loadTxt, finalSelectionCount, iterNum, schemaConst, learningRate, accumulatorVal)
    accuracy = max(results)
    modelNum = results.index(accuracy)
    # trains a final set of models with the best schema and finds the best one

    return accuracy, modelNum, iterNum, schemaConst, learningRate, accumulatorVal


def runModel(myModel, testSet, evalSet, columns):  # runs neural network
    testDict = {}
    evalDict = {}

    for z in columns:  # initializes keys of dictionary
        testDict[z] = []

    for x in testSet:  # adds data to the dictionaries
        for y in range(len(x)):
            testDict[columns[y]].append(x[y])

    for n in columns:  # converts to numpy arrays
        testDict[n] = np.asarray(testDict[n])

    testResultSet = testDict["Result"]
    testResultSet = testResultSet.astype(np.int32)
    del testDict["Result"]
    # isolates results values

    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=testDict,
        y=testResultSet,
        num_epochs=1,
        shuffle=False
    )
    # formats input to feed model

    metrics = myModel.evaluate(input_fn=test_input_fn)
    # evaluates accuracy based on test inputs

    testPredictions = list(myModel.predict(input_fn=test_input_fn))
    # generates a list of predictions for test inputs

    probableAccuracy = np.zeros(2)

    if evalSet is not None:  # if there is a set to evaluate, it will be passed through the model
        for y in range(1, len(evalSet) + 1):
            evalDict[columns[y]] = np.array([evalSet[y - 1]])

        eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x=evalDict,
            num_epochs=1,
            shuffle=False
        )
        # creates an input

        prediction = myModel.predict(input_fn=eval_input_fn)
        prediction = next(prediction)
        # makes a prediction

        predictionProbabilities = sorted(prediction["probabilities"])
        predictionRange = predictionProbabilities[2] - predictionProbabilities[1]

        for x in range(len(testResultSet)):  # finds probable accuracy value (confidence in prediction)
            testProbabilities = testPredictions[x]["probabilities"]
            testProbabilities.sort()

            if predictionRange - 0.2 < testProbabilities[2] - testProbabilities[1] < predictionRange + 0.2:   # finds average accuracy in the range around other data with similar probabilities
                if testResultSet[x] == testPredictions[x]["class_ids"][0]:
                    probableAccuracy[0] += 1
                else:
                    probableAccuracy[1] += 1

        try:
            probableAccuracy = probableAccuracy[0] / (probableAccuracy[0] + probableAccuracy[1])
        except:
            probableAccuracy = 0

    else:
        prediction = None
        probableAccuracy = None

    return prediction, metrics, probableAccuracy


def createModel(evalType, trainingSet, columns, schemaConst, learningRate, accumulatorVal, modelName, loadTxt):  # creates a neural network that can be run
    trainingDict = {}
    hiddenLayers = []
    columnLen = int(schemaConst * len(columns))

    for z in columns:  # initializes keys of dictionary
        trainingDict[z] = []

    while columnLen > 1:  # determines the schema for the model's hidden layers
        hiddenLayers.append(columnLen)
        columnLen = int(schemaConst * columnLen)

    for x in trainingSet:  # adds data to the dictionaries
        for y in range(len(x)):
            trainingDict[columns[y]].append(x[y])

    for n in columns:  # converts to numpy arrays
        trainingDict[n] = np.asarray(trainingDict[n])

    resultSet = trainingDict["Result"]
    resultSet = resultSet.astype(np.int32)
    del trainingDict["Result"]
    # isolates results values

    featureColumns = []
    testColumns = columns[1:]

    for x in testColumns:
        featureColumns.append(tf.feature_column.numeric_column(key=x))

    myModel = tf.estimator.DNNClassifier(
        model_dir=modelName,
        hidden_units=hiddenLayers,
        feature_columns=featureColumns,
        n_classes=3,
        activation_fn=tf.nn.leaky_relu,
        optimizer=tf.optimizers.Adagrad(
            learning_rate=learningRate,
            initial_accumulator_value=accumulatorVal
        )
    )
    # initializes model/loads from directory

    if evalType:  # if specified, the model will undergo training
        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x=trainingDict,
            y=resultSet,
            batch_size=500,
            num_epochs=None,
            shuffle=True
        )
        # creates input

        loadTxt.set("Initializing Model...")

        myModel.train(input_fn=train_input_fn, steps=1000)
        # trains model

    return myModel


def getColumns(columns):  # gets column names
    columns = columns[2:]
    homeColumns = []
    awayColumns = []

    for x in range(len(columns)):  # creates home and away team columns
        homeColumns.append("HomeTeam" + columns[x])
        awayColumns.append("AwayTeam" + columns[x])

    homeColumns.extend(awayColumns)
    columns = ["Result", "HomeWins", "HomeLosses", "HomeDraws", "AwayWins", "AwayLosses", "AwayDraws"] + homeColumns
    # combines home and away team columns, and adds extra labels

    return columns


def getEvalSet(rawData, homeTeamID, awayTeamID, iterNum, columns):  # gets the dataset that needs to be evaluated (for prediction)
    homeDataSet = np.zeros(len(columns) - 2)
    awayDataSet = np.zeros(len(columns) - 2)

    homeResults = np.zeros(3)
    awayResults = np.zeros(3)

    homeIter = 0
    awayIter = 0

    for teamData in rawData:  # processes raw data
        teamData = teamData[1:]

        if teamData[0] == homeTeamID and homeIter < iterNum:  # uses last "n" games for home team

            for x in range(2, len(columns)):
                homeDataSet[x - 2] += teamData[x]

            if teamData[2] > teamData[3]:
                homeResults[0] += 1
            elif teamData[2] < teamData[3]:
                homeResults[1] += 1
            else:
                homeResults[2] += 1

            homeIter += 1

        elif teamData[1] == awayTeamID and awayIter < iterNum:  # uses last "n" games for away team

            for y in range(2, len(columns)):
                awayDataSet[y - 2] += teamData[y]

            if teamData[2] < teamData[3]:
                awayResults[0] += 1
            elif teamData[2] > teamData[3]:
                awayResults[1] += 1
            else:
                awayResults[2] += 1

            awayIter += 1

        if homeIter >= iterNum and awayIter >= iterNum:
            break

    evalSet = np.concatenate((homeResults, awayResults, homeDataSet, awayDataSet), axis=None)

    return evalSet


def getMainSet(iterNum):  # gets the main dataset used to train and evaluate the model
    connection = sqlite3.connect("myData.db")
    crsr = connection.cursor()

    crsr.execute("SELECT * FROM GAMES")
    columns = list(map(lambda x: x[0], crsr.description))[1:]
    data = pd.read_sql("SELECT * FROM GAMES", connection)

    try:  # gets sport information
        selectedSport = str(crsr.execute("SELECT Name FROM SPORTS WHERE Selector=1").fetchone()[0])
        crsr.execute("SELECT * FROM " + selectedSport)
        columns.extend(list(map(lambda x: x[0], crsr.description))[1:])
        data = data.merge(pd.read_sql("SELECT * FROM " + selectedSport, connection))
    except:
        selectedSport = "DEFAULT"

    data = data.to_numpy()
    teamsIdDict = {}
    IDs = crsr.execute("SELECT ID FROM TEAMS").fetchall()

    for ID in range(len(IDs)):
        teamsIdDict[IDs[ID][0]] = ID

    teamCount = crsr.execute("SELECT COUNT(*) FROM TEAMS").fetchone()[0]
    connection.close()

    homeTempStorage = np.zeros((teamCount, iterNum, len(columns)))
    awayTempStorage = np.zeros((teamCount, iterNum, len(columns)))
    homeTeamIterCount = np.zeros(teamCount)
    awayTeamIterCount = np.zeros(teamCount)
    mainSet = np.empty((0, 2 * len(columns) + 3))
    # holds sets of information about the last "n" home/away games

    for datList in reversed(data):
        datList = datList[1:]

        homeIdIndex = teamsIdDict[datList[0]]
        awayIdIndex = teamsIdDict[datList[1]]
        result = [2]

        # attempts to create a data point from game (if enough previous data available to do so)
        if homeTeamIterCount[homeIdIndex] >= iterNum and awayTeamIterCount[awayIdIndex] >= iterNum:
            homeTestSet = np.zeros(len(columns) - 2)
            awayTestSet = np.zeros(len(columns) - 2)

            homeResults = np.zeros(3)  # 0 = won, 1 = lost, 2 = draw
            awayResults = np.zeros(3)

            for y in range(iterNum):  # gets data from the last "n" home/away games
                for x in range(2, len(columns)):
                    homeTestSet[x - 2] += homeTempStorage[homeIdIndex][y][x]
                    awayTestSet[x - 2] += awayTempStorage[awayIdIndex][y][x]

                if homeTempStorage[homeIdIndex][y][2] > homeTempStorage[homeIdIndex][y][3]:
                    homeResults[0] += 1
                elif homeTempStorage[homeIdIndex][y][2] < homeTempStorage[homeIdIndex][y][3]:
                    homeResults[1] += 1
                else:
                    homeResults[2] += 1

                if awayTempStorage[awayIdIndex][y][2] < awayTempStorage[awayIdIndex][y][3]:
                    awayResults[0] += 1
                elif awayTempStorage[homeIdIndex][y][2] > awayTempStorage[awayIdIndex][y][3]:
                    awayResults[1] += 1
                else:
                    awayResults[2] += 1

                if datList[2] > datList[3]:
                    result = [0]
                elif datList[2] < datList[3]:
                    result = [1]

            stackArr = np.concatenate((result, homeResults, awayResults, homeTestSet, awayTestSet), axis=None)
            mainSet = np.vstack([mainSet, stackArr])

        for x in reversed(range(iterNum - 1)):  # gathering data from game
            homeTempStorage[homeIdIndex][x + 1] = homeTempStorage[homeIdIndex][x]
            awayTempStorage[awayIdIndex][x + 1] = awayTempStorage[awayIdIndex][x]

        homeTempStorage[homeIdIndex][0] = datList
        awayTempStorage[awayIdIndex][0] = datList
        homeTeamIterCount[homeIdIndex] += 1
        awayTeamIterCount[awayIdIndex] += 1

    return mainSet, data, columns, selectedSport


def normalize(rawSet, normFileAddition, mode):  # normalizes the datasets

    if mode:  # if specified, data from a set is normalized
        setLen = len(rawSet[0])
        maxArr = np.empty(setLen)
        minArr = np.empty(setLen)

        for x in rawSet:
            for y in range(setLen):
                if x[y] is None:
                    maxArr[y] = x[y]
                    minArr[y] = x[y]
                else:
                    if x[y] > maxArr[y]:
                        maxArr[y] = x[y]

                    if x[y] < minArr[y]:
                        minArr[y] = x[y]

        maxArr = maxArr[1:]
        minArr = minArr[1:]

        f = open(os.path.join("norm", "NORMDATA" + normFileAddition + ".txt"), "w")

        for z in range(setLen - 1):
            f.write(str(maxArr[z]) + " " + str(minArr[z]) + "\n")

        for m in range(len(rawSet)):
            for n in range(setLen - 1):
                rawSet[m][n + 1] = (rawSet[m][n + 1] - minArr[n]) / (maxArr[n] - minArr[n])

        f.close()

    else:  # if specified, from a set is normalized using data from a file
        f = open(os.path.join("norm", "NORMDATA" + normFileAddition + ".txt"), "r")
        fileVals = f.readlines()

        maxArr = []
        minArr = []

        for x in range(len(rawSet)):
            indVals = fileVals[x].split()

            maxArr.append(float(indVals[0]))
            minArr.append(float(indVals[1]))

            rawSet[x] = (rawSet[x] - minArr[x]) / (maxArr[x] - minArr[x])

    return rawSet


def clearDir(folder):  # clears a directory
    if os.path.isdir(folder):

        for filename in os.listdir(folder):  # finds all files and subdirectories, and removes them
            file_path = os.path.join(folder, filename)

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except:
                pass