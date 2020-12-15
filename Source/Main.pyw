import Computation
import threading
import time
import csv
import ntpath
import os
import sqlite3
import pandas as pd
import tkinter as tk
from sqlalchemy import create_engine
from shutil import copyfile
from tkinter import messagebox, filedialog, ttk
from tkinter import *
from tkinter.ttk import *
# manages project libraries


class PlaceholderEntry(ttk.Entry):  # defines a class for an input field with a placeholder when empty
    def __init__(self, container, placeholder, isFilled, *args, **kwargs):
        super().__init__(container, *args, style="Placeholder.TEntry", **kwargs)
        self.placeholder = placeholder
        self.isFilled = isFilled

        self.insert("0", self.placeholder)
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

    def _clear_placeholder(self, e):  # clears placeholder when field has text or is selected
        if self["style"] == "Placeholder.TEntry":
            self.delete("0", "end")
            self["style"] = "TEntry"
            self.isFilled = True

    def _add_placeholder(self, e):  # adds placeholder when no text is in field and it is not selected
        if not self.get():
            self.insert("0", self.placeholder)
            self["style"] = "Placeholder.TEntry"
            self.isFilled = False

    def getIsFilled(self):  # returns whether or not the field is currently filled
        return self.isFilled


def sportInfoSetup():  # initializes the database that holds sport data

    if not os.path.isfile("INIT.txt"):  # if the initialization file doesn't exist, one is created
        e = open("INIT.txt", "w")
        e.write("# This file is responsible for initializing the different attributes associated with each sport\n"
                "# To initialize a new sport, type its name, ID, and total number of additional attributes (not including game score) in the below format -\n"
                '# "SPORT_NAME" ID#: NUMBER\n'
                '# For example, "FOOTBALL" 1: 4\n'
                '# To initialize a new sport with no additional attributes, you can also merely type the data as such: "SPORT_NAME" ID#\n'
                "# Attribute information should contain no spaces, and note that if a sport NAME is solely a numerical value, it will be overridden by ID value\n")
        e.close()

    f = open("INIT.txt", "r")

    connection = sqlite3.connect('myData.db')
    connection.row_factory = lambda cursor, row: row[0]
    crsr = connection.cursor()

    selectorID = crsr.execute("SELECT ID FROM SPORTS WHERE Selector=1").fetchone()
    selectorName = crsr.execute("SELECT Name FROM SPORTS WHERE Selector=1").fetchone()
    crsr.execute("DELETE FROM SPORTS")
    z = crsr.execute("SELECT name FROM sqlite_master WHERE type ='table'").fetchall()
    # resets database information

    for y in z:  # removes all sport-specific tables in database
        if y != "TEAMS" and y != "SPORTS" and y != "GAMES" and y != selectorName:
            try:
                crsr.execute("DROP TABLE " + y)
            except:
                pass

    try:  # attempts to read data from the file to the database
        for x in f:
            if x.replace(" ", "") == "\n" or x[0] == "#":
                continue
            else:
                name = re.search('"(.+?)"', x).group(1)

                id = x[(len(name) + 2):]

                try:
                    id = re.search(' (.+?):', id).group(1)
                except:
                    try:
                        id = re.search(' (.+?)\n', id).group(1)
                    except:
                        id = re.search(' (.+?)', id).group(1)

                try:
                    attNum = int(re.search(':(.+?)\n', x).group(1).strip())
                except:
                    try:
                        attNum = int(x.split(":", 1)[1].strip())
                    except:
                        attNum = 0

                attributes = []

                for z in range(attNum):
                    attributes.append("ATT" + str(z))

                name = name.replace(" ", "_")
                tableInit = "CREATE TABLE " + name + " (GameID INTEGER PRIMARY KEY UNIQUE NOT NULL,"

                for n in attributes:
                    tableInit = tableInit + n + " DECIMAL,"

                tableInit = tableInit[:-1]
                tableInit += ")"

                if len(attributes) > 0 and name != selectorName:
                    crsr.execute(tableInit)

                crsr.execute("INSERT INTO SPORTS (ID, Name) VALUES (?,?)", (id, name))

        crsr.execute("UPDATE SPORTS SET Selector=1 WHERE ID=?", (selectorID,))

    except:  # if this fails, an error message is displayed
        tk.messagebox.showerror("ERROR: partial or complete sport initialization failure", "An error occurred when attempting to initialize the database.  Make sure the sport data initialization file is correctly formatted")

    connection.commit()
    connection.close()
    f.close()


def teamRetrieval():  # gets team data
    global team1
    global team2

    connection = sqlite3.connect('myData.db')
    connection.row_factory = lambda cursor, row: row[0]
    crsr = connection.cursor()

    teams = crsr.execute("SELECT Name FROM TEAMS").fetchall()
    connection.close()
    # gets team data from database

    team1.set("")
    team2.set("")
    team1["values"] = teams
    team2["values"] = teams
    # sets global team values to the found values


def initializeDatabase(database, fileData, mode):  # initializes the database from a file
    fileName = fileData.get()

    if fileName != "" and fileData.getIsFilled():

        if fileName[-4:].lower() != ".txt" and fileName[-4:].lower() != ".csv":
            fileName += fileExtension

        if mode:  # if specified, team and game tables are emptied in the database
            connection = sqlite3.connect(database)
            crsr = connection.cursor()
            crsr.execute("DELETE FROM TEAMS")
            crsr.execute("DELETE FROM GAMES")
            connection.commit()
            connection.close()

        if fileName[-4:].lower() == ".csv":  # the file type is checked and the appropriate function is run
            readCsvFile(fileName, database, mode)
        else:
            readTxtFile(fileName, database, mode)

        readTxt.delete(0, END)

    else:  # if no file is entered, an error message is displayed
        tk.messagebox.showerror("No File Entered", "Enter a file name in the field to access team data options")


def readCsvFile(fileName, database, mode):  # writes to the database from a CSV file
    global savedSportName1
    global savedSportName2
    sportName = "DEFAULT"

    try:  # tries writing to the database
        data = pd.read_csv(os.path.join("user_data", fileName))
        cut = len(data.index) - 2
        sportInfo = data[cut:]

        for x in sportInfo:
            try:
                if sportInfo.loc[cut + 1, x][:6].upper() == "SPORT=":
                    sportInfo = sportInfo.loc[cut + 1, x][6:]
            except:
                pass

        if isinstance(sportInfo, str):
            data = data[:cut]

            try:  # gets sport information
                connection = sqlite3.connect(database)
                crsr = connection.cursor()
                sportName = crsr.execute("SELECT Name FROM SPORTS WHERE ID=?", (sportInfo,)).fetchone()
                sportName = sportName[0]
                connection.close()
            except:
                sportName = sportInfo

            if mode:
                savedSportName1 = sportName
            else:
                savedSportName2 = sportName

        connection = sqlite3.connect(database)
        crsr = connection.cursor()

        if mode:
            crsr.execute("UPDATE SPORTS SET Selector=0")

        try:
            if mode:
                crsr.execute("DELETE FROM " + sportName)

            crsr.execute("UPDATE SPORTS SET Selector=1 WHERE Name=?", (sportName,))
        except:
            pass

        teams = list(data.Home_Team_Name)
        teams.extend(list(data.Away_Team_Name))
        teams = list(dict.fromkeys(teams))

        i = 0
        teamsDict = {}

        for t in teams:  # iterates through options and manages team and ID values
            if not mode:
                currentNames = crsr.execute("SELECT Name FROM TEAMS").fetchall()

                if (t,) not in currentNames:
                    isID = (0,)

                    while isID:
                        isID = crsr.execute("SELECT * FROM TEAMS WHERE ID=?", (i,)).fetchone()
                        i += 1

                    i -= 1

                    teamsDict[i] = t
                    data = data.replace(t, i)
                else:
                    data = data.replace(t, crsr.execute("SELECT ID FROM TEAMS WHERE Name=?", (t,)).fetchone()[0])
            else:
                teamsDict[i] = t
                data = data.replace(t, i)

            i += 1

        for x in teamsDict.keys():
            crsr.execute("INSERT INTO TEAMS (Name, ID) VALUES (?,?)", (teamsDict[x], x))

        if not mode:
            isGameID = (0,)
            gameID = 0

            while isGameID:
                isGameID = crsr.execute("SELECT * FROM GAMES WHERE GameID=?", (gameID,)).fetchone()
                gameID += 1

            gameID -= 1
            data = data.rename(index=lambda x: x + gameID)

        connection.commit()
        connection.close()

        mainData = data[["Home_Team_Name", "Away_Team_Name", "Home_Team_Score", "Away_Team_Score"]].copy()
        del data["Home_Team_Name"]
        del data["Away_Team_Name"]
        del data["Home_Team_Score"]
        del data["Away_Team_Score"]
        # splits dataset into main and added data values

        mainData.columns = ["HomeID", "AwayID", "HomeScore", "AwayScore"]

        engine = create_engine("sqlite:///" + database, echo=False)
        connection = engine.connect()

        mainData.to_sql("GAMES", connection, if_exists='append', index_label='GameID')

        try:  # inserts data into the database
            data.to_sql(sportName, connection, if_exists='append', index_label='GameID')
        except:
            pass

        connection.close()
        teamRetrieval()

        if database == "myData.db":
            tk.messagebox.showinfo("SUCCESS", "Database successfully updated")

    except:  # if there is an error, different error massages are displayed based on function inputs
        if database == "myData.db":
            tk.messagebox.showerror("ERROR: partial or complete initialization failure", "An error occurred when attempting to initialize the database.  Make sure the file with the given name is present, it is the correct file type (.txt or .csv), and the contents are correctly formatted")
        elif database == "tempData.db":
            tk.messagebox.showerror("ERROR: could not create copy", "Make sure the file you are trying to copy is the correct file type (.txt or .csv), and is formatted correctly")
        else:
            tk.messagebox.showerror("Merge Error", "Make sure the files you are trying to merge are the correct file type (.txt or .csv), are formatted correctly, and both contain data for the same sport")


def readTxtFile(fileName, database, mode):  # writes to the database from a TXT file
    global savedSportName1
    global savedSportName2
    readIn = 0
    gameID = 0
    numOfElems = 0
    sportName = "DEFAULT"
    sportAttributeStr = ""
    qMarks = ""
    usedIDs = []
    oldIDsDict = {}

    connection = sqlite3.connect(database)
    crsr = connection.cursor()

    if not mode:  # if selected, finds the game ID of a specific value
        isGameID = (0,)

        while isGameID:
            isGameID = crsr.execute("SELECT * FROM GAMES WHERE GameID=?", (gameID,)).fetchone()
            gameID += 1

        gameID -= 1

    try:  # attempts to write to the database from a TXT file
        f = open(os.path.join("user_data", fileName), "r")

        for x in f:

            if x.replace(" ", "") == "\n" or x[0] == "#":  # ignores comments
                continue
            elif x[:6].upper() == "SPORT=":
                sportVal = x[6:]

                try:  # gets sport name
                    sportName = crsr.execute("SELECT Name FROM SPORTS WHERE ID=?", (sportVal,)).fetchone()
                    sportName = sportName[0]
                except:
                    sportName = sportVal
                    sportName = sportName.rstrip("\n")

                if mode:
                    savedSportName1 = sportName
                else:
                    savedSportName2 = sportName

                try:  # gets current table information
                    crsr.execute("PRAGMA table_info(%s)" % sportName)
                    numOfElems = len(crsr.fetchall())

                    crsr.execute("SELECT * FROM " + sportName)
                    columnNames = list(map(lambda x: x[0], crsr.description))

                    if mode:
                        crsr.execute("DELETE FROM " + sportName)

                except:
                    columnNames = ()

                if mode:
                    crsr.execute("UPDATE SPORTS SET Selector=0")

                crsr.execute("UPDATE SPORTS SET Selector=1 WHERE Name=?", (sportName,))

                for v in columnNames:  # sets up strings to be used in prepared statements later
                    sportAttributeStr += v + ","
                    qMarks += "?,"

                sportAttributeStr = sportAttributeStr[:-1]
                qMarks = qMarks[:-1]

            elif x.upper() == "--TEAMS--\n":
                readIn = 1
            elif x.upper() == "--GAMES--\n":
                readIn = 2
            elif readIn == 1:
                name = re.search('"(.+?)"', x).group(1)
                id = x[(len(name) + 3):]

                if not mode:
                    isID = (0,)
                    id = int(id)
                    oldID = id

                    currentNames = crsr.execute("SELECT Name FROM TEAMS").fetchall()

                    if (name,) not in currentNames:  # gets team info based on ID or name
                        while isID or id in usedIDs:
                            isID = crsr.execute("SELECT * FROM TEAMS WHERE ID=?", (id,)).fetchone()
                            id += 1

                        id -= 1
                        crsr.execute("INSERT INTO TEAMS (Name, ID) VALUES (?,?)", (name, id))

                        usedIDs.append(id)
                        oldIDsDict[oldID] = id
                    else:
                        oldIDsDict[id] = crsr.execute("SELECT ID FROM TEAMS WHERE Name=?", (name,)).fetchone()[0]

                else:
                    crsr.execute("INSERT INTO TEAMS (Name, ID) VALUES (?,?)", (name, id))

            elif readIn == 2:

                if numOfElems > 0:
                    insertValues = x.split("_")
                    simpleValues = insertValues[0].split()

                    if not mode:
                        simpleValues[0] = oldIDsDict[int(simpleValues[0])]
                        simpleValues[1] = oldIDsDict[int(simpleValues[1])]

                    complexValues = insertValues[1].split()
                    complexValues.insert(0, gameID)
                    crsr.execute("INSERT INTO " + sportName + " (" + sportAttributeStr + ") VALUES (" + qMarks + ")", complexValues)
                    # executed prepared statements to insert data into database

                else:
                    simpleValues = x.split()

                simpleValues.insert(0, gameID)
                crsr.execute("INSERT INTO GAMES (GameID, HomeID, AwayID, HomeScore, AwayScore) VALUES (?,?,?,?,?)", simpleValues)
                gameID += 1
                # inserts base values into a separate table

        connection.commit()
        connection.close()
        f.close()

        teamRetrieval()

        if database == "myData.db":  # if data is written to this database, a message is displayed
            tk.messagebox.showinfo("SUCCESS", "Database successfully updated")

    except:  # if errors occur, messages are displayed to the user
        if database == "myData.db":
            tk.messagebox.showerror("ERROR: partial or complete initialization failure", "An error occurred when attempting to initialize the database.  Make sure the file with the given name is present, it is the correct file type (.txt or .csv), and the contents are correctly formatted")
        elif database == "tempData.db":
            tk.messagebox.showerror("ERROR: could not create copy", "Make sure the file you are trying to copy is the correct file type (.txt or .csv), and is formatted correctly")
        else:
            tk.messagebox.showerror("Merge Error", "Make sure the files you are trying to merge are the correct file type (.txt or .csv), are formatted correctly, and both contain data for the same sport")


def openFile():  # allows user to open a data file
    fileName = readTxt.get()

    if fileName != "" and readTxt.getIsFilled():

        if fileName[-4:].lower() != ".txt" and fileName[-4:].lower() != ".csv":
            fileName += fileExtension

        if os.path.isfile(os.path.join("user_data", fileName)):  # if the file exists, it is opened
            root.withdraw()
            os.system('"' + os.path.join("user_data", fileName) + '"')
            root.deiconify()
            root.lift()

        else:  # if the file does not exist, the user is prompted to create a new one
            newFilePrompt(fileName)

        readTxt.delete(0, END)

    else:  # if field is empty, a message is displayed
        tk.messagebox.showerror("No File Entered", "Enter a file name in the field to access team data options")


def newFilePrompt(fileName):  # allows users to create new data files
    fileCreationType = getFileCreationType()
    MsgBox = tk.messagebox.askquestion("Create New File?", f'A file with the name "{fileName}" does not yet exist.  Would you like to create a new file?', icon="question")
    # prompts users if they would like to create a new file

    if MsgBox == "yes":  # if they choose to do so, a new file is created
        f = open(os.path.join("user_data", fileName), "w")

        if fileName[-4:].lower() == ".csv":  # creates CSV file

            with open(os.path.join("user_data", fileName), mode="w", newline="") as fileWrite:  # opens CSV file
                fileWrite = csv.writer(fileWrite, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

                if fileCreationType == 1:  # defines minimal file template
                    fileWrite.writerow(["Home_Team_Name", "Away_Team_Name", "Home_Team_Score", "Away_Team_Score"])

                elif fileCreationType == 2:  # defines example file template
                    fileWrite.writerow(["Home_Team_Name", "Away_Team_Name", "Home_Team_Score", "Away_Team_Score", "att1", "att2", "att3"])
                    fileWrite.writerow(["Patriots", "Redskins", "7", "3", "1", "22", "3"])
                    fileWrite.writerow(["Browns", "Patriots", "10", "49", "4", "25", "7"])
                    fileWrite.writerow(["Giants", "49ers", "3", "12", "2", "34", "4"])
                    fileWrite.writerow(["Redskins", "Saints", "14", "3", "6", "12", "1"])
                    fileWrite.writerow(["", "", "", "", "", "", ""])
                    fileWrite.writerow(["SPORT=FOOTBALL", "", "", "", "", "", ""])

        else:  # creates TXT file

            if fileCreationType == 1:  # defines instructional file template
                f.write("SPORT=\n"
                        '# Specify sport by name or ID (Example: SPORT="FOOTBALL" or SPORT=1).  Set SPORT=DEFAULT or remove line for non-sport specific prediction\n\n'
                        "# Modify this document to change the data used by the prediction program\n"
                        '# Use the "#" symbol at the beginning of a line to write comments\n\n'
                        "--TEAMS--\n"
                        '# Write information here to add team data using the following format: "Team_Name" ID#\n\n'
                        "--GAMES--\n"
                        "# Write information here to add game data using the following format: Home_Team_ID# Away_Team_ID# Home_Team_Score Away_Team_Score\n"
                        '# Add a "_" and then sport-specific information if applicable (Example: 1 2 7 7_50 100, where 50 and 100 correspond to extra information for the specified sport)\n')

            elif fileCreationType == 2:  # defines example file template
                f.write("SPORT=FOOTBALL\n\n"
                        "--TEAMS--\n"
                        '"Patriots" 0\n'
                        '"Browns" 1\n'
                        '"Giants" 2\n'
                        '"Redskins" 3\n'
                        '"49ers" 4\n'
                        '"Saints" 5\n\n'
                        "--GAMES--\n"
                        "0 3 7 3_1 22 3\n"
                        "1 0 10 49_4 25 7\n"
                        "2 4 3 12_2 34 4\n"
                        "3 5 14 3_6 12 1\n")

        f.close()

        root.withdraw()
        os.system('"' + os.path.join("user_data", fileName) + '"')
        root.deiconify()
        root.lift()


def getFileCreationType():  # returns which file template to use

    try:  # attempts to read from settings and return value
        f = open("SETTINGS.txt", "r")
        settings = f.readlines()
        f.close()

        if settings[2] == "Blank":
            return 0
        elif settings[2] == "Title Values (DEFAULT)":
            return 1
        else:
            return 2

    except:  # returns a base value if no file exists or there was an error
        return 1


def deleteFile():  # allows users to delete data files
    fileName = readTxt.get()

    if fileName != "" and readTxt.getIsFilled():
        if fileName[-4:].lower() != ".txt" and fileName[-4:].lower() != ".csv":
            fileName += fileExtension

        if os.path.isfile(os.path.join("user_data", fileName)):  # if file found, the user is prompted for deletion
            deletePrompt(fileName)

        else:  # if no file is found, an error message is shown
            tk.messagebox.showerror("File Not Found", f'The file with the name "{fileName}" does not exist')

        readTxt.delete(0, END)

    else:  # if no file is entered, an error message is shown
        tk.messagebox.showerror("No File Entered", "Enter a file name in the field to access team data options")


def deletePrompt(fileName):  # confirms user deletion
    MsgBox = tk.messagebox.askquestion("Deletion Confirmation", f'You are about to delete the file with the name "{fileName}"\n\nAre you sure you would like to proceed?  This action cannot be undone.', icon="warning")

    if MsgBox == "yes":  # if confirmed, the file is deleted
        os.remove(os.path.join("user_data", fileName))
        tk.messagebox.showinfo("File Deleted", "File successfully deleted")


def listFiles():  # lists user data files
    fileList = os.listdir("user_data")
    fileNums = len(fileList)
    fileStr = ""

    for x in fileList:  # adds file names to a string
        fileStr += "[" + x + "]  "

    if fileNums != 1:  # displays the files
        tk.messagebox.showinfo(f"{fileNums} Files Found", fileStr)
    else:
        tk.messagebox.showinfo("1 File Found", fileStr)


def initAccess():  # allows users to access the initialization file
    root.withdraw()

    if os.path.isfile("INIT.txt"):  # opens file if it exists
        e = open("INIT.txt", "r")
        oldContents = e.read()
        e.close()

        os.system("INIT.txt")

    else:  # creates a new base file if it does not exist
        oldContents = "# This file is responsible for initializing the different attributes associated with each sport\n" \
                "# To initialize a new sport, type its name and attributes in the below format -\n" \
                '# "SPORT_NAME" ID#: Attribute#1 Attribute#2...\n' \
                '# For example, "FOOTBALL" 1: HomeYardDiff AwayYardDiff HomeTurnovers AwayTurnovers\n' \
                '# To initialize a new sport with no additional attributes, you can also merely type the data as such: "SPORT_NAME" ID#\n' \
                "# Attribute information should contain no spaces"

        f = open("INIT.txt", "w")
        f.write(oldContents)

        f.close()
        os.system("INIT.txt")

    g = open("INIT.txt")
    newContents = g.read()
    g.close()

    if oldContents != newContents:  # if anything is changed, the database is updated
        connection = sqlite3.connect("myData.db")
        crsr = connection.cursor()
        crsr.execute("UPDATE SPORTS SET Selector=0")
        crsr.execute("DELETE FROM TEAMS")
        connection.commit()
        connection.close()

    root.deiconify()
    sportInfoSetup()
    teamRetrieval()


def openAdvancedWindow():  # shows advanced program options
    global FT

    root.withdraw()

    advancedWindow = Toplevel(root)
    centerWindow(advancedWindow, 360, 350)
    advancedWindow.title("Advanced Options")
    advancedWindow.wm_iconbitmap("icon.ico")
    # creates new advanced options window

    adOp1 = Label(advancedWindow, text="Model Training Options:", font=("Arial", 9))
    adOp1.grid(row=0, column=0, padx=5, pady=(5, 0), stick=W)

    numSelect = Combobox(advancedWindow, state="readonly", values=("Quick Train (DEFAULT)", "Selective Train (slower)", "GA Deep Train (very slow)", "Extended GA Train (slowest)"), width=25)
    numSelect.grid(row=0, column=1, padx=5, pady=(5, 0), stick=W)

    fileFormatSelect = Combobox(advancedWindow, state="readonly", values=("Blank", "Title Values (DEFAULT)", "Titles + Example Values"), width=25)
    fileFormatSelect.grid(row=2, column=1, padx=5, stick=NW)

    adOp2 = Label(advancedWindow, text="Default File Type:", font=("Arial", 9))
    adOp2.grid(row=1, column=0, padx=5, pady=25, stick=W)

    F1 = Radiobutton(advancedWindow, text="TXT", variable=FT, value=1, command=lambda: saveSettings(numSelect.get(), fileFormatSelect.get()))
    F1.grid(row=1, column=1, padx=5, stick=W)

    F2 = Radiobutton(advancedWindow, text="CSV", variable=FT, value=2, command=lambda: saveSettings(numSelect.get(), fileFormatSelect.get()))
    F2.grid(row=1, column=1, padx=110, stick=W)

    adOp3 = Label(advancedWindow, text="New File Format:", font=("Arial", 9))
    adOp3.grid(row=2, column=0, padx=5, pady=(0, 25), stick=W)

    adOp4 = Label(advancedWindow, text="Copy/Convert File Type:", font=("Arial", 9))
    adOp4.grid(row=3, column=0, padx=5, pady=(0, 5), stick=W)

    oldFileField = PlaceholderEntry(advancedWindow, "Original File", False)
    oldFileField.configure(width=20)
    oldFileField.grid(row=4, column=0, columnspan=10, padx=5, stick=W)

    arrow = Label(advancedWindow, text=chr(8594), font=("Arial", 12))
    arrow.grid(row=4, column=0, columnspan=10, padx=140, stick=NW)

    newFileField = PlaceholderEntry(advancedWindow, "New File", False)
    newFileField.configure(width=20)
    newFileField.grid(row=4, column=0, columnspan=10, padx=169, stick=W)

    copy = tk.Button(advancedWindow, text="Copy", bg="lightgrey", command=lambda: copyFiles(oldFileField, newFileField))
    copy.grid(row=4, column=0, columnspan=10, padx=305, stick=NW)

    adOp5 = Label(advancedWindow, text="Merge Files:", font=("Arial", 9))
    adOp5.grid(row=5, column=0, padx=5, pady=(25, 5), stick=W)

    fileField1 = PlaceholderEntry(advancedWindow, "File #1", False)
    fileField1.configure(width=12)
    fileField1.grid(row=6, column=0, columnspan=10, padx=5, stick=W)

    plus = Label(advancedWindow, text="+", font=("Arial", 12))
    plus.grid(row=6, column=0, columnspan=10, padx=90, stick=W)

    fileField2 = PlaceholderEntry(advancedWindow, "File #2", False)
    fileField2.configure(width=12)
    fileField2.grid(row=6, column=0, columnspan=10, padx=110, stick=W)

    equals = Label(advancedWindow, text="=", font=("Arial", 12))
    equals.grid(row=6, column=0, columnspan=10, padx=195, stick=W)

    mergedFile = PlaceholderEntry(advancedWindow, "New File", False)
    mergedFile.configure(width=12)
    mergedFile.grid(row=6, column=0, columnspan=10, padx=215, stick=W)

    merge = tk.Button(advancedWindow, text="Merge", bg="lightgrey", command=lambda: mergeFiles(fileField1, fileField2, mergedFile))
    merge.grid(row=6, column=0, columnspan=10, padx=300, stick=NW)

    adOp6 = Label(advancedWindow, text="Import File:", font=("Arial", 9))
    adOp6.grid(row=7, column=0, padx=5, pady=(25, 5), stick=W)

    importButton = tk.Button(advancedWindow, text=f"Find {chr(128270)}", bg="lightgrey", command=lambda: getNewFile(advancedWindow))
    importButton.grid(row=8, column=0, padx=5, stick=W)

    resetDefaults = tk.Button(advancedWindow, text="Reset to Defaults", bg="lightgrey", command=lambda: resetToDefaults(numSelect, fileFormatSelect))
    resetDefaults.grid(row=8, column=1, padx=50, stick=W)

    getSettings(numSelect, fileFormatSelect)
    numSelect.bind("<<ComboboxSelected>>", lambda event: saveSettings(numSelect.get(), fileFormatSelect.get()))
    fileFormatSelect.bind("<<ComboboxSelected>>", lambda event: saveSettings(numSelect.get(), fileFormatSelect.get()))
    advancedWindow.protocol("WM_DELETE_WINDOW", lambda: close())
    # deals with advanced window closure

    advancedWindow.lift()

    def close():  # cleanly closes window and reopens base window when prompted to do so
        advancedWindow.destroy()
        root.deiconify()
        root.lift()


def copyFiles(oldFileField, newFileField):  # allows users to copy data files
    oldFile = oldFileField.get()
    newFile = newFileField.get()

    if (oldFile != "" and oldFileField.getIsFilled()) and (newFile != "" and newFileField.getIsFilled()):

        if oldFile[-4:].lower() != ".txt" and oldFile[-4:].lower() != ".csv":  # gets old file extension
            oldFile += fileExtension

        if newFile[-4:].lower() != ".txt" and newFile[-4:].lower() != ".csv":  # gets new file extension
            newFile += fileExtension

        if os.path.isfile(os.path.join("user_data", oldFile)) and not os.path.isfile(os.path.join("user_data", newFile)):

            try:  # attempts to copy file
                if oldFile[-4:].lower() == newFile[-4:].lower():
                    copyfile(os.path.join("user_data", oldFile), os.path.join("user_data", newFile))
                    tk.messagebox.showinfo("Copy Success", "File successfully copied")
                else:
                    copyfile("myData.db", "tempData.db")
                    initializeDatabase("tempData.db", oldFileField, True)

                    if newFile[-4:].lower() == ".txt":
                        writeToTxt(newFile, "tempData.db")
                    else:
                        writeToCsv(newFile, "tempData.db")

            except:  # if file copy fails, an error message is displayed
                tk.messagebox.showerror("Copy Error", "Copy could not be made.  Make sure the original file is correctly formatted.")

            try:
                os.remove("tempData.db")
            except:
                pass

        else:  # if the new file already exists, an error is shown
            tk.messagebox.showerror("File Name Error", "Make sure the file you are trying to copy exists and that the new file name is not already present.")

    else:  # if not all needed data is entered by users, an error is shown
        tk.messagebox.showerror("File Entry Error", "Enter data in all file fields to copy a file")

    oldFileField.delete(0, END)
    oldFileField.insert(0, "Original File")
    oldFileField['style'] = "Placeholder.TEntry"

    newFileField.delete(0, END)
    newFileField.insert(0, "New File")
    newFileField['style'] = "Placeholder.TEntry"


def getFileElems(database):  # gets sport elements from database
    connection = sqlite3.connect(database)
    crsr = connection.cursor()
    sportName = None
    sportInfo = []
    gameInfo = []
    specificInfo = []

    try:  # attempts to get sport name and information
        sportName = crsr.execute("SELECT Name FROM SPORTS WHERE Selector=1").fetchone()
        sportName = sportName[0]
        sportInfo = crsr.execute("SELECT * FROM " + sportName).fetchall()
    except:
        pass

    if sportName is None:
        sportName = "DEFAULT"

    teams = crsr.execute("SELECT * FROM TEAMS").fetchall()
    games = crsr.execute("SELECT * FROM GAMES").fetchall()
    connection.close()

    for y in games:
        gameInfo.append(y)

    for z in sportInfo:
        specificInfo.append(z)

    return sportName, gameInfo, specificInfo, teams
    # returns the values that it finds


def writeToTxt(file, database):  # writes data to a text file
    sportName, gameInfo, specificInfo, teams = getFileElems(database)

    f = open(os.path.join("user_data", file), "w")
    f.write("SPORT=" + sportName + "\n\n--TEAMS--\n")
    # writes main titles

    for x in teams:  # writes team data
        f.write('"' + x[1] + '" ' + str(x[0]) + "\n")

    f.write("\n--GAMES--\n")

    for a in range(len(gameInfo)):  # writes game data
        gamesStr = ""

        for b in range(4):
            gamesStr += str(gameInfo[a][b + 1]) + " "

        if len(specificInfo) > 0:
            gamesStr = gamesStr[:-1] + "_"

        try:
            for c in range(len(specificInfo[a]) - 1):
                gamesStr = gamesStr + str(specificInfo[a][c + 1]) + " "
        except:
            pass

        gamesStr = gamesStr[:-1] + "\n"
        f.write(gamesStr)

    f.close()

    if database == "tempData.db":  # shows messages based on the context of function call
        tk.messagebox.showinfo("Copy Success", "File successfully copied")
    else:
        tk.messagebox.showinfo("Merge Success", "Files successfully merged")


def writeToCsv(file, database):  # writes data to a CSV file
    sportName, gameInfo, specificInfo, teams = getFileElems(database)
    infoTitles = ["Home_Team_Name", "Away_Team_Name", "Home_Team_Score", "Away_Team_Score"]
    teamDict = {}
    mainList = []

    connection = sqlite3.connect(database)
    crsr = connection.cursor()

    for x in teams:  # creates a dictionary of teams to link team name to ID to build CSV
        teamDict[x[0]] = x[1]

    for y in range(len(gameInfo)):  # creates a list of game information
        try:
            newList = [teamDict[gameInfo[y][1]], teamDict[gameInfo[y][2]]]
            newList.extend(gameInfo[y][3:])
            newList.extend(list(specificInfo[y])[1:])
            mainList.append(newList)
        except:
            pass

    try:
        crsr.execute("SELECT * FROM " + sportName)
        infoTitles.extend(list(map(lambda x: x[0], crsr.description))[1:])
    except:
        pass

    with open(os.path.join("user_data", file), mode='w', newline='') as fileWrite:  # creates and writes to CSV file
        fileWrite = csv.writer(fileWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileWrite.writerow(infoTitles)

        for z in mainList:
            fileWrite.writerow(z)

    if database == "tempData.db":  # shows messages based on the context of function call
        tk.messagebox.showinfo("Copy Success", "File successfully copied")
    else:
        tk.messagebox.showinfo("Merge Success", "Files successfully merged")


def mergeFiles(fieldFile1, fieldFile2, fieldFile3):  # merges files into a larger one
    file1 = fieldFile1.get()
    file2 = fieldFile2.get()
    file3 = fieldFile3.get()

    if (fieldFile1 != "" and fieldFile1.getIsFilled()) and (fieldFile2 != "" and fieldFile2.getIsFilled()) and (fieldFile3 != "" and fieldFile3.getIsFilled()):

        if file1[-4:].lower() != ".txt" and file1[-4:].lower() != ".csv":  # gets file #1 extension
            file1 += fileExtension

        if file2[-4:].lower() != ".txt" and file2[-4:].lower() != ".csv":  # gets file #2 extension
            file2 += fileExtension

        if file3[-4:].lower() != ".txt" and file3[-4:].lower() != ".csv":  # gets new file extension
            file3 += fileExtension

        if os.path.isfile(os.path.join("user_data", file1)) and os.path.isfile(os.path.join("user_data", file2)) and not os.path.isfile(os.path.join("user_data", file3)):  # merges files
            copyfile("myData.db", "mergeData.db")
            initializeDatabase("mergeData.db", fieldFile1, True)
            initializeDatabase("mergeData.db", fieldFile2, False)

            if savedSportName1 == savedSportName2:
                if file3[-4:].lower() == ".txt":
                    writeToTxt(file3, "mergeData.db")
                else:
                    writeToCsv(file3, "mergeData.db")  # extra attribute names not present
            else:
                tk.messagebox.showerror("Files Not Compatible", "Files must be associated with the same sport to be merged")

            try:
                os.remove("mergeData.db")
            except:
                pass

        else:  # displays error if a file does not exist
            tk.messagebox.showerror("Merging File Error", "Make sure that both of the files you are attempting to merge exist, and that the new file you are attempting to create is not already present in the application")

    else:  # displays error if not enough information is given
        tk.messagebox.showerror("File Entry Error", "Enter data in all file fields to merge files")

    fieldFile1.delete(0, END)
    fieldFile1.insert(0, "File #1")
    fieldFile1['style'] = "Placeholder.TEntry"

    fieldFile2.delete(0, END)
    fieldFile2.insert(0, "File #2")
    fieldFile2['style'] = "Placeholder.TEntry"

    fieldFile3.delete(0, END)
    fieldFile3.insert(0, "New File")
    fieldFile3['style'] = "Placeholder.TEntry"


def getNewFile(window):  # allows user to import new files into the application
    global fileSearch

    if not fileSearch:
        fileSearch = True
        window.withdraw()

        try:  # tries to import file
            filePath = filedialog.askopenfilename()
            fileSearch = False
            fileName = pathLeaf(filePath)

            if fileName == "":
                pass
            elif fileName[-4:].lower() == ".txt" or fileName[-4:].lower() == ".csv":

                currentFiles = os.listdir("user_data")
                fileAlreadyPresent = False

                for x in currentFiles:
                    if x.lower() == fileName.lower():
                        fileAlreadyPresent = True

                if fileAlreadyPresent:  # if the file name is already used, an error message is displayed
                    tk.messagebox.showerror("File Name Error", f'File with name "{fileName}" already exists in the application')

                else:  # imports data file
                    copyfile(filePath, os.path.join("user_data", fileName))
                    tk.messagebox.showinfo("Copy Success", "File successfully added to the application data")

            else:  # if the selected import is the wrong file type, an error message is displayed
                tk.messagebox.showerror("File Type Error", 'Imported data files must be of type "TXT" or "CSV"')

        except:  # displays an error if file copying fails
            tk.messagebox.showerror("File Error", "An error occurred when attempting to create a file copy")

    try:
        window.deiconify()
        window.lift()
    except:
        pass


def pathLeaf(path):  # gets file info from path
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def resetToDefaults(numSelect, fileFormatSelect):  # resets base program settings
    global FT

    numSelect.set("Quick Train (DEFAULT)")
    fileFormatSelect.set("Title Values (DEFAULT)")
    FT.set(1)
    # settings are set to their base values

    try:  # attempts to remove settings file
        os.remove("SETTINGS.txt")
    except:
        pass


def saveSettings(numSelect, fileFormatSelect):  # saves user settings to file
    global fileExtension

    try:  # attempts to save to file
        f = open("SETTINGS.txt", "w")
        f.write(numSelect + "\n" + str(FT.get()) + "\n" + fileFormatSelect)
        f.close()
    except:
        pass

    fileExtension = getFileType()
    # sets global base file extension


def getSettings(numSelect, fileFormatSelect):  # gets current user settings

    try:  # attempts to get user settings
        f = open("SETTINGS.txt", "r")
        settings = f.readlines()
        settings[0] = settings[0].rstrip("\n")
        numSelect.set(settings[0])
        FT.set(settings[1].rstrip("\n"))
        fileFormatSelect.set(settings[2])
        f.close()

    except:  # if this fails, the base settings are used
        numSelect.set("Quick Train (DEFAULT)")
        fileFormatSelect.set("Title Values (DEFAULT)")
        FT.set(1)


def getFileType():  # returns file type from base settings

    try:  # attempts to find base file type from settings
        f = open("SETTINGS.txt", "r")
        settings = f.readlines()
        f.close()

        if settings[1].rstrip("\n") == "2":
            return ".csv"
        else:
            return ".txt"

    except:  # if nothing is found, the TXT file type is returned as the default
        return ".txt"


def centerWindow(myRoot, width, height):  # centers window with a specific size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    myRoot.geometry('%dx%d+%d+%d' % (width, height, x, y))


def loadingManager():  # manages loading progress of the neural network
    load = ttk.Progressbar(root, orient=HORIZONTAL, length=90, mode="indeterminate")
    load.grid(row=3, column=0, padx=5, pady=(20, 2), stick=W)

    incomplete = True
    fileText = ""

    while incomplete:  # shows loading animation
        for x in range(6):
            load['value'] = x * 20
            time.sleep(0.4)

        f = open("MINFO.txt", "r")
        fileText = f.read()
        f.close()

        if fileText != "":  # checks if data file has returned values
            incomplete = False

    if fileText == "\n\n":  # compiles data from file and adds to a string to be displayed
        displayText = "\nYour model is trained and ready for use!"
    elif fileText == "\n\n\n":
        displayText = "\nMake sure your model is trained and try again."
    else:
        try:  # attempts to compile data fro file
            fileLine = fileText.split("\n")

            probabilities = fileLine[0].strip("[]").split()
            probabilities = [float(probability) for probability in probabilities]

            predictedProbability = max(probabilities)
            result = probabilities.index(predictedProbability)

            if result == 0:
                displayText = "\nPredicted Result: Home Team Victory\n\n"
            elif result == 1:
                displayText = "\nPredicted Result: Away Team Victory\n\n"
            else:
                displayText = "\nPredicted Result: Tie\n\n"

            displayText += "Probabilities --\n"
            displayText += f"Home Win: {round(probabilities[0] * 100, 2)}%\n"
            displayText += f"Away Win: {round(probabilities[1] * 100, 2)}%\n"
            displayText += f"Tie: {round(probabilities[2] * 100, 2)}%\n\n"

            displayText += f"Evaluated Model Accuracy: {round(float(fileLine[1]) * 100, 2)}%\n"
            displayText += f"Prediction Confidence: {round(float(fileLine[2]) * 100, 2)}%"
            centerWindow(root, 400, 450)

        except:  # if there is an error, this is displayed instead of the data
            displayText = "\nOh No!  Your data cannot be displayed...\nPlease try again later."

    load.grid_remove()

    infoDisplay = Label(root, text=displayText)
    infoDisplay.grid(row=4, column=0, columnspan=3, padx=5, pady=(25, 5), stick=W)

    closeDisplay = tk.Button(root, text="Close", bg="lightgrey", command=lambda: resetWindow(infoDisplay, closeDisplay))
    closeDisplay.grid(row=5, column=0, padx=5, stick=W)


def NNSetup(evalType):  # sets up variables needed to run the neural network, and starts threads
    mode, testAmount, homeTeamID, awayTeamID = getArgs()

    if homeTeamID is not None and awayTeamID is not None or evalType:
        if homeTeamID != awayTeamID or evalType:
            if homeTeamID == awayTeamID:
                homeTeamID = None
                awayTeamID = None

            start.grid_remove()
            train.grid_remove()

            loadTxt = StringVar()
            loadLabel = Label(root, textvariable=loadTxt, font=("Arial", 10))
            loadLabel.grid(row=5, column=0, padx=5, pady=(0, 20), stick=W)
            loadTxt.set("Retrieving Data...")

            f = open("MINFO.txt", "w")
            f.write("")
            f.close()

            t1 = threading.Thread(target=loadingManager)
            t1.start()
            # starts loading manager in a new thread

            t2 = threading.Thread(target=Computation.startNeuralNet, args=(evalType, mode, testAmount, homeTeamID, awayTeamID, loadTxt))
            t2.start()
            # starts neural network in a new thread

        else:  # if users attempt to predict by selecting the same two teams, an error is shown
            tk.messagebox.showerror("Same Teams Selected", "You cannot select the same team for both home and away")

    else:  # if users attempt to predict without selecting two teams, an error is shown
        tk.messagebox.showerror("Empty Team Selection", "Make sure to select two teams for the program to predict an outcome")


def getArgs():  # gets arguments needed to run neural network

    try:  # attempts to find run settings from the user settings file
        f = open("SETTINGS.txt", "r")
        settings = f.readlines()
        mode = settings[0]

        mode = mode.rstrip("\n")

        if mode == "Quick Train (DEFAULT)":
            mode = 0
        elif mode == "Selective Train (slower)":
            mode = 1
        elif mode == "GA Deep Train (very slow)":
            mode = 2
        elif mode == "Extended GA Train (slowest)":
            mode = 3

        f.close()

    except:  # if this fails, the base mode is set
        mode = 0

    connection = sqlite3.connect("myData.db")
    crsr = connection.cursor()
    homeTeam = team1.get()
    awayTeam = team2.get()

    try:  # attempts to find team IDs
        homeTeam = crsr.execute("SELECT ID FROM TEAMS WHERE Name=?", (homeTeam,)).fetchone()[0]
        awayTeam = crsr.execute("SELECT ID FROM TEAMS WHERE Name=?", (awayTeam,)).fetchone()[0]

    except:  # returns with them as none if failed
        return mode, 0.3, None, None

    return mode, 0.3, homeTeam, awayTeam
    # returns all values if this point is reached


def resetWindow(infoDisplay, closeDisplay):  # resets the window after neural network data is displayed
    infoDisplay.grid_remove()
    closeDisplay.grid_remove()

    centerWindow(root, 400, 360)
    start.grid(row=3, column=0, padx=5, pady=(20, 5), stick=W)
    train.grid(row=4, column=0, padx=5, pady=(0, 20), stick=W)


root = Tk()
style = ttk.Style(root)
style.configure("Placeholder.TEntry", foreground="grey")
# creates root

centerWindow(root, 400, 360)
root.title("PythoSport")
root.wm_iconbitmap("icon.ico")
# creates window

fileSearch = False
savedSportName1 = "DEFAULT"
savedSportName2 = "DEFAULT"
fileExtension = getFileType()
FT = IntVar()

lb1 = Label(root, text="Home Team:", font=("Arial", 12))
lb1.grid(row=0, column=0, padx=5, pady=5, stick=W)

lb2 = Label(root, text="Away Team:", font=("Arial", 12))
lb2.grid(row=0, column=2, padx=5, pady=5, stick=W)

vs = Label(root, text="vs.")
vs.grid(row=1, column=1, padx=5, pady=5, stick=W)

team1 = Combobox(root, state="readonly")
team1.grid(row=1, column=0, padx=5, pady=5, stick=W)

team2 = Combobox(root, state="readonly")
team2.grid(row=1, column=2, padx=5, pady=5, stick=W)

start = tk.Button(root, text="Predict", bg="lightgrey", command=lambda: NNSetup(False))
start.grid(row=3, column=0, padx=5, pady=(20, 5), stick=W)

train = tk.Button(root, text="Train New Model", bg="lightgrey", command=lambda: NNSetup(True))
train.grid(row=4, column=0, padx=5, pady=(0, 20), stick=W)

root.rowconfigure(6, weight=1)

rdl = Label(root, text="Team Data File Options:", font=("Arial", 9))
rdl.grid(row=7, column=0, padx=5, stick=W)

readTxt = PlaceholderEntry(root, "Enter File Name", False)
readTxt.configure(width=25)
readTxt.grid(row=8, column=0, padx=5, pady=5, stick=W)

startRead = tk.Button(root, text="Read Data", bg="lightgrey", command=lambda: initializeDatabase("myData.db", readTxt, True))
startRead.grid(row=9, column=0, columnspan=3, padx=5, pady=5, stick=W)

fileOpen = tk.Button(root, text="Create/Open File", bg="lightgrey", command=openFile)
fileOpen.grid(row=9, column=0, columnspan=3, padx=75, stick=W)

deleteFile = tk.Button(root, text="Delete", bg="lightgrey", command=deleteFile)
deleteFile.grid(row=9, column=0, columnspan=3, padx=180, stick=W)

listFiles = tk.Button(root, text="List", bg="lightgrey", command=listFiles)
listFiles.grid(row=10, column=0, padx=5, columnspan=3, pady=5, stick=W)

editInit = tk.Button(root, text="Sport Options", bg="lightgrey", command=initAccess)
editInit.grid(row=10, column=0, padx=40, columnspan=3, pady=5, stick=W)

advancedOptions = tk.Button(root, text="Advanced", bg="lightgrey", command=openAdvancedWindow)
advancedOptions.grid(row=10, column=0, padx=130, columnspan=3, pady=5, stick=W)

sportInfoSetup()
teamRetrieval()
root.mainloop()