from __future__ import division
import csv

def getBattersData():
	with open('data/Batting.csv', 'rt') as csvfile:
		data = []
		reader = csv.reader(csvfile)
		next(reader, None)
		for row in reader:
			if '' not in row:
				data.append([row[0], int(row[1]), int(row[2]), row[3], row[4], int(row[5]), int(row[6]), int(row[7]), int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), int(row[13]), int(row[14]), int(row[15]), int(row[16]), int(row[17]), int(row[18]), int(row[19]), int(row[20]), int(row[21])])
	return data

def getPitchersData():
	with open('data/Pitching.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile)
		data = []
		next(reader, None)
		for row in reader:
			data.append(row)
	return data

def loadPlayerInfo():
	with open('data/Master.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile)
		data = {}
		next(reader, None)
		for row in reader:
			temp = list(row[1:])
			temp.append([])
			data[row[0]] = temp

	return data

def getTeamData():
	with open('data/Teams.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile)
		data = {}
		leagueAdjust = {}
		next(reader, None)
		for row in reader:
			if row[2] not in data:
				data[row[2]] = {}
			data[row[2]][int(row[0])] = row
			if int(row[0]) not in leagueAdjust:
				leagueAdjust[int(row[0])] = []
			leagueAdjust[int(row[0])].append(int(row[14]))

	for year, values in leagueAdjust.items():
		leagueAdjust[year] = sum(values) / len(values)
		
	avg = 0
	for year, values in leagueAdjust.items():
		avg += values

	avg /= len(leagueAdjust)
	
	for year, values in leagueAdjust.items():
		leagueAdjust[year] = 2 - (leagueAdjust[year] / avg)

	return (data, leagueAdjust)

def combineYearsByKey(years):
	finalYears = []
	runningYears = {}
	for year in years:
		if year[2] not in runningYears:
			runningYears[year[2]] = year
		else:
			for i in range(6, len(year)):
				runningYears[year[2]][i] += year[i]
	
	for key, value in runningYears.items():
		finalYears.append(value)

	return finalYears

def getFinalData():
	batterInfo = getBattersData()
	playerInfoBat = loadPlayerInfo()
	teamInfo = getTeamData()

	leagueAdjust = teamInfo[1]
	teamInfo = teamInfo[0]
	
	for season in batterInfo:
		parkFactor = 2 - (int(teamInfo[season[3]][int(season[1])][43]) / 100)
		yearFactor = leagueAdjust[int(season[1])]
		for i in range(7, 13):
			season[i] = season[i] * parkFactor * yearFactor

		if playerInfoBat[season[0]][0] != '' and season[1] != '':
			playerInfoBat[season[0]][23].append([int(season[1]) - int(playerInfoBat[season[0]][0])] + season)

	for player in playerInfoBat:
		playerInfoBat[player][23] = combineYearsByKey(playerInfoBat[player][23])
	return playerInfoBat

def getYearsBack(i, playerSeasons, yearsBack):
	sample = []
	current = playerSeasons[i]
	for year in range(yearsBack):
		if i < yearsBack or i-(year+1) < 0:
			sample += [current[0] - year - 1, current[1], current[2] - year - 1, current[3], current[4], current[5]]
			for j in range(6, len(current)):
				sample += [0]
		else:
			sample += playerSeasons[i-(year+1)]

	return sample

def generateSamples(yearsBack=1):
	trainX = []
	trainY = []
	
	data = getFinalData()

	for key in data:
		for i, playerSeason in enumerate(data[key][23]):
			trainX.append(getYearsBack(i, data[key][23], yearsBack))
			trainY.append(playerSeason)

	return (trainX, trainY)
