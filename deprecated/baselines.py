import gzip
from collections import defaultdict


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

# Rating baseline: compute averages for each user, or return the global
# average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
    user, item = l['reviewerID'], l['itemID']
    allRatings.append(l['rating'])
    userRatings[user].append(l['rating'])

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if u in userAverage:
        predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
    else:
        predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')

predictions.close()

# Helpfulness baseline: similar to the above. Compute the global average
# helpfulness rate, and the average helpfulness rate for each user

allHelpful = []
userHelpful = defaultdict(list)

for l in readGz("train.json.gz"):
    user, item = l['reviewerID'], l['itemID']
    allHelpful.append(l['helpful'])
    userHelpful[user].append(l['helpful'])

averageRate = sum([x['nHelpful'] for x in allHelpful]) * \
    1.0 / sum([x['outOf'] for x in allHelpful])
userRate = {}
for u in userHelpful:
    totalU = sum([x['outOf'] for x in userHelpful[u]])
    if totalU > 0:
        userRate[u] = sum([x['nHelpful']
                           for x in userHelpful[u]]) * 1.0 / totalU
    else:
        userRate[u] = averageRate

predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, i, outOf = l.strip().split('-')
    outOf = int(outOf)
    if u in userRate:
        predictions.write(
            u + '-' + i + '-' + str(outOf) + ',' + str(outOf * userRate[u]) + '\n')
    else:
        predictions.write(
            u + '-' + i + '-' + str(outOf) + ',' + str(outOf * averageRate) + '\n')

predictions.close()

# Purchasing baseline: just rank which items are popular and which are
# not, and return '1' if an item is among the top-ranked

itemCount = defaultdict(int)
totalPurchases = 0

for l in readGz("train.json.gz"):
    user, item = l['reviewerID'], l['itemID']
    itemCount[item] += 1
    totalPurchases += 1

mostPopular = [(itemCount[x], x) for x in itemCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPurchases / 2:
        break

predictions = open("predictions_Purchase.txt", 'w')
for l in open("pairs_Purchase.txt"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if i in return1:
        predictions.write(u + '-' + i + ",1\n")
    else:
        predictions.write(u + '-' + i + ",0\n")

predictions.close()
