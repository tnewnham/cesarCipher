from pyspark import SparkConf, SparkContext
from collections import Counter
from operator import add
from spellchecker import SpellChecker

# Read all the lines
# Count the chars and words
# Count the most frequent chars
# FrequentChars * Common English Chars
# Determing ROT13 shift of each FrequentChar * CommonChar
# Get Random Sample of words
# Decrypt random sample by each ROT13 shift
# Get Known/Unknown (True/False) word count from each ROT13 shift
# Get ROT13 shift with most correct (Known/True) words
# Decrypt entire file with that ROT13 shift


def debugPrint(x):
    print(x)

def debugTopPrint(x, y):
    collection = x.take(y)
    for x in collection:
        print(x)

def strip(lines):
    return "".join(char for char in lines if char.isalnum()).lower()


conf = SparkConf().setMaster("local").setAppName("cesarCipher")
sc = SparkContext(conf=conf)


# encryptedFileName = "./Encrypted-1.txt"
# decryptedFileName = "./Decrypted-1.txt"
# encryptedFileName = "./Encrypted-2.txt"
# decryptedFileName = "./Decrypted-2.txt"
encryptedFileName = "./Encrypted-3.txt"
decryptedFileName = "./Decrypted-3.txt"
textFile = sc.textFile(encryptedFileName)

lines = textFile.map(lambda line: str(line))
lines.persist()
# lines.unpersist

# Split by word and count
words = lines.flatMap(lambda string: string.split(" "))
wordCount = words.count()

# Get count of distinct word occurrences
# Also strip words to the uppercase alphanumeric representations
wordsCounted = words.map(lambda word: (strip(word), 1)).reduceByKey(add)
print("wordsCounted:")
debugTopPrint(wordsCounted, 10)

# Get count of char occurrences
chars = lines.flatMap(lambda string: list(string))
charCount = chars.count()
charsCounted = (
    chars.map(lambda char: (char.upper(), 1))
    .reduceByKey(add)
    .map(lambda char: (char[0], char[1]))
)
print("charsCounted:")
debugTopPrint(charsCounted, 10)

expectedFrequencies = {
    "E": 12.02,
    "T": 9.10,
    "A": 8.12,
    "O": 7.68,
    "I": 7.31,
    "N": 6.95,
    "S": 6.28,
    "R": 6.02,
    "H": 5.92,
    "D": 4.32,
    "L": 3.98,
    "U": 2.88,
    "C": 2.71,
    "M": 2.61,
    "F": 2.30,
    "Y": 2.11,
    "W": 2.09,
    "G": 2.03,
    "P": 1.82,
    "B": 1.49,
    "V": 1.11,
    "K": 0.69,
    "X": 0.17,
    "Q": 0.11,
    "J": 0.10,
    "Z": 0.07,
}

# Get Frequencies of chars in file
charsFreq = charsCounted.filter(lambda x: x[0].isalpha()).map(
    lambda x: (x[0], x[1] / charCount * 100)
)
charsFreq.foreach(
    lambda x: print(f"Cipher Char={x[0]} Freq={x[1]} vs {expectedFrequencies[x[0]]}")
)

# Sort by most frequent alpha chars
frequentAlphaChars = (
    charsCounted.filter(lambda x: x[0].isalpha()).sortBy(
        lambda x: x[1], ascending=False
    )
    # Performance Increase: Limit list by top X chars. This would limit O(n*m) complexity
    # Needs balancing because it skip weird edge case files?
    # .take(4) but it doesn't work????
)
print("frequentAlphaChars:")
debugTopPrint(frequentAlphaChars, 5)

# List of common english characters
# By adding more letters you increase search time O(n*m)
commonChars = ["E", "A", "R", "I"]  # , 'O', 'T', 'N', 'S']

dict1 = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
}

dict2 = {
    0: "Z",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
}

# Determines shift for ROT13 (See decrypt())
def mapCommonCharShift(encryptedChar):
    return map(
        lambda commonChar: (dict1[commonChar] - dict1[encryptedChar] + 26) % 26,
        commonChars,
    )


# Map each frequent char to a list of it's ROT13 shift from english common Chars
"""
O(n*length(commonChars))
List of common chars
[
    'R': 6000,
    'G': 3000
]
[
    'R': 13, # 'E'
    'R': 9,  # 'A'
    'R': 0,  # 'R'
    'R': 17  # 'I',
    'G': 24  # 'E'
    'G': 20  # 'A'
    ...
]
"""
frequentCharShifts = frequentAlphaChars.map(
    lambda x: (x[0], mapCommonCharShift(x[0]))
).flatMapValues(lambda x: x)
print("frequentCharShifts:")
debugTopPrint(frequentCharShifts, 5)

# ROT13 Decrypt algorithim
def decrypt(cipher, shift):
    decipher = ""
    for char in cipher:
        if char.isalpha():
            num = (dict1[char.upper()] - shift + 26) % 26
            decipher += dict2[num]
        else:
            decipher += char
    return decipher


# Get a 30% sample of words from the document
sampleWords = wordsCounted.sample(False, 0.3, 81).map(lambda x: x[0]).collect()

# Decrypt each sample word by the ROT13 shift
def mapDecryptSampleWords(shift):
    return map(lambda sampleWord: decrypt(sampleWord, shift), sampleWords)


# Create map of ROT13 shift by decrypted sample word
decryptedWords = frequentCharShifts.map(lambda x: (x[1], mapDecryptSampleWords(x[1]))).flatMapValues(lambda x: x)
print("decryptedWords:")
debugTopPrint(decryptedWords, 5)

# Init natural language processor
spell = SpellChecker()

# Check decrypted word is known
def isInDictionary(word):
    return len(spell.known([word])) > 0


# Create map of true/false known decrypted words
shiftByIsInDictionary = decryptedWords.mapValues(
    lambda x: isInDictionary(x)
)
print("shiftByIsInDictionary:")
debugTopPrint(shiftByIsInDictionary, 20)

# Count how many known words in each ROT13 shift
countOfShiftByIsInDictionary = (
    shiftByIsInDictionary.filter(lambda x: x[1] == True)
    .map(lambda x: (x[0], 1))
    .reduceByKey(add)
    .sortBy(lambda x: x[1], ascending=False)
)
print("countOfShiftByIsInDictionary:")
debugTopPrint(countOfShiftByIsInDictionary, 20)

mostFrequentShift = countOfShiftByIsInDictionary.first()
print(f"shift={mostFrequentShift[0]}")

# Dechipher every line in original cipher
decipher = lines.map(lambda l: decrypt(l, mostFrequentShift[0]))
print("decipher:")
debugTopPrint(decipher, 20)

# Throws odd exception. Might have conflict with dropbox or OneDrive?
# decipher.saveAsTextFile(decryptedFileName)
# Instead manually collect and print :'(
with open(decryptedFileName, "w") as text_file:
    for line in decipher.collect():
        print(line, file=text_file)
