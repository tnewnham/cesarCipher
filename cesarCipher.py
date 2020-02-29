from pyspark import SparkConf, SparkContext
from collections import Counter
from operator import add
from spellchecker import SpellChecker

def debugPrint(x):
    print(x)


def strip(lines):
    return "".join(char for char in lines if char.isalnum()).lower()

conf = SparkConf().setMaster("local").setAppName("cesarCipher")
sc = SparkContext(conf=conf)

# textFile = sc.textFile("C:/Users/tnewn/OneDrive/Desktop/BigDataProgramming/(1) The Hunger Games.txt")
# textFile = sc.textFile("C:/Users/tnewn/OneDrive/Desktop/BigDataProgramming/wordCountText.txt")
textFile = sc.textFile(
    "./Encrypted-1.txt"
)
# textFile = sc.textFIle("C:/Users/tnewn/OneDrive/Desktop/BigDataProgramming/Encrypted-2.txt")
# textFile = sc.textFIle("C:/Users/tnewn/OneDrive/Desktop/BigDataProgramming/Encrypted-3.txt")

lines = textFile.map(lambda line: str(line))
lines.persist()
# lines.unpersist

# Split by word and count
words = lines.flatMap(lambda string: string.split(" "))
wordCount = words.count()

# Get count of distinct word occurrences
# Also strip words to the uppercase alphanumeric representations
wordsCounted = words.map(lambda word: (strip(word), 1)).reduceByKey(add)
wordsCounted.foreach(debugPrint)

# Get count of char occurrences
chars = lines.flatMap(lambda string: list(string))
charCount = chars.count()
charCounted = (
    chars.map(lambda char: (char.upper(), 1))
    .reduceByKey(add)
    .map(lambda char: (char[0], char[1]))
)
charCounted.foreach(debugPrint)

# Sort by most frequent alpha chars
frequentAlphaChars = charCounted.filter(lambda x: x[0].isalpha()).sortBy(
    lambda x: x[1], ascending=False
)
frequentAlphaChars.foreach(debugPrint)

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
frequentCharShifts.foreach(debugPrint)

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
decryptedWords = frequentCharShifts.map(lambda x: (x[1], mapDecryptSampleWords(x[1])))
decryptedWords.foreach(debugPrint)

spell = SpellChecker()


# Check decrypted word is known
def isInDictionary(word):
    return len(spell.known([word])) > 0

# Create map of true/false known decrypted words
shiftByIsInDictionary = decryptedWords.flatMapValues(lambda x: x).mapValues(
    lambda x: isInDictionary(x)
)
shiftByIsInDictionary.foreach(debugPrint)

# Count how many known words in each ROT13 shift 
countOfShiftByIsInDictionary = (
    shiftByIsInDictionary.filter(lambda x: x[1] == True)
    .map(lambda x: (x[0], 1))
    .reduceByKey(add)
    .sortBy(lambda x: x[1], ascending=False)
)
countOfShiftByIsInDictionary.foreach(debugPrint)

mostFrequentShift = countOfShiftByIsInDictionary.first()
print(f"shift={mostFrequentShift[0]}")

# Dechipher every line in original cipher
decipher = lines.map(lambda l: decrypt(l, mostFrequentShift[0])).collect()
print(decipher)

# check for most frequent letter
# take a sample of words (rdd and rdd function)
# sample (rdd function) of words rdd
# shift every character in words by the difference between most common charater and e
# use some natural language procession library
# if majority of words return in english language library
