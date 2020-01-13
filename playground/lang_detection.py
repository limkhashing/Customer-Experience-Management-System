# ms
# en
# zh-CN
from textblob import TextBlob

print(TextBlob("selamat pagi").detect_language())
print(TextBlob("good morning").detect_language())
print(TextBlob("早上好").detect_language())
