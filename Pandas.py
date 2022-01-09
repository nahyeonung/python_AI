import pandas as pd

# array = pd.Series(['사과', '바나나', '당근'], index=['a','b','c'])

# print(array)
# print(array['a'])

word_dict = {
    'Apple' : '사과',
    'Banana' : '바나나',
    'Carrot' : '당근'
}

frequency_dict = {
    'Apple' : 3,
    'Banana' : 5,
    'Carrot' : 7
}
plus_dict = {
    'Apple' : 3,
    'Banana' : 5,
    'Carrot' : 7
}
test_dict = {
    'Apple' : 3,
    'Banana' : 5,
    'Carrot' : 7
}
word = pd.Series(word_dict)
frequency = pd.Series(frequency_dict)
plus = pd.Series(plus_dict)

# 이름(Name) : 값(Values)
summary = pd.DataFrame({
    'word' : word,
    'frequency' : frequency,
    'plus' : plus
})
double = summary['frequency'] + summary['plus']
summary['double'] = double
summary['test'] = test_dict #이렇게 추가하게 되면 원하는 apple은 3이라는 값이 추가되지 않고 그냥 apple만 추가됨.다른 속성들이랑 연관이 없음.
print(summary)

#이름을 기준으로 슬라이싱
print(summary.loc['Banana':'Carrot', 'plus'])

#인덱스를 기준으로 슬라이싱
print(summary.iloc[1:3, 2:])