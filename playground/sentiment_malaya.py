# try bert on linux
# multinomial, bert, albert, xlnet
# {'bert': ['base', 'small'], 'xlnet': ['base'], 'albert': ['base']}
# albert and bert got error
import operator
import malaya

positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan sebanyak RM50 juta. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
neutral_text = 'Sekurang-kurangnya ia bukan buku yang dahsyat.'

model = malaya.sentiment.transformer(model='xlnet', size='base')
# model = malaya.sentiment.multinomial()

positive = model.predict(neutral_text,get_proba=True,add_neutral=True)
print(positive)
print(positive['positive'])
print(positive['negative'])
print(positive['neutral'])
print(max(positive.items(), key=operator.itemgetter(1))[0])