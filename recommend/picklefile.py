def parse(text):
   mecab = MeCab.Tagger('-Ochasen')
   mecab.parse('')
   node = mecab.parseToNode(text)
   word_list = list()
   while node:
       word = node.surface
       word_type = node.feature.split(",")[0]

       if word_type in ["名詞", "動詞", "形容詞", "副詞"]:
           if word != "*":
               word_list.append(word)
               
       node = node.next
   return word_list