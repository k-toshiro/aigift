#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import MeCab


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aigift.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

def parse(text):
   mecab = MeCab.Tagger('-Owakati')
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

if __name__ == '__main__':
    main()
