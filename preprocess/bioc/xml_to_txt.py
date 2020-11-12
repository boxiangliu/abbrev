#!/mnt/home/kwc/my_venv/bin/python

import xmltodict,sys

try:
    obj=xmltodict.parse(sys.stdin.read())
except:
    print('read failed')
    sys.exit()

print('source:\t' + str(obj['collection']['source']))
print('key:\t' + str(obj['collection']['key']))
print('date:\t' + str(obj['collection']['date']))

def get_location(a):
    loc = a['location']
    if isinstance(loc, list):
        return '|'.join([l['@offset'] + '+' + l['@length'] for l in loc])
    else: 
        return loc['@offset'] + '+' + loc['@length']

def old_do_passage(passage):
    if isinstance(passage, list):
        for p in passage:
            do_passage(p)
    elif not 'text' in passage:
        print('text:\t' + str(passage))
    else:
        print('text:\t' + str(passage))
        if 'annotation' in passage:
            for annotation in passage['annotation']:
                print('\t'.join(['annotation:', str(annotation['@id']), str(annotation['text']), get_location(annotation)]))

def get_text_from_passage(passage):
    if 'text' in passage:
        return passage['text']
    elif 'infon' in passage:
        if 'text' in passage['infon']:
            return passage['infon']['text']
    return str(passage)

def do_passage(passage):
     print('text:\t' + get_text_from_passage(passage))
     if 'annotation' in passage:
         for annotation in passage['annotation']:
             print('\t'.join(['annotation:', str(annotation['@id']), str(annotation['text']), get_location(annotation)]))

for doc in obj['collection']['document']:
    print('\nid:\t' + doc['id'])
    passages = doc['passage']
    if isinstance(passages, list):
        for passage in passages:
            do_passage(passage)
    else:
        do_passage(passages)





      
