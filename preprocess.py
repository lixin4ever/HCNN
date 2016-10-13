__author__ = 'lixin77'



import cPickle
from os import listdir
import numpy as np
import sys

def GetDf(df_path, cur_word_to_df={}, cur_vocab={}):
    word_to_df = cur_word_to_df.copy()
    vocab = cur_vocab.copy()
    idx = len(vocab) + 1
    with open(df_path) as fp:
        for line in fp:
            word, df = line.strip().split()
            if word in word_to_df:
                word_to_df[word] += int(df)
            else:
                word_to_df[word] = int(df)
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab, word_to_df

def Lookup(LookupTable, vocab, dim, cur_embeddings=np.array([])):
    # embeddings record information in and outside from the dataset
    embeddings = np.zeros((len(vocab) + 1, dim))
    n_cur_words = len(cur_embeddings)
    if n_cur_words:
        embeddings[:n_cur_words, :] = cur_embeddings.copy() 
    for w in vocab:
        wid = vocab[w]
        if wid < n_cur_words:
            # word has occured in previous vocab
            continue
        embeddings[wid, :] = [float(ele) for ele in LookupTable[w]] if w in LookupTable else np.random.uniform(-0.25, 0.25, dim)
    return embeddings

def GetText(text_path, cur_texts=[]):
    # shallow copy
    texts = cur_texts[:]
    with open(text_path) as fp:
        for line in fp:
            t = line.strip()
            texts.append(t.lower())
    return texts

def GetLabel(label_path, cur_labels=[]):
    labels = cur_labels[:]
    with open(label_path) as fp:
        for line in fp:
            y = int(line.strip())
            labels.append(y)
    return labels

def GetStrength(strength_path, cur_strengths=[]):
    strengths = cur_strengths[:]
    with open(strength_path) as fp:
        for line in fp:
            strengths.append([float(ele) for ele in line.strip().split()])
    return strengths

def GetPOS(pos_path, cur_pos=[]):
    # shallow copy
    pos = cur_pos[:]
    with open(pos_path) as fp:
        for line in fp:
            pos.append([int(ele) for ele in line.strip().split()])
    return pos

def GetRecord(texts, strengths, labels, pos, is_single=True, cur_records=[], n_prev=0):
    records = cur_records[:]
    n_prev_record = n_prev
    for i in xrange(len(texts)):
        if i < n_prev_record:
            continue
        r = {}
        r['text'] = texts[i]
        if not r['text']:
            # filter samples whose text fields are empty
            continue
        r['strength'] = strengths[i]
        r['label'] = labels[i]
        r['pos'] = pos[i]
        records.append(r)

    n_cur_record = len(records) - len(cur_records)
    if not is_single and cur_records:
        print "n_ds1:", n_prev_record
        print "n_ds2:", n_cur_record
    for i in xrange(len(records)):
        if not is_single:
            if i < len(cur_records):
                records[i]['type'] = 'train'
            elif i < len(cur_records) + int(n_cur_record * 0.8):
                records[i]['type'] = 'test'
            elif i >= len(cur_records) + int(n_cur_record * 0.8):
                records[i]['type'] = 'val'
        else:
            if i < int(len(records) * 0.6):
                records[i]['type'] = 'train'
            elif i < int(len(records) * 0.8):
                records[i]['type'] = 'val'
            else:
                records[i]['type'] = 'test'
    if not is_single and cur_records:
        print "n_train:", len([r for r in records if r['type'] == 'train'])
        print "n_test:", len([r for r in records if r['type'] == 'test'])
        print "n_val:", len([r for r in records if r['type'] == 'val'])
    return records

def BuildDataset(datasets, is_single):
    # glove word representation
    glove_vectors = {}
    glove_dim = 200

    print "load glove word vector..."
    with open('glove.twitter.27B.200d.txt') as fp:
        for line in fp:
            eles = line.strip().split(' ')
            w, values = eles[0], eles[1:]
            glove_vectors[w] = values
    print "done!\n\n"

    if is_single:
        assert isinstance(datasets, list)
        for ds in datasets:
            print "process dataset", ds
            dir_name = './data/%s' % ds

            df_path = '%s/%s_df.txt' % (dir_name, ds)
            vocab, word_to_df = GetDf(df_path=df_path)

            # embeddings record information in and outside from the dataset
            glove_embeddings = Lookup(LookupTable=glove_vectors, vocab=vocab, dim=glove_dim)

            text_path = '%s/%s_text.txt' % (dir_name, ds)
            texts = GetText(text_path=text_path)

            label_path = '%s/%s_label.txt' % (dir_name, ds)
            labels = GetLabel(label_path=label_path)

            strength_path = '%s/%s_strength.txt' % (dir_name, ds)
            strengths = GetStrength(strength_path=strength_path)

            pos_path = '%s/%s_pos.txt' % (dir_name, ds)
            pos_info = GetPOS(pos_path=pos_path)

            assert len(texts) == len(labels) == len(strengths) == len(pos_info)

            records = GetRecord(texts=texts, strengths=strengths, labels=labels, pos=pos_info)
            print "dataset: %s, n_records: %s\n\n" % (ds, len(records))
            #cPickle.dump([records, glove_embeddings, vocab, word_to_df], open("./pkl/%s.pkl" % (ds), "wb"))
    else:
        assert isinstance(datasets, dict)
        for ds1 in datasets[1]:
            print "ds1: %s...\n" % ds1
            dir_name_ds1 = './data/%s' % ds1

            df_path_ds1 = '%s/%s_df.txt' % (dir_name_ds1, ds1)
            vocab_ds1, word_to_df_ds1 = GetDf(df_path=df_path_ds1)

            glove_embeddings_ds1 = Lookup(LookupTable=glove_vectors, vocab=vocab_ds1, dim=glove_dim)

            text_path_ds1 = '%s/%s_text.txt' % (dir_name_ds1, ds1)
            texts_ds1 = GetText(text_path=text_path_ds1)
            #print "%s: %s texts" % (ds1, len(texts_ds1))

            label_path_ds1 = '%s/%s_label.txt' % (dir_name_ds1, ds1)
            labels_ds1 = GetLabel(label_path=label_path_ds1)

            strength_path_ds1 = '%s/%s_strength.txt' % (dir_name_ds1, ds1)
            strengths_ds1 = GetStrength(strength_path=strength_path_ds1)

            pos_path_ds1 = '%s/%s_pos.txt' % (dir_name_ds1, ds1)
            pos_info_ds1 = GetPOS(pos_path=pos_path_ds1)

            assert len(texts_ds1) == len(labels_ds1) == len(strengths_ds1) == len(pos_info_ds1)

            records_ds1 = GetRecord(texts=texts_ds1, labels=labels_ds1,
                strengths=strengths_ds1, pos=pos_info_ds1, is_single=is_single)

            if not 'type' in records_ds1[1]:
                print "no type field"

            for ds2 in datasets[2]:
                if ds1 == ds2:
                    continue        
                print "ds2: %s..." % ds2
                dir_name_ds2 = './data/%s' % ds2

                df_path_ds2 = '%s/%s_df.txt' % (dir_name_ds2, ds2)
                print "n_w before:", len(vocab_ds1)
                vocab, word_to_df = GetDf(df_path=df_path_ds2, cur_word_to_df=word_to_df_ds1,
                    cur_vocab=vocab_ds1)
                print "n_w after:", len(vocab)
                glove_embeddings = Lookup(LookupTable=glove_vectors, vocab=vocab, dim=glove_dim, 
                    cur_embeddings=glove_embeddings_ds1)

                text_path_ds2 = '%s/%s_text.txt' % (dir_name_ds2, ds2)
                texts = GetText(text_path=text_path_ds2, cur_texts=texts_ds1)

                #print "%s+%s: %s texts" % (ds1, ds2, len(texts))

                label_path_ds2 = '%s/%s_label.txt' % (dir_name_ds2, ds2)
                labels = GetLabel(label_path=label_path_ds2, cur_labels=labels_ds1)

                strength_path_ds2 = '%s/%s_strength.txt' % (dir_name_ds2, ds2)
                strengths = GetStrength(strength_path=strength_path_ds2, 
                    cur_strengths=strengths_ds1)

                pos_path_ds2 = '%s/%s_pos.txt' % (dir_name_ds2, ds2)
                pos_info = GetPOS(pos_path=pos_path_ds2, cur_pos=pos_info_ds1) 

                assert len(texts) == len(labels) == len(strengths) == len(pos_info)

                records = GetRecord(texts=texts, labels=labels, strengths=strengths, pos=pos_info,
                    is_single=is_single, cur_records=records_ds1, n_prev=len(texts_ds1))

                print "dataset: %s+%s, n_records: %s\n\n" % (ds1, ds2, len(records))
                #cPickle.dump([records, glove_embeddings, vocab, word_to_df], open("./pkl/%s_%s.pkl" % (ds1, ds2), "wb"))

if __name__ == '__main__':
    ds_name = sys.argv[1] # ds_name: your own dataset name
    mode = sys.argv[2] # mode: single or multiple, 
    # representing datasets built from single source or multiple sources 
    datasets = {}
    is_single = True
    if ds_name == 'all' and mode == 'single':
        datasets = listdir('./data')
    elif ds_name == 'all' and mode == 'multiple':
        datasets[1] = listdir('./data')
        datasets[2] = listdir('./data')
        is_single = False
    elif ds_name != 'all' and mode == 'single':
        datasets = [ds_name]
    elif ds_name != 'all' and mode == 'multiple':
        datasets[1] = [ds_name]
        datasets[2] = [ds for ds in listdir('./data') if ds != ds_name]
        is_single = False
    BuildDataset(datasets=datasets, is_single=is_single)

