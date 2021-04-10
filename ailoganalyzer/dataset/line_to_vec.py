from ailoganalyzer.dataset.utils_semantic_vec import *

def import_word_vec():
    fasttext = load_vectors('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/cc.en.300.vec')
    return fasttext

def line_to_vec(word_vec):
    fasttext = word_vec
    data = data_read_template("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/templates.csv")
    print(data)
    result = {}

    # extraction des mots
    for i in range(len(data)):
        temp = data[i]
        temp = camel_to_snake(temp)
        temp = replace_all_blank(temp)
        temp = " ".join(temp.split())
        temp = lemmatize_stop(temp)
        result[i] = temp
    print(result)

    # garde que les mots dans fasttext
    motok = set(fasttext.keys())
    for e in result:
        result[e] = list(set(result[e]) & motok)

    dump_2_json(result, '/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')

    # 单独保存需要用到的fasttext词向量
    template_set = set()
    for key in result.keys():
        for word in result[key]:
            template_set.add(word)

    template_fasttext_map = {}

    for word in template_set:
        if word in fasttext:
            template_fasttext_map[word] = list(fasttext[word])

    dump_2_json(template_fasttext_map,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')

    #2 ..

    eventid2template = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')
    fasttext_map = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')
    print(eventid2template)
    dataset = list()
    with open('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/train/train', 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            dataset.append(line)
    print(len(dataset))
    idf_matrix = list()
    for seq in dataset:
        for event in seq:
            idf_matrix.append(eventid2template[str(event)])
    print(len(idf_matrix))
    idf_matrix = np.array(idf_matrix)
    X_counts = []
    for i in range(idf_matrix.shape[0]):
        word_counts = Counter(idf_matrix[i])
        X_counts.append(word_counts)
    print(X_counts[1000])
    X_df = pd.DataFrame(X_counts)
    X_df = X_df.fillna(0)
    print(len(X_df))
    print(X_df.head())
    events = X_df.columns
    print(events)
    X = X_df.values
    num_instance, num_event = X.shape

    print('tf-idf here')
    df_vec = np.sum(X > 0, axis=0)
    print(df_vec)
    print('*'*20)
    print(num_instance)
    # smooth idf like sklearn
    idf_vec = np.log((num_instance + 1)  / (df_vec + 1)) + 1
    print(idf_vec)
    idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
    X_new = idf_matrix
    print(X_new.shape)
    print(X_new[1000])

    word2idf = dict()
    for i,j in zip(events,idf_vec):
        word2idf[i]=j
        # smooth idf when oov
        word2idf['oov'] = (math.log((num_instance + 1)  / (29+1)) + 1)

    print(word2idf)
    dump_2_json(word2idf,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/word2idf.json')

    # 3..

    event2template = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')
    fasttext = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')
    word2idf = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/word2idf.json')

    event2semantic_vec = dict()
    # todo :
    # 计算每个seq的tf，然后计算句向量
    for event in event2template.keys():
        template = event2template[event]
        tem_len = len(template)
        count = dict(Counter(template))
        for word in count.keys():
            # TF
            TF = count[word]/tem_len
            # IDF
            IDF = word2idf.get(word,word2idf['oov'])
            # print(word)
            # print(TF)
            # print(IDF)
            # print('-'*20)
            count[word] = TF*IDF
        # print(count)
        # print(sum(count.values()))
        value_sum = sum(count.values())
        for word in count.keys():
            count[word] = count[word]/value_sum
        semantic_vec = np.zeros(300)
        for word in count.keys():
            fasttext_weight = np.array(fasttext[word])
            semantic_vec += count[word]*fasttext_weight
        event2semantic_vec[event] = list(semantic_vec)
    dump_2_json(event2semantic_vec,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/event2semantic_vec.json')
