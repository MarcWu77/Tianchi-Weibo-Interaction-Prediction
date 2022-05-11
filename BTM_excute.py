import bitermplus as btm

class BTM_excute():
    """
    Parameters:
    data -> pandas.DataFrame;
    """
    def __init__(self,data):
        self.data = data

    def preprocess(self):
        self.X, self.vocabulary, self.vocab_dict = btm.get_words_freqs(self.data)
        #tf = np.array(self.X.sum(axis=0)).ravel()
        print('Preprocess Results:\n')
        print('Vocabulary list:',self.vocabulary,'\n','Total number of unique words:',len(self.vocabulary))
        print('*'*30)
        print('Word pairs and their frequencies:\n',self.X)
        print('*'*30)
        # Vectorize documents
        self.docs_vec = btm.get_vectorized_docs(self.data, self.vocabulary)
        docs_lens = list(map(len, self.docs_vec))
        # Generate biterms
        self.biterms = btm.get_biterms(self.docs_vec)
        #print('An Example: the first document includes:',self.biterms[0],'\n','Which has',len(self.biterms[0]),'pairs~')
        print('Preprocess done~')


    def modeling(self,topic_num,wordz_size,alpha,beta,iterations,seed):
        """
        Parameters:
        topic_num -> int, the numbers of topics;
        word_size -> int, the numbers of unique words;
        alpha -> float, hyperparameter for choosing a topic;
        beta -> float, hyperparameter for choosing a word
        seed -> int
        """
        self.model = btm.BTM(self.X,self.vocabulary,seed=seed,
                    T=topic_num, M=wordz_size, alpha=alpha, beta=beta)
        self.model.fit(self.biterms,iterations=iterations)
        print('Training done~')
        return self.model

    def inference(self):
        self.p_zd = self.model.transform(self.docs_vec)
        self.doc_topic_df = btm.get_docs_top_topic(self.data, self.model.matrix_docs_topics_)
        self.doc_topic_df.rename(columns={'documents':'seg_content','label':'topic'},inplace=True)
        print('Topic inference done~')
        return self.doc_topic_df

    def visualize_results(self):
        pass
