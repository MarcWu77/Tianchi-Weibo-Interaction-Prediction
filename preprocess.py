import pandas as pd
import numpy as np
import math
import jieba
from chinese_calendar import is_workday

class preprocess():
    """
    Parameters:
    data -> pandas.DataFrame
    training_month -> int
    start -> datetime, 'yyyy-mm-dd'
    end -> datetime, 'yyyy-mm-dd'
    """
    def __init__(self,data,training_month,start,end):
        self.data = data
        self.training_month = training_month
        self.start = start
        self.end = end

    def post_frequency(self):
        df = pd.DataFrame(self.data.uid.value_counts(),columns=['uid','freq'])
        df.columns = ['freq','uid']
        df['uid'] = df.index
        self.data = pd.merge(self.data,df)
        self.data['freq_month'] = round(self.data['freq']/self.training_month)
        return self.data

    def fans_features(self):
        self.data['avg_forward'] = self.data.groupby('uid')['forward_count'].transform('mean')
        self.data['avg_comment'] = self.data.groupby('uid')['comment_count'].transform('mean')
        self.data['avg_like'] = self.data.groupby('uid')['like_count'].transform('mean')
        
        # set the mask of forward
        L1 = (self.data.avg_forward<3)
        L2 = (self.data.avg_forward>=3)&(self.data.avg_forward<9)
        L3 = (self.data.avg_forward>=9)&(self.data.avg_forward<27)
        L4 = (self.data.avg_forward>=27)&(self.data.avg_forward<81)
        L5 = (self.data.avg_forward>=81)&(self.data.avg_forward<243)
        L6 = (self.data.avg_forward>=243)&(self.data.avg_forward<729)
        L7 = (self.data.avg_forward>=729)&(self.data.avg_forward<2187)
        L8 = (self.data.avg_forward>=2187)&(self.data.avg_forward<6561)
        L9 = (self.data.avg_forward>=6561)&(self.data.avg_forward<19683)
        L10 = (self.data.avg_forward>19683)
        # assign forward level
        self.data.loc[L1,'forward_level'] = 1
        self.data.loc[L2,'forward_level'] = 2
        self.data.loc[L3,'forward_level'] = 3
        self.data.loc[L4,'forward_level'] = 4
        self.data.loc[L5,'forward_level'] = 5
        self.data.loc[L6,'forward_level'] = 6
        self.data.loc[L7,'forward_level'] = 7
        self.data.loc[L8,'forward_level'] = 8
        self.data.loc[L9,'forward_level'] = 9
        self.data.loc[L10,'forward_level'] = 10

        # set the mask of comment
        L1 = (self.data.avg_comment<1)
        L2 = (self.data.avg_comment>=1)&(self.data.avg_comment<3)
        L3 = (self.data.avg_comment>=3)&(self.data.avg_comment<9)
        L4 = (self.data.avg_comment>=9)&(self.data.avg_comment<27)
        L5 = (self.data.avg_comment>=27)&(self.data.avg_comment<81)
        L6 = (self.data.avg_comment>=81)&(self.data.avg_comment<243)
        L7 = (self.data.avg_comment>=243)&(self.data.avg_comment<729)
        L8 = (self.data.avg_comment>=729)&(self.data.avg_comment<2187)
        L9 = (self.data.avg_comment>=2187)&(self.data.avg_comment<6561)
        L10 = (self.data.avg_comment>=6561)
        # assign comment level
        self.data.loc[L1,'comment_level'] = 1
        self.data.loc[L2,'comment_level'] = 2
        self.data.loc[L3,'comment_level'] = 3
        self.data.loc[L4,'comment_level'] = 4
        self.data.loc[L5,'comment_level'] = 5
        self.data.loc[L6,'comment_level'] = 6
        self.data.loc[L7,'comment_level'] = 7
        self.data.loc[L8,'comment_level'] = 8
        self.data.loc[L9,'comment_level'] = 9
        self.data.loc[L10,'comment_level'] = 10

        # set the mask of like
        L1 = (self.data.avg_like<1)
        L2 = (self.data.avg_like>=1)&(self.data.avg_like<3)
        L3 = (self.data.avg_like>=3)&(self.data.avg_like<9)
        L4 = (self.data.avg_like>=9)&(self.data.avg_like<27)
        L5 = (self.data.avg_like>=27)&(self.data.avg_like<81)
        L6 = (self.data.avg_like>=81)&(self.data.avg_like<243)
        L7 = (self.data.avg_like>=243)&(self.data.avg_like<729)
        L8 = (self.data.avg_like>=729)&(self.data.avg_like<2187)
        L9 = (self.data.avg_like>=2187)&(self.data.avg_like<6561)
        L10 = (self.data.avg_like>=6561)
        # assign like level
        self.data.loc[L1,'like_level'] = 1
        self.data.loc[L2,'like_level'] = 2
        self.data.loc[L3,'like_level'] = 3
        self.data.loc[L4,'like_level'] = 4
        self.data.loc[L5,'like_level'] = 5
        self.data.loc[L6,'like_level'] = 6
        self.data.loc[L7,'like_level'] = 7
        self.data.loc[L8,'like_level'] = 8
        self.data.loc[L9,'like_level'] = 9
        self.data.loc[L10,'like_level'] = 10

        #df = self.data[['uid','forward_level','comment_level','like_level','freq_month']]
        #df.drop_duplicates(inplace=True)
        #df.to_csv('data/user_features/user_features.txt',sep=',',index=False)

        return self.data

    def time_features(self):
        days = pd.date_range(start=self.start,end=self.end)
        workdays = [day for day in days if is_workday(day)]
        self.data['workday'] = np.where(self.data.time.dt.date.isin(workdays)==1,1,0)
        return self.data

    def explicit_content_features(self):
        self.data['tag'] = np.where(self.data.content.str.contains(r'#(.{0,30})#')==1,1,0) # add tags 0/1 variable
        self.data['url'] = np.where(self.data.content.str.contains(r'(http://[a-zA-z./\d]*)|(https://[a-zA-z./\d]*)')==1,1,0) # add url 0/1 variable
        self.data['at'] = np.where(self.data.content.str.contains(r'@([^@]{0,30})\s')==1,1,0) # add @ 0/1 variable
        self.data['emoj'] = np.where(self.data.content.str.contains(r'\[.{0,12}\]')==1,1,0) # add emoj 0/1 variable
        return self.data

    def tokenize_content(self):
        # Clean first
        self.data['content'] = self.data.content.str.replace(r'(http://[a-zA-z./\d]*)|(https://[a-zA-z./\d]*)','') # remove url
        self.data['content'] = self.data.content.str.replace(r'\[.{0,12}\]','') # remove emoj
        self.data['content'] = self.data.content.str.replace(r'@([^@]{0,30})\s','') # remove @sb
        self.data['content'] = self.data.content.str.replace(r'[^\u4e00-\u9fa5]+','') # remove non-Chinese strings
        self.data['content'] = self.data.content.str.replace(r'\uAC00-\uD7AF+','') # remove Korean
        self.data['content'] = self.data.content.str.replace(r'\u3040-\u31FF+','') # remove Japanese
        # Tokenization
        with open("tools/stopwords_cn_HIT.txt",encoding='utf8') as f: # Chinese & English stopwords
            stopwords = [stopword.strip() for stopword in f.readlines()]
        def cut_words(x):
            words = jieba.lcut(str(x))
            return [word for word in words if word not in stopwords]
        self.data['seg_content_lis'] = self.data.content.apply(cut_words)
        self.data['seg_content'] = self.data.seg_content_lis.apply(lambda x:' '.join(x))

    def TFIDF(self):
        global words_df
        words_df = {}
        doc_num = len(self.data)
        def DF(xs):
            for x in xs:
                if x in words_df:
                    words_df[x] += 1
                else:
                    words_df[x] = 1
        def TF_iDF(xs):
            words_tf = {}
            words_num = len(xs)
            for x in xs:
                if x in words_tf:
                    words_tf[x] += 1
                else:
                    words_tf[x] = 1
            words_tf = {k:v/words_num for k,v in words_tf.items()}
            words_tfidf = {
                k:float(words_tf[k])/math.log(doc_num/(words_df[k]+1)) # TF-iDF formula
                for k,v in words_tf.items()
            }
            return ' '.join([i[0] for i in sorted(words_tfidf.items(),key=lambda x:x[1],reverse=True)])
        self.data['unique_words'] = self.data.seg_content_lis.apply(lambda x:set(x))
        self.data.unique_words.apply(DF)
        self.data['KW_by_tfidf'] = self.data.seg_content_lis.apply(TF_iDF)

    def sentiment_analysis(self):
        # Load Semtiment Dictionary
        sentiment_dic = {}
        with open('tools/positive.txt',encoding='utf8') as f1:
            positive_list = list(f1.readlines())
            for word in positive_list:
                sentiment_dic[word.strip()] = 1
        with open('tools/negative.txt',encoding='utf8') as f2:
            negative_list = list(f2.readlines())
            for word in negative_list:
                sentiment_dic[word.strip()] = -1
        # Calculate the Sentiment Scores
        def Calculate_sentiment_score(xs):
            score = 0
            for x in xs:
                if x in sentiment_dic:
                    score += sentiment_dic[x]
            return score
        self.data['sentiment'] = self.data.seg_content_lis.apply(Calculate_sentiment_score)