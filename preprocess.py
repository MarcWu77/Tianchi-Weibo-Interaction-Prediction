import pandas as pd
import numpy as np
import math
import jieba
from chinese_calendar import is_workday

def post_frequency(data,training_month):
    """
    data -> pandas.Dataframe
    training_month -> int
    Functions: Compute the Post Freqency of users by month
    """
    df = pd.DataFrame(data.uid.value_counts(),columns=['uid','freq'])
    df.columns = ['freq','uid']
    df['uid'] = df.index
    data = pd.merge(data,df)
    data['freq_month'] = data['freq']/training_month
    return data

def fans_features(data):
    """
    Parameters:
    data -> pandas.Dataframe 
    Functions: Compute the Fans Score of users
    """

    data['avg_forward'] = data.groupby('uid')['forward_count'].transform('mean')
    data['avg_comment'] = data.groupby('uid')['comment_count'].transform('mean')
    data['avg_like'] = data.groupby('uid')['like_count'].transform('mean')
    
    # set the mask of forward
    L1 = (data.avg_forward<3)
    L2 = (data.avg_forward>=3)&(data.avg_forward<9)
    L3 = (data.avg_forward>=9)&(data.avg_forward<27)
    L4 = (data.avg_forward>=27)&(data.avg_forward<81)
    L5 = (data.avg_forward>=81)&(data.avg_forward<243)
    L6 = (data.avg_forward>=243)&(data.avg_forward<729)
    L7 = (data.avg_forward>=729)&(data.avg_forward<2187)
    L8 = (data.avg_forward>=2187)&(data.avg_forward<6561)
    L9 = (data.avg_forward>=6561)&(data.avg_forward<19683)
    L10 = (data.avg_forward>19683)
    # assign forward level
    data.loc[L1,'forward_level'] = 1
    data.loc[L2,'forward_level'] = 2
    data.loc[L3,'forward_level'] = 3
    data.loc[L4,'forward_level'] = 4
    data.loc[L5,'forward_level'] = 5
    data.loc[L6,'forward_level'] = 6
    data.loc[L7,'forward_level'] = 7
    data.loc[L8,'forward_level'] = 8
    data.loc[L9,'forward_level'] = 9
    data.loc[L10,'forward_level'] = 10

    # set the mask of comment
    L1 = (data.avg_comment<1)
    L2 = (data.avg_comment>=1)&(data.avg_comment<3)
    L3 = (data.avg_comment>=3)&(data.avg_comment<9)
    L4 = (data.avg_comment>=9)&(data.avg_comment<27)
    L5 = (data.avg_comment>=27)&(data.avg_comment<81)
    L6 = (data.avg_comment>=81)&(data.avg_comment<243)
    L7 = (data.avg_comment>=243)&(data.avg_comment<729)
    L8 = (data.avg_comment>=729)&(data.avg_comment<2187)
    L9 = (data.avg_comment>=2187)&(data.avg_comment<6561)
    L10 = (data.avg_comment>=6561)
    # assign comment level
    data.loc[L1,'comment_level'] = 1
    data.loc[L2,'comment_level'] = 2
    data.loc[L3,'comment_level'] = 3
    data.loc[L4,'comment_level'] = 4
    data.loc[L5,'comment_level'] = 5
    data.loc[L6,'comment_level'] = 6
    data.loc[L7,'comment_level'] = 7
    data.loc[L8,'comment_level'] = 8
    data.loc[L9,'comment_level'] = 9
    data.loc[L10,'comment_level'] = 10

    # set the mask of like
    L1 = (data.avg_like<1)
    L2 = (data.avg_like>=1)&(data.avg_like<3)
    L3 = (data.avg_like>=3)&(data.avg_like<9)
    L4 = (data.avg_like>=9)&(data.avg_like<27)
    L5 = (data.avg_like>=27)&(data.avg_like<81)
    L6 = (data.avg_like>=81)&(data.avg_like<243)
    L7 = (data.avg_like>=243)&(data.avg_like<729)
    L8 = (data.avg_like>=729)&(data.avg_like<2187)
    L9 = (data.avg_like>=2187)&(data.avg_like<6561)
    L10 = (data.avg_like>=6561)
    # assign like level
    data.loc[L1,'like_level'] = 1
    data.loc[L2,'like_level'] = 2
    data.loc[L3,'like_level'] = 3
    data.loc[L4,'like_level'] = 4
    data.loc[L5,'like_level'] = 5
    data.loc[L6,'like_level'] = 6
    data.loc[L7,'like_level'] = 7
    data.loc[L8,'like_level'] = 8
    data.loc[L9,'like_level'] = 9
    data.loc[L10,'like_level'] = 10

    df = data[['uid','forward_level','comment_level','like_level']]
    df.drop_duplicates(inplace=True)
    df.to_csv('data/user/user_fans_features.txt',sep=',') # 存儲粉絲特徵

    return data

def time_features(data,start,end):
    """
    Parameters:
    data -> pandas.Dataframe
    start -> str,'yyyy-mm-dd'
    end -> str,'yyyy-mm-dd'
    Functions: Find the post time in Workday/Worktime or not and assign 0/1
    """
    days = pd.date_range(start=start,end=end)
    workdays = [day for day in days if is_workday(day)]
    data['workday'] = np.where(data.time.dt.date.isin(workdays)==1,1,0)
    return data

def explicit_content_features(data):
    """
    Parameters:
    data -> must be pandas.Dataframe 
    Functions: Find 'tag/url/@/emoj' exist or not and assign 0/1
    """
    data['tag'] = np.where(data.content.str.contains(r'#(.{0,30})#')==1,1,0) # add tags 0/1 variable
    data['url'] = np.where(data.content.str.contains(r'(http://[a-zA-z./\d]*)|(https://[a-zA-z./\d]*)')==1,1,0) # add url 0/1 variable
    data['at'] = np.where(data.content.str.contains(r'@([^@]{0,30})\s')==1,1,0) # add @ 0/1 variable
    data['emoj'] = np.where(data.content.str.contains(r'\[.{0,12}\]')==1,1,0) # add emoj 0/1 variable
    return data

def tokenize_content(data):
    """
    Parameters:
    data -> must be pandas.Dataframe 
    Functions: Tokenize and remove stopwords
    """
    # Clean first
    data['content'] = data.content.str.replace(r'(http://[a-zA-z./\d]*)|(https://[a-zA-z./\d]*)','') # remove url
    data['content'] = data.content.str.replace(r'\[.{0,12}\]','') # remove emoj
    data['content'] = data.content.str.replace(r'@([^@]{0,30})\s','') # remove @sb
    data['content'] = data.content.str.replace(r'[^\u4e00-\u9fa5]+','') # remove non-Chinese strings
    data['content'] = data.content.str.replace(r'\uAC00-\uD7AF+','') # remove Korean
    data['content'] = data.content.str.replace(r'\u3040-\u31FF+','') # remove Japanese
    # Tokenization
    with open("tools/stopwords_cn_HIT.txt",encoding='utf8') as f: # Chinese & English stopwords
        stopwords = [stopword.strip() for stopword in f.readlines()]
    def cut_words(x):
        words = jieba.lcut(str(x))
        return [word for word in words if word not in stopwords]
    data['seg_content'] = data.content.apply(cut_words)

def TFIDF(data):
    """
    Parameters:
    data -> must be pandas.Dataframe 
    Functions: Compute TF-iDF rank(document interest/locate keywords) and sum(how unique a document is)
    """
    words_df = {}
    doc_num = len(data)
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
        return dict(sorted(words_tfidf.items(),key=lambda x:x[1],reverse=True))
    data['unique_words'] = data.seg_content.apply(lambda x:set(x)) # 去重
    data.unique_words.apply(DF)
    data['tfidf_rank'] = data.seg_content.apply(TF_iDF)
    data['tfidf_sum'] = data.tfidf_rank.apply(lambda x:sum(x.values()))

def sentiment_analysis(data):
    """
    Parameters:
    data -> must be pandas.Dataframe 
    Functions: Compute the sentiment tendency of a document,named negative(-)/neutral(0)/positive(+)
    """
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
        """
        xs must be iterable
        """
        score = 0
        for x in xs:
            if x in sentiment_dic:
                score += sentiment_dic[x]
        return score
    data['sentiment'] = data.seg_content.apply(Calculate_sentiment_score)



