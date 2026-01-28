from gensim import corpora, models

def extract_topics(
    normalized_comments: list[str],
    num_topics: int = 4,
    words_per_topic: int = 5,
):
    texts = [c.split() for c in normalized_comments if c.strip()]
    if len(texts) < 5:
        return []

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42,
    )

    topics = []
    for i in range(num_topics):
        words = [w for w, _ in lda.show_topic(i, words_per_topic)]
        topics.append({
            "topic_id": i,
            "keywords": words
        })

    return topics
