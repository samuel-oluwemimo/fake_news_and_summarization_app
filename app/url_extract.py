from newspaper import Article

def extract_article_details(url: str) -> dict:
    """
    Extracts the title and content from a news article URL.

    Args:
        url (str): The URL of the news article.

    Returns:
        dict: A dictionary containing the title and content of the article.
    """
    article = Article(url)
    article.download()
    article.parse()

    return {
        'title': article.title,
        'content': article.text
    }
