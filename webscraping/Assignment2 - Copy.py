import pandas as pd
import requests
from bs4 import BeautifulSoup
import textstat
from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# load input file
input_file = pd.read_excel('input.xlsx')

# define a function to extract article text from a given URL
def extract_article_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # find the article title
            article_title = soup.find('title').text.strip()
            # find the article text
            article_text = ''
            for paragraph in soup.find_all('p'):
                article_text += paragraph.text.strip() + '\n'
            return article_title, article_text
        else:
            print(f"Error: {response.status_code} - Could not retrieve {url}")
            return None, None
    except:
        print(f"Error: Could not retrieve {url}")
        return None, None

# loop over input file rows and extract article text
output_data = []
for index, row in input_file.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    article_title, article_text = extract_article_text(url)
    if article_title and article_text:
        # write article text to file
        with open(f"{url_id}.txt", 'w', encoding='utf-8') as file:
            file.write(article_title + '\n\n' + article_text)
            print(f"Saved article {url_id}.txt")
        # calculate text stats
        word_count = textstat.lexicon_count(article_text)
        sentence_count = textstat.sentence_count(article_text)
        syllable_count = textstat.syllable_count(article_text)
        flesch_reading_ease = textstat.flesch_reading_ease(article_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(article_text)
        smog_index = textstat.smog_index(article_text)
        gunning_fog = textstat.gunning_fog(article_text)
        automated_readability_index = textstat.automated_readability_index(article_text)
        # calculate sentiment score
        blob = TextBlob(article_text)
        polarity_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity
        
        # calculate additional text stats
        avg_sentence_length = round(word_count / sentence_count, 2)
        complex_word_count = textstat.difficult_words(article_text)
        percentage_complex_words = round(100 * complex_word_count / word_count, 2)
        fog_index = round(0.4 * (avg_sentence_length + percentage_complex_words), 2)
        avg_words_per_sentence = round(word_count / sentence_count, 2)
        syllable_per_word = round(syllable_count / word_count, 2)
        tokens = word_tokenize(article_text)
        tagged_tokens = pos_tag(tokens)
        personal_pronouns = sum(1 for word, pos in tagged_tokens if pos == 'PRP' or pos == 'PRP$')

        #personal_pronouns = textstat.personal_pronoun_count(article_text)
        #avg_word_length = round(textstat.avg_word_length(article_text), 2)
        words = [word for word in tokens if word.isalnum()]
        avg_word_length = round(sum(len(word) for word in words) / len(words), 2)

        
        # append output data
        output_data.append([url_id, url, polarity_score, 1 - polarity_score, polarity_score, 
                            subjectivity_score, avg_sentence_length, percentage_complex_words, 
                            fog_index, avg_words_per_sentence, complex_word_count, word_count, 
                            syllable_per_word, personal_pronouns, avg_word_length])
    else:
        # write blank file
        with open(f"{url_id}.txt", 'w', encoding='utf-8') as file:
            print(f"Saved blank file {url_id}.txt")

# create
output_columns = ['URL_ID', 'URL', 'Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score',
                  'Avg_Sentence_Length', 'Percentage_of_Complex_Words', 'FOG_Index', 'Avg_Number_of_Words_per_Sentence',
                  'Complex_Word_Count', 'Word_Count', 'Syllable_per_Word', 'Personal_Pronouns', 'Avg_Word_Length']

output_file = pd.DataFrame(output_data, columns=output_columns)
output_file.to_csv('output.csv', index=False)
