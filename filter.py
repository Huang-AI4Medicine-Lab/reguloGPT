import requests
import nltk
from nltk.tokenize import word_tokenize
import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
# SEARCH_TERM = "m6A OR m(6) OR N6-methyladenosine OR N(6)-methyladenosine AND 2022:3000[Date - Publication]"
SEARCH_TERM = "(\"m6A\" OR \"m(6)A\" OR \"N6-methyladenosine\" OR \"N(6)-methyladenosine\" OR \"N-6-methyladenosine\") AND 2022:3000[Date - " \
              "Publication]"

PUBTATOR_API = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids="


def fetch_pmids(query):
    pmids = []
    retstart = 0
    retmax = 500  # Number of records per request
    while True:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retstart": retstart,
            "retmode": "json",
            "usehistory": "y"
        }
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("Error fetching the results starting from record:", retstart)
            break

        data = response.json()
        fetched_pmids = data.get('esearchresult', {}).get('idlist', [])
        if not fetched_pmids:
            break  # No more PMIDs found, so exit the loop.

        pmids.extend(fetched_pmids)
        retstart += retmax  # Increment the starting record for next batch
    return pmids


def chunker(seq, size):
    # Split the sequence into chunks of given size
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def hit_pubtator_api(pmids_chunk):
    pmids_str = ",".join(pmids_chunk)
    response = requests.get(PUBTATOR_API + pmids_str)
    return response.text if response.status_code == 200 else None

# Title have prognosis, prognostic,correlated, biomarker, analysis
#Corrigendum, Correction
def contains_keywords(title):
    title_lower = title.lower()
    # List of keyword groups where presence of any keyword makes the condition True
    positive_keywords = [
        ["review", "survey"],
        ["predict", "predicted", "predicting", "predicts"],
        ["prognosis", "prognostic", "corrigendum"],
        ["correlated", "biomarker", "analysis"],
        ["correction", "correct"]
    ]
    for keyword_group in positive_keywords:
        if any(keyword in title_lower for keyword in keyword_group):
            return True
    # Keywords that should NOT be in the title
    # negative_keywords = ["via", "by", "through"]
    # if all(keyword not in title_lower for keyword in negative_keywords):
    #     return True
    return False


def title_contains_gene_type(section):
    lines = section.split("\n")
    title_line = [line for line in lines if "|t|" in line]
    if not title_line:
        return False

    title = title_line[0].split("|t|")[-1]
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 5 and parts[3] in title and parts[4] == "Gene" \
                and parts[3] != "m6A" and parts[3] != "N6-methyladenosine" and parts[3] != "N(6)-methyladenosine"\
                and parts[3] != "N-6-methyladenosine":
            #\"m(6)A\" OR \"N6-methyladenosine\" OR \"N(6)-methyladenosine\" OR \"N-6-methyladenosine\"
            return True
    return False
def contains_verbs(title):
    doc = nlp(title)
    for token in doc:
        if "VERB" in token.pos_:
            return True
    return False
def is_complete_sentence(title):
    words = word_tokenize(title)
    pos_tags = nltk.pos_tag(words)

    # Define tags based on Penn Treebank POS tags
    subject_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
    verb_tags = ['VB', 'VBD', 'VBN', 'VBP', 'VBZ']  # Main verb forms
    object_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'DT']
    gerund_or_participle = ['VBG']

    # Check for presence of main verbs and absence of only gerunds/participles
    has_subject = any(tag in subject_tags for _, tag in pos_tags)
    has_main_verb = any(tag in verb_tags for _, tag in pos_tags)
    has_object = any(tag in object_tags for _, tag in pos_tags)
    only_gerund_or_participle = all(tag in gerund_or_participle for _, tag in pos_tags)

    return has_subject and has_main_verb and has_object and not only_gerund_or_participle


if __name__ == "__main__":
    all_pmids = fetch_pmids(SEARCH_TERM)
    print("Total PMIDs fetched:", len(all_pmids))
    number = len(all_pmids)
    filtered_responses = []
    excluded_responses = []
    interesting_response = []
    for chunk in chunker(all_pmids, 100):
        response_data = hit_pubtator_api(chunk)
        if response_data:
            pmid_sections = response_data.strip().split("\n\n")
            for section in pmid_sections:
                lines = section.split("\n")
                if not lines:
                    continue

                # Extract the title
                title_line = [line for line in lines if "|t|" in line]
                if title_line:
                    title = title_line[0].split("|t|")[-1]
                    if not is_complete_sentence(title) or contains_keywords(title) or not contains_verbs(title) or not title_contains_gene_type(section):
                        number -= 1
                        excluded_responses.append(section)
                    else:
                        filtered_responses.append(section)

    # Save filtered responses to a text file
    with open('filtered_pubtator_responses_V4.txt', 'w') as file:
        file.write("\n\n".join(filtered_responses))
    with open('excluded_pubtator_responses_V4.txt', 'w') as file:
        file.write("\n\n".join(excluded_responses))

    print("Total PMIDs remaining:", number)
    print("Filtered responses saved to filtered_pubtator_responses_V4.txt")
    print("Filtered responses saved to exluded_pubtator_responses_V4.txt")
