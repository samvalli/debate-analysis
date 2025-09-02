import requests
from urllib.parse import urlparse, unquote

BASE_URL= "https://en.wikiversity.org/w/api.php"

# Function to extract the title from the URL
def extract_title_from_url(url):
    path = urlparse(url).path
    title = path.split('/')[-1]
    return unquote(title)

# Function to fetch the content of a category page on Wikiversity
def fetch_wikiversity_category_content(title):
    
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": title,
        "cmlimit": "max",  # Fetch as many members as possible (max is 500)
        "format": "json"
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Extracting category members
    members = data.get('query', {}).get('categorymembers', [])
    
    # Formatting the output
    content = "\n".join([member['title'] for member in members])
    
    return content

def get_all_wikidebate_titles():
    wiki_url= "https://en.wikiversity.org/wiki/Category:Wikidebates"
    title = extract_title_from_url(wiki_url)
    content = fetch_wikiversity_category_content(title)
    titles=[]
    title=" "
    for ch in content[41:]:
        if ch=='\n':
            titles.append(title.lstrip())
            title=" "
            continue
        title+=ch
    return titles