from selenium.webdriver import Chrome, ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from collections import defaultdict
import time
import random
import urllib.parse
random.seed(1)

abbrev_fn = "../processed_data/preprocess/count_abbrev/count.tsv"
driver = Chrome(ChromeDriverManager().install())
base_url = "https://www.allacronyms.com/"


def get_abbrev(fn, top):
    with open(fn, "r") as fin:
        abbrev_list = list()
        for i, line in enumerate(fin):
            if i > top:
                break
            split_line = line.strip().split("\t")
            abbrev = split_line[0]
            abbrev_list.append(abbrev)
    return abbrev_list

query_list = get_abbrev(abbrev_fn, top=42278)

abbrev = defaultdict(list)
not_found = set()

for query in query_list:
    
    print(f"Query\t{query}")
    if query in abbrev["sf"]:
        print("Query already exists.")
        continue 
    if query in not_found:
        print("Query is not found on allacronyms.com")
        continue

    safe_query = urllib.parse.quote(query, safe="")
    
    url = f"{base_url}{safe_query}"
    driver.get(url)
    sleep = random.randint(2,20)
    time.sleep(sleep)
    try: 
        terms_items = driver.find_element_by_class_name("terms_items")
    except Exception:
        suggest_new = driver.find_elements_by_partial_link_text("Suggest New")
        if suggest_new != []:
            print(f"{query} not found.")
            not_found.add(query)
            continue

        print("Hit bot detector.")
        driver.back()
        time.sleep(60)

        url = f"{base_url}{safe_query}"
        driver.get(url)
        sleep = random.randint(2,20)
        time.sleep(sleep)
        terms_items = driver.find_element_by_class_name("terms_items")

    for i, entry in enumerate(terms_items.text.strip().split("\n")):
        if i % 2 == 0:
            abbrev["sf"].append(entry)
        else:
            abbrev["lf"].append(entry)
    assert i % 2 == 1
