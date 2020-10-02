from selenium.webdriver import Chrome, ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from collections import defaultdict

driver = Chrome(ChromeDriverManager().install())
base_url = "https://www.allacronyms.com/"
query_list = ["QE", "5-HT"]
abbrev = defaultdict(list)

for query in query_list:
    url = f"{base_url}{query}"
    driver.get(url)
    terms_items = driver.find_element_by_class_name("terms_items")

    for i, entry in enumerate(terms_items.text.strip().split("\n")):
        if i % 2 == 0:
            abbrev["sf"].append(entry)
        else:
            abbrev["lf"].append(entry)
    assert i % 2 == 1
